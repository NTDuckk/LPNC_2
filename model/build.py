import copy
from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights,tokenize
import torch
import torch.nn as nn
from .triplet_loss import TripletLoss
from .supcontrast import SupConLoss
from torch.cuda.amp import autocast


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class IM2TEXT(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x  = self.bottleneck(x)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        outputs = self.transformer([x])
        x = outputs[0]
        att = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD   # x,att
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_feature = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_feature



class LPNC(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        self.classifier_proj = nn.Linear(self.embed_dim, self.num_classes, bias=False)
        self.bottleneck_proj = nn.BatchNorm1d(self.embed_dim)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=args.cmt_depth,
                                                   heads=self.embed_dim //
                                                         64)
        scale = self.cross_modal_transformer.width ** -0.5
        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)
        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.text_encoder = TextEncoder(copy.deepcopy(self.base_model))
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.img2text = IM2TEXT(embed_dim=512,
                                middle_dim=512,
                                output_dim=512,
                                n_layer=2)
        dataset_name = args.dataset_name
        self.W = nn.Parameter(torch.eye(512))
        self.num_attr_queries = 64   # n: learnable query embeddings for attribute extraction
        self.num_attr_select = 16    # k: number of selected attribute tokens
        self.prompt_learner = PromptLearner(num_classes, dataset_name, self.base_model.dtype,
                                            self.base_model.token_embedding,
                                            num_attr_tokens=self.num_attr_select)

        # CGI Local: learnable queries + cross-attention on patch tokens for S*_local
        self.cgi_num_queries = 2
        self.cgi_queries = nn.Parameter(torch.randn(self.cgi_num_queries, self.embed_dim) * 0.02)
        self.cgi_ln_q = LayerNorm(self.embed_dim)
        self.cgi_ln_kv = LayerNorm(self.embed_dim)
        self.cgi_local_cross_attn = nn.MultiheadAttention(
            self.embed_dim, self.embed_dim // 64, batch_first=True
        )
        self.cgi_ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )
        self.cgi_local_map = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # IADT Attribute-aware tokens: extraction, filtering, mapping
        self.attr_queries = nn.Parameter(torch.randn(self.num_attr_queries, self.embed_dim) * 0.02)
        # 3-layer single-head Transformer for attribute extraction (self-attention on [queries | patches])
        self.attr_transformer = Transformer(width=self.embed_dim, layers=3, heads=1)
        attr_scale = self.attr_transformer.width ** -0.5
        attr_proj_std = attr_scale * ((2 * self.attr_transformer.layers) ** -0.5)
        attr_attn_std = attr_scale
        attr_fc_std = (2 * self.attr_transformer.width) ** -0.5
        for block in self.attr_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attr_attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=attr_proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=attr_fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=attr_proj_std)
        # FC for dimension alignment after Transformer
        self.attr_fc = nn.Linear(self.embed_dim, self.embed_dim)
        # Attribute mapping network f_A (3-layer MLP)
        self.attr_map = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x[0].unsqueeze(0)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_image1(self, image):
        x, _ = self.base_model.encode_image(image)
        return x.float()

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_promptsg(self, image):
        """PromptSG-style inference: image -> cross_x_bn features.
        S* = img2text(CLS) only (no S*_local, no caption multiplication).
        A*_k attribute tokens generated from raw patch tokens.
        """
        image_feats, _ = self.base_model.encode_image(image)
        i_feats = image_feats[:, 0, :].float()         # CLS token [B, 512]
        patch_tokens = image_feats[:, 1:, :].float()    # patches [B, M, 512]
        B_size = i_feats.shape[0]

        # S* = S*_global from inversion network only
        s_star = self.img2text(i_feats.half())           # [B, 512]

        # IADT Attribute extraction
        attr_q = self.attr_queries.unsqueeze(1).expand(-1, B_size, -1)   # [n, B, 512]
        patch_lnd = patch_tokens.permute(1, 0, 2)                        # [M, B, 512]
        concat_input = torch.cat([attr_q.type_as(patch_lnd), patch_lnd], dim=0)
        attr_out = self.attr_transformer([concat_input])
        if isinstance(attr_out, (tuple, list)):
            attr_out = attr_out[0]
        attr_feats = attr_out[:self.num_attr_queries].permute(1, 0, 2)   # [B, n, 512]
        X_prime = self.attr_fc(attr_feats)

        # Top-k filtering by cosine similarity with CLS
        X_prime_norm = X_prime / (X_prime.norm(dim=-1, keepdim=True) + 1e-8)
        global_norm = i_feats.unsqueeze(1) / (i_feats.unsqueeze(1).norm(dim=-1, keepdim=True) + 1e-8)
        sim_scores = (global_norm * X_prime_norm).sum(dim=-1)
        topk_indices = sim_scores.topk(self.num_attr_select, dim=-1).indices
        W_attr = torch.gather(X_prime, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Attribute mapping
        attr_tokens = self.attr_map(W_attr)              # [B, k, 512]

        # Prompt -> TextEncoder -> CrossFormer -> BN
        prompts = self.prompt_learner(s_star, attr_tokens)
        text_feature = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
        cross_x = self.cross_former(text_feature.unsqueeze(1), image_feats, image_feats)
        cross_x_bn = self.bottleneck_proj(cross_x.squeeze(1))

        return cross_x_bn.float()

    def forward(self, batch):
        ret = dict()
        device = "cuda"
        triplet = TripletLoss(margin=0.3)
        supcon = SupConLoss(device)

        ret.update({'temperature': 1 / self.logit_scale})
        logit_scale = self.logit_scale
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        cross_x_bn = None  # set by supid branch, used by cotrl

        if 'supid' in self.current_task:
            # Element-wise multiplication: image features x text features
            refined_cls = i_feats * t_feats                                      # CLS x text EOT: [B, 512]
            refined_patch = image_feats[:, 1:, :].float() * t_feats.unsqueeze(1) # patch x text EOT: [B, M, 512]

            _half = self.cgi_local_cross_attn.in_proj_weight.dtype  # half from convert_weights
            token_features = self.img2text(i_feats.to(_half))  # S*_global: [B, 512]
            with autocast():
                # CGI Local: compute S*_local from image patch tokens
                patch_tokens = refined_patch.to(_half)  # [B, M, 512] cast to half for modules
                B_size = patch_tokens.shape[0]
                queries = self.cgi_queries.unsqueeze(0).expand(B_size, -1, -1).to(_half)
                q_ln = self.cgi_ln_q(queries).to(_half)
                kv_ln = self.cgi_ln_kv(patch_tokens).to(_half)
                attn_out, _ = self.cgi_local_cross_attn(q_ln, kv_ln, kv_ln)  # [B, K, 512]
                attn_out = queries + attn_out  # residual connection (both half)
                P = attn_out + self.cgi_ffn(attn_out)  # FFN with residual: [B, K, 512]
                s_local = self.cgi_local_map(P.mean(dim=1))  # Avg pool + MLP: [B, 512]

                # Combine S*_global + S*_local
                s_star = token_features + s_local

                # --- IADT Attribute-aware tokens ---
                # 1) Attribute extraction: concat [queries | patches] -> Transformer -> FC
                attr_q = self.attr_queries.unsqueeze(1).expand(-1, B_size, -1).to(_half)  # [n, B, 512]
                patch_lnd = patch_tokens.permute(1, 0, 2)  # [M, B, 512] LND (already half)
                concat_input = torch.cat([attr_q, patch_lnd], dim=0)  # [n+M, B, 512]
                attr_out = self.attr_transformer([concat_input])  # [n+M, B, 512]
                if isinstance(attr_out, (tuple, list)):
                    attr_out = attr_out[0]
                attr_feats = attr_out[:self.num_attr_queries].permute(1, 0, 2)  # [B, n, 512]
                X_prime = self.attr_fc(attr_feats)  # [B, n, 512]

                # 2) Attribute filtering: cosine similarity with global feat, select top-k
                X_prime_norm = X_prime / (X_prime.norm(dim=-1, keepdim=True) + 1e-8)
                global_feat = i_feats.unsqueeze(1).to(X_prime.dtype)  # [B, 1, 512]
                global_norm = global_feat / (global_feat.norm(dim=-1, keepdim=True) + 1e-8)
                sim_scores = (global_norm * X_prime_norm).sum(dim=-1)  # [B, n]
                topk_indices = sim_scores.topk(self.num_attr_select, dim=-1).indices  # [B, k]
                W_attr = torch.gather(X_prime, 1,
                    topk_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))  # [B, k, 512]

                # 3) Orthogonal loss: L_ortho = ||WW^T - I||^2_F
                ortho_matrix = torch.bmm(W_attr, W_attr.transpose(1, 2))  # [B, k, k]
                identity = torch.eye(self.num_attr_select, device=ortho_matrix.device).unsqueeze(0)
                L_ortho = ((ortho_matrix - identity) ** 2).sum(dim=(1, 2)).mean()

                # 4) Attribute mapping: f_A(W) -> attribute pseudo-word tokens
                attr_tokens = self.attr_map(W_attr)  # [B, k, 512]

                # Construct IADT prompt: "a [S*] person with [A*_1]...[A*_k]"
                prompts = self.prompt_learner(s_star, attr_tokens)
                text_feature = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

                cross_x = self.cross_former(text_feature.unsqueeze(1), image_feats, image_feats)
                cross_x_bn = self.bottleneck_proj(cross_x.squeeze(1))
                cls_score = self.classifier_proj(cross_x_bn.half()).float()
            supcon_loss = supcon(i_feats, text_feature.float(), batch['pids'], batch['pids']) + supcon(text_feature.float(),i_feats,batch['pids'],batch['pids'])
            cross_id_loss = objectives.compute_id(cls_score, batch['pids'])
            ret.update({'supid_loss': self.args.lambda1_weight * supcon_loss + self.args.lambda2_weight * cross_id_loss + L_ortho})

        if 'cotrl' in self.current_task:
            # Triplet loss on cross-attention features (same features used for ID loss in supid)
            triloss, _, _ = triplet(cross_x_bn.float(), batch['pids'])
            ret.update({'cotrl_loss': triloss})

        return ret


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding, num_attr_tokens=16):
        super().__init__()
        self.num_attr_tokens = num_attr_tokens
        # IADT-style template: "a [S*] person with [A*_1] [A*_2] ... [A*_k]"
        attr_placeholder = " X" * num_attr_tokens
        ctx_init = "a X person with" + attr_placeholder
        n_ctx = 1  # context words before S*: just "a"
        tokenized_prompts = tokenize(ctx_init).cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenized_prompts = tokenized_prompts.to(device)
        token_embedding = token_embedding.to(device)
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # Positions: [SOS, "a", S*, "person", "with", A*_1, ..., A*_k, EOS, PAD...]
        # prefix: [SOS, "a"] -> [:n_ctx+1] = [:2]
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        # middle: ["person", "with"] -> [n_ctx+2 : n_ctx+4] = [3:5]
        self.register_buffer("token_middle", embedding[:, n_ctx + 2:n_ctx + 4, :])
        # suffix: [EOS, PAD, ...] -> [n_ctx+4+k:]
        self.register_buffer("token_suffix", embedding[:, n_ctx + 4 + num_attr_tokens:, :])
        self.num_class = num_class
        self.token_ = token_embedding.to(device)
        self.dtype = dtype

    def forward(self, s_star, attr_tokens=None):
        # s_star: [B, embed_dim] - subject-oriented pseudo-word token
        # attr_tokens: [B, k, embed_dim] - attribute-aware pseudo-word tokens (optional for simplified inference)
        B = s_star.shape[0]
        prefix = self.token_prefix.expand(B, -1, -1)   # [B, 2, d]
        middle = self.token_middle.expand(B, -1, -1)    # [B, 2, d]
        suffix = self.token_suffix.expand(B, -1, -1)    # [B, *, d]
        s_star = s_star.unsqueeze(1)                     # [B, 1, d]
        if attr_tokens is None:
            # Simplified inference: zero attribute tokens
            d = s_star.shape[-1]
            attr_tokens = torch.zeros(B, self.num_attr_tokens, d,
                                      dtype=s_star.dtype, device=s_star.device)
        # [SOS, "a"] + [S*] + ["person", "with"] + [A*_1...A*_k] + [EOS, PAD...]
        prompts = torch.cat([prefix, s_star, middle, attr_tokens, suffix], dim=1)
        return prompts

def build_model(args, num_classes=11003):
    model = LPNC(args, num_classes)
    if not getattr(args, 'fp16', False):
        # manual fp16 weights -- skip when using AMP (autocast handles fp16 automatically)
        convert_weights(model)
    return model

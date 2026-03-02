import copy

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, tokenize
from .triplet_loss import TripletLoss
from .supcontrast import SupConLoss


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


class IM2TEXT(nn.Module):
    """global image embedding -> pseudo token"""
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            layers.append(nn.Sequential(
                nn.Linear(dim, middle_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            ))
            dim = middle_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


class TextEncoder(nn.Module):
    """Frozen CLIP text transformer (copied from CLIP)."""
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
        x = outputs[0].permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        text_feature = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_feature


class LocalPseudoWord(nn.Module):
    """SCGI-style local pseudo-word extraction.

    Learnable queries attend to patch features via cross-attention + FFN,
    then average-pooled and mapped to a pseudo-token (S*_local).
    """
    def __init__(self, embed_dim=512, num_queries=2, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        self.ln_q = LayerNorm(embed_dim)
        self.ln_kv = LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ln_ffn = LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Mapping network fMl: 3-layer MLP (same depth as SCGI paper)
        self.mapping = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """Args:
            patch_feats: [B, M, D]  (local visual tokens, already refined)
        Returns:
            s_local: [B, D]  pseudo-word embedding
        """
        B = patch_feats.shape[0]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)     # [B, K, D]

        # Cross-attention: queries attend to patch features
        q_normed = self.ln_q(Q)
        kv_normed = self.ln_kv(patch_feats)
        qc = Q + self.cross_attn(q_normed, kv_normed, kv_normed, need_weights=False)[0]  # [B, K, D]

        # Feed-forward
        P = qc + self.ffn(self.ln_ffn(qc))   # [B, K, D]

        # Average pool over K queries -> mapping network
        avg_p = P.mean(dim=1)                 # [B, D]
        s_local = self.mapping(avg_p)          # [B, D]
        return s_local


class PromptLearner(nn.Module):
    """Template: 'A photo of a X person' (insert ONE pseudo-token at X)."""
    def __init__(self, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X person"
        n_ctx = 4

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenized_prompts = tokenize(ctx_init).to(device)
        token_embedding = token_embedding.to(device)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + 1:, :])
        self.dtype = dtype

    def forward(self, bias: torch.Tensor):
        b = bias.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        bias = bias.unsqueeze(1)
        return torch.cat([prefix, bias, suffix], dim=1)


class LPNC(nn.Module):
    """
    - Keep ONLY supid branch
    - Keep caption t_feats fusion
    - Add Triplet on cross-attention feature (PromptSG-style)
    - Remove token selection / cotrl / cid
    """
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes

        self.current_task = ["supid"]
        print("Training Model with ['supid'] only (cotrl/cid/token-selection removed).")

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        self.classifier_proj = nn.Linear(self.embed_dim, self.num_classes, bias=False)
        self.bottleneck_proj = nn.BatchNorm1d(self.embed_dim)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64, batch_first=True)
        self.cross_modal_transformer = Transformer(width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64)

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
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.text_encoder = TextEncoder(copy.deepcopy(self.base_model))
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.img2text = IM2TEXT(embed_dim=512, middle_dim=512, output_dim=512, n_layer=2)       # fMg (S*_global)
        self.local_pseudo = LocalPseudoWord(embed_dim=self.embed_dim, num_queries=2,              # S*_local (SCGI)
                                            num_heads=self.embed_dim // 64, ffn_dim=self.embed_dim * 4)
        self.W = nn.Parameter(torch.eye(512))
        self.prompt_learner = PromptLearner(self.base_model.dtype, self.base_model.token_embedding)

    def cross_former(self, q, k, v):
        x = self.cross_attn(self.ln_pre_t(q), self.ln_pre_i(k), self.ln_pre_i(v), need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x[0].unsqueeze(0).permute(1, 0, 2)  # LND -> NLD
        return self.ln_post(x)

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_image1(self, image):
        x, _ = self.base_model.encode_image(image)
        return x.float()

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = {}
        device = batch['images'].device

        supcon = SupConLoss(str(device).split(":")[0])
        triplet = TripletLoss(margin=getattr(self.args, "triplet_margin", 0.3))

        ret['temperature'] = 1 / self.logit_scale

        images = batch['images']
        caption_ids = batch['caption_ids']

        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)

        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()  # [B, D]

        # ---- SCGI-style caption-guided refinement (Eq.3: ṽ = v ⊙ t) ----
        refined_image_feats = image_feats * t_feats.unsqueeze(1)               # [B, M+1, D] element-wise
        i_feats = refined_image_feats[:, 0, :].float()                         # [B, D] CLS token

        # S*_global  (from refined CLS token)
        token_features = self.img2text(i_feats.half())                         # [B, D]

        # S*_local   (use refined patch tokens from refined_image_feats, then project with W)
        # ensure patch_feats has same dtype as parameter `W` to avoid dtype mismatch
        patch_feats = refined_image_feats[:, 1:, :].to(self.W.dtype)           # [B, M, D]
        local_features = self.local_pseudo(patch_feats @ self.W)               # [B, D]

        # S* = S*_global + S*_local
        pseudo_token = token_features + local_features.half()                   # [B, D]

        with autocast():
            prompts = self.prompt_learner(pseudo_token)                         # no W here anymore
            text_feature = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

            cross_x = self.cross_former(text_feature.unsqueeze(1), image_feats, image_feats)
            cross_x_bn = self.bottleneck_proj(cross_x.squeeze(1))

            cls_score = self.classifier_proj(cross_x_bn.half()).float()

        supcon_loss = (
            supcon(i_feats, text_feature.float(), batch['pids'], batch['pids'])
            + supcon(text_feature.float(), i_feats, batch['pids'], batch['pids'])
        )
        id_loss = objectives.compute_id(cls_score, batch['pids'])

        # Triplet on feature AFTER cross-attn (PromptSG style overall objective)
        tri_loss, _, _ = triplet(cross_x_bn.float(), batch['pids'])

        ret['supcon_loss'] = supcon_loss
        ret['id_loss'] = id_loss
        ret['triplet_loss'] = tri_loss

        ret['supid_loss'] = (
            self.args.lambda1_weight * supcon_loss
            + self.args.lambda2_weight * id_loss
            + getattr(self.args, "lambda3_weight", 1.0) * tri_loss
        )
        return ret

def build_model(args, num_classes=11003):
    model = LPNC(args, num_classes)
    convert_weights(model)
    return model
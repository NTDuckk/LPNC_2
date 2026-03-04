import copy

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F

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


class LocalMappingNetwork(nn.Module):
    """Simple mapping network to convert averaged local queries -> S*_local.

    This mirrors `IM2TEXT` style: a small MLP that maps a [B, D] vector
    to the pseudo-word embedding space [B, D]. The cross-attention and
    query handling are moved into `LPNC`.
    """
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
    - Keep caption t_feats fusion
    - Add Triplet on cross-attention feature (PromptSG-style)
    - Remove token selection / cotrl / cid
    """
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes


        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        # Keep base_model text-related parameters trainable so caption-based
        # features (from `encode_text`) can be fine-tuned during training.
        # The separate `self.text_encoder` (a frozen copy used for prompts)
        # will remain frozen below.
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
        # Freeze the copied text encoder as well (no gradients)
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.img2text = IM2TEXT(embed_dim=512, middle_dim=512, output_dim=512, n_layer=2)       # fMg (S*_global)
        # Local SCGI-style components: queries + cross-attn + FFN are kept in LPNC.
        # Learnable local queries (moved from LocalPseudoWord)
        self.local_num_queries = 2
        self.local_queries = nn.Parameter(torch.randn(self.local_num_queries, self.embed_dim) * 0.02)

        # Layer norms and cross-attention for local queries attending to patch tokens
        self.ln_q_local = LayerNorm(self.embed_dim)
        self.ln_kv_local = LayerNorm(self.embed_dim)
        self.cross_attn_local = nn.MultiheadAttention(self.embed_dim, self.embed_dim // 64, batch_first=True)
        # initialize local cross-attention weights now that the module exists
        nn.init.normal_(self.cross_attn_local.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn_local.out_proj.weight, std=proj_std)

        # Local FFN (SCGI-style)
        self.ln_ffn_local = LayerNorm(self.embed_dim)
        self.ffn_local = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(0.1),
        )

        # Mapping network that consumes averaged local outputs -> S*_local
        self.local_mapping = LocalMappingNetwork(embed_dim=self.embed_dim, middle_dim=self.embed_dim,
                                                output_dim=self.embed_dim, n_layer=3)

        self.prompt_learner = PromptLearner(self.base_model.dtype, self.base_model.token_embedding)

    def cross_former(self, q, k, v):
        x = self.cross_attn(self.ln_pre_t(q), self.ln_pre_i(k), self.ln_pre_i(v), need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x[0].unsqueeze(0).permute(1, 0, 2)  # LND -> NLD
        return self.ln_post(x)

    def cross_former_local(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """SCGI-style local cross-attention + FFN. Returns averaged pooled vector [B, D].

        Args:
            patch_feats: [B, M, D] local visual tokens (refined)
        Returns:
            avg_p: [B, D] averaged pooled output over K learnable queries
        """
        B = patch_feats.shape[0]
        Q = self.local_queries.unsqueeze(0).expand(B, -1, -1)     # [B, K, D]

        q_normed = self.ln_q_local(Q)
        kv_normed = self.ln_kv_local(patch_feats)
        qc = Q + self.cross_attn_local(q_normed, kv_normed, kv_normed, need_weights=False)[0]  # [B, K, D]

        P = qc + self.ffn_local(self.ln_ffn_local(qc))   # [B, K, D]
        avg_p = P.mean(dim=1)  # [B, D]
        return avg_p

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
        img_feats = image_feats[:, 0, :].float()                               # [B, D] CLS token
        # ---- SCGI-style caption-guided refinement (Eq.3: ṽ = v ⊙ t) ----
        refined_image_feats = image_feats * t_feats.unsqueeze(1)               # [B, M+1, D] element-wise
        i_feats = refined_image_feats[:, 0, :].float()                         # [B, D] CLS token

        # S*_global  (from refined CLS token)
        token_features = self.img2text(i_feats)                                # [B, D]

            # S*_local   (use refined patch tokens from refined_image_feats)
        patch_feats = refined_image_feats[:, 1:, :].float()                    # [B, M, D]

        # Local SCGI cross-attention: compute averaged pooled local representation in LPNC
        avg_p = self.cross_former_local(patch_feats)                             # [B, D]
        local_features = self.local_mapping(avg_p)                               # [B, D]

        # S* = S*_global + S*_local
        pseudo_token = token_features + local_features                         # [B, D]

        with autocast():
            prompts = self.prompt_learner(pseudo_token)                         # no W here anymore
            text_feature = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)

            cross_x = self.cross_former(text_feature.unsqueeze(1), image_feats, image_feats)
            cross_x_bn = self.bottleneck_proj(cross_x.squeeze(1))

            cls_score = self.classifier_proj(cross_x_bn)

        # L2-normalize features before SupConLoss as requested
        img_feats_norm = F.normalize(img_feats, p=2, dim=1)
        text_feature_norm = F.normalize(text_feature.float(), p=2, dim=1)

        supcon_loss = (
            supcon(img_feats_norm, text_feature_norm, batch['pids'], batch['pids'])
            + supcon(text_feature_norm, img_feats_norm, batch['pids'], batch['pids'])
        )
        id_loss = objectives.compute_id(cls_score, batch['pids'])

        # Triplet on feature AFTER cross-attn (PromptSG style overall objective)
        tri_loss, _, _ = triplet(cross_x_bn.float(), batch['pids'])

        ret['supcon_loss'] = supcon_loss
        ret['id_loss'] = id_loss
        ret['triplet_loss'] = tri_loss

        # Combined loss is intentionally not returned; keep individual component losses only
        return ret

def build_model(args, num_classes=11003):
    model = LPNC(args, num_classes)
    # FP32 mode: keep all parameters and buffers in float32 for training stability.
    # GradScaler in the processor handles mixed-precision efficiency.
    model.float()
    # dtype is a plain Python attribute (not a buffer/parameter), so model.float() does
    # not update it. Override manually so internal .type(self.dtype) calls stay FP32.
    model.text_encoder.dtype = torch.float32
    model.prompt_learner.dtype = torch.float32
    return model
"""
visualize_xai.py
================
XAI visualization cho mô hình LPNC (CLIP ViT-B/16 backbone).

Hỗ trợ 3 phương pháp:
  1. last_attn   – attention từ lớp cuối (CLS → patches)
  2. rollout     – Attention Rollout qua tất cả các lớp ViT
  3. gradcam     – Grad-CAM trên feature map của lớp cuối ViT

Cách dùng
---------
  # 1 ảnh, tất cả 3 phương pháp
  python visualize_xai.py --img path/to/person.jpg --checkpoint run_logs/baseline/best.pth

  # Thư mục ảnh -> lưu từng kết quả
  python visualize_xai.py --img_dir data/query/ --checkpoint run_logs/baseline/best.pth \\
                           --method rollout --save_dir xai_outputs/

  # Không có checkpoint (xem raw CLIP attention pattern)
  python visualize_xai.py --img path/to/person.jpg --no_checkpoint
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─────────────────────────────────────────────────────────────────────────────
# helpers để import module gốc của project
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def _make_args_for_build():
    """Tạo namespace args đơn giản đủ để build model."""
    import types
    a = types.SimpleNamespace()
    a.pretrain_choice = "ViT-B/16"
    a.img_size        = (384, 128)
    a.stride_size     = 16
    a.temperature     = 0.02
    a.cmt_depth       = 2
    a.MLM             = False
    a.masked_token_rate           = 0.8
    a.masked_token_unchanged_rate = 0.1
    a.lr_factor       = 5.0
    a.triplet_margin  = 0.3
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Image pre-processing
# ─────────────────────────────────────────────────────────────────────────────
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]

def load_image(path: str, img_size=(384, 128)) -> Tuple[torch.Tensor, np.ndarray]:
    """Đọc ảnh, trả về (tensor CHW float16, numpy HWC uint8 gốc)."""
    img = Image.open(path).convert("RGB")
    orig = np.array(img.resize((img_size[1], img_size[0]), Image.BILINEAR))  # (H,W,3)

    # Normalize
    t = torch.from_numpy(orig).float() / 255.0          # (H,W,3)
    t = t - torch.tensor(MEAN)
    t = t / torch.tensor(STD)
    t = t.permute(2, 0, 1).unsqueeze(0)                 # (1,3,H,W)
    return t, orig


# ─────────────────────────────────────────────────────────────────────────────
# Patch grid info
# ─────────────────────────────────────────────────────────────────────────────
def patch_grid(img_size=(384, 128), patch_size=16, stride_size=16):
    """Trả về (num_y, num_x) = (height_patches, width_patches)."""
    num_x = (img_size[1] - patch_size) // stride_size + 1
    num_y = (img_size[0] - patch_size) // stride_size + 1
    return num_y, num_x


# ─────────────────────────────────────────────────────────────────────────────
# Hook-based attention collector
# ─────────────────────────────────────────────────────────────────────────────
class AttentionCollector:
    """
    Đăng ký hook vào tất cả ResidualAttentionBlock trong visual transformer.
    Sau forward pass, self.attentions chứa list [L x (B, heads, N, N)].
    """
    def __init__(self, vit_transformer):
        self.attentions = []
        self._hooks     = []
        for block in vit_transformer.resblocks:
            h = block.attn.register_forward_hook(self._hook)
            self._hooks.append(h)

    def _hook(self, module, inputs, output):
        # CLIP's ResidualAttentionBlock calls attention with need_weights=False,
        # so output[1] is None. Re-run with need_weights=True to get attention weights.
        attn_w = output[1]
        if attn_w is None:
            with torch.no_grad():
                q, k, v = inputs[0], inputs[1], inputs[2]
                _, attn_w = module(q, k, v,
                                   need_weights=True,
                                   average_attn_weights=True)
        self.attentions.append(attn_w.detach().cpu().float())

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 – Last-layer attention (CLS → patches)
# ─────────────────────────────────────────────────────────────────────────────
def last_layer_attention(attn_list: List[torch.Tensor]) -> np.ndarray:
    """
    Lấy attention map của lớp cuối, hàng CLS (index 0), loại bỏ CLS→CLS.
    Trả về (N_patches,) array.
    """
    last = attn_list[-1]  # (B, N, N)
    cls_attn = last[0, 0, 1:]  # (N_patches,)  -- lớp cuối, ảnh 0, CLS→patches
    cls_attn = cls_attn.numpy()
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    return cls_attn


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 – Attention Rollout
# ─────────────────────────────────────────────────────────────────────────────
def attention_rollout(attn_list: List[torch.Tensor],
                      discard_ratio: float = 0.9) -> np.ndarray:
    """
    Attention Rollout (Abnar & Zuidema 2020).
    attn_list: list of (B, N, N) – mỗi phần tử là một lớp.
    discard_ratio: loại bỏ phần nhỏ nhất trước khi tích lũy.
    Trả về (N_patches,) bản đồ attention từ CLS.
    """
    N = attn_list[0].shape[-1]                # N = num_patches + 1
    result = torch.eye(N, N)                  # identity

    for attn in attn_list:
        a = attn[0].float()                   # (N, N) – ảnh 0

        # Giữ lại proportion lớn nhất, bỏ nhỏ (discard_ratio)
        flat = a.view(-1)
        threshold = flat.quantile(discard_ratio)
        a = torch.where(a > threshold, a, torch.zeros_like(a))

        # Thêm residual connection và chuẩn hóa theo cột
        a = a + torch.eye(N)
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
        result = a @ result

    # CLS row (index 0), loại bỏ CLS→CLS (index 0)
    cls_attn = result[0, 1:].numpy()          # (N_patches,)
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    return cls_attn


# ─────────────────────────────────────────────────────────────────────────────
# Method 3 – Grad-CAM trên lớp cuối của ViT
# ─────────────────────────────────────────────────────────────────────────────
class GradCAMViT:
    """
    Gradient-weighted Class Activation Mapping cho ViT.
    Target: output của lớp cuối visual transformer (feature sau residual + MLP).
    Gradient: của L2 norm CLS feature w.r.t. patch features.
    """
    def __init__(self, model):
        self.model    = model
        self._feats   = None
        self._grads   = None
        self._hooks   = []
        # Lớp cuối của visual transformer
        last_block = model.base_model.visual.transformer.resblocks[-1]

        # Hook forward để lấy output patch features
        def fwd_hook(module, inp, outp):
            # outp là list [x, attn_weight] từ ResidualAttentionBlock.forward
            x_out = outp[0]                            # (L, B, D)  – LND
            self._feats = x_out.permute(1, 0, 2)       # (B, L, D)  – NLD

        # Hook backward để lấy gradient
        def bwd_hook(module, grad_in, grad_out):
            g = grad_out[0]                            # (L, B, D)  – LND
            self._grads = g.permute(1, 0, 2)           # (B, L, D)  – NLD

        self._hooks.append(last_block.register_forward_hook(fwd_hook))
        self._hooks.append(last_block.register_full_backward_hook(bwd_hook))

    def compute(self, image_tensor: torch.Tensor,
                device: str = "cuda") -> np.ndarray:
        """
        Trả về (N_patches,) GradCAM score.
        image_tensor: (1,3,H,W) float32 hay float16
        """
        self.model.eval()
        image_tensor = image_tensor.to(device).half()

        # Forward – cần grad
        image_tensor.requires_grad_(False)
        with torch.enable_grad():
            x, _ = self.model.base_model.visual(image_tensor)
            # Dùng L2 norm của CLS token làm scalar target
            cls_feat = x[:, 0, :].float()
            score = cls_feat.norm()
            self.model.zero_grad()
            score.backward()

        feats = self._feats[0].float()   # (L, D)  – L = N_patches + 1
        grads = self._grads[0].float()   # (L, D)

        # Loại bỏ CLS padding (index 0)
        patch_feats = feats[1:]   # (N_patches, D)
        patch_grads = grads[1:]   # (N_patches, D)

        # Weighted activation: alpha * activation (theo Selvaraju et al.)
        weights    = patch_grads.mean(dim=-1, keepdim=True)  # (N_patches, 1)
        cam        = (weights * patch_feats).sum(dim=-1)      # (N_patches,)
        cam        = F.relu(cam).cpu().numpy()
        cam        = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self):
        for h in self._hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap overlay
# ─────────────────────────────────────────────────────────────────────────────
def patch_scores_to_heatmap(scores: np.ndarray,
                             img_hw: Tuple[int, int],
                             num_y: int, num_x: int,
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Reshape (N_patches,) -> (num_y, num_x), resize về img_hw, tô màu.
    Trả về np.uint8 (H, W, 3) BGR.
    """
    grid = scores.reshape(num_y, num_x)
    grid_u8 = (grid * 255).astype(np.uint8)
    heatmap = cv2.resize(grid_u8, (img_hw[1], img_hw[0]),
                         interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    return heatmap


def overlay(orig_rgb: np.ndarray, heatmap_bgr: np.ndarray,
            alpha: float = 0.45) -> np.ndarray:
    """Kết hợp ảnh gốc (RGB) + heatmap (BGR) -> RGB overlay."""
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_rgb + (1 - alpha) * orig_rgb).astype(np.uint8)
    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Visualize một ảnh với N phương pháp
# ─────────────────────────────────────────────────────────────────────────────
def visualize_image(model,
                    img_path: str,
                    methods: List[str],
                    device: str = "cuda",
                    img_size=(384, 128),
                    patch_size=16,
                    stride_size=16,
                    discard_ratio=0.9,
                    alpha=0.45,
                    save_path: Optional[str] = None):
    """
    Vẽ heatmap theo các method được chọn.
    methods: subset của ['last_attn', 'rollout', 'gradcam']
    """
    img_tensor, orig_rgb = load_image(img_path, img_size)
    num_y, num_x = patch_grid(img_size, patch_size, stride_size)
    N_patches = num_y * num_x

    results = {}   # method -> np (H,W,3) overlay

    # ── last_attn và rollout cần collect attention qua hooks ─────────────────
    if "last_attn" in methods or "rollout" in methods:
        vit_transformer = model.base_model.visual.transformer
        collector = AttentionCollector(vit_transformer)

        model.eval()
        with torch.no_grad():
            img_in = img_tensor.to(device).half()
            model.base_model.visual(img_in)

        collector.remove()
        attn_list = collector.attentions  # list[L x (B, N, N)]

        if "last_attn" in methods:
            scores = last_layer_attention(attn_list)
            assert len(scores) == N_patches, \
                f"Kỳ vọng {N_patches} patches, nhận {len(scores)}"
            hm = patch_scores_to_heatmap(scores, img_size, num_y, num_x)
            results["last_attn"] = overlay(orig_rgb, hm, alpha)

        if "rollout" in methods:
            scores = attention_rollout(attn_list, discard_ratio=discard_ratio)
            assert len(scores) == N_patches, \
                f"Kỳ vọng {N_patches} patches, nhận {len(scores)}"
            hm = patch_scores_to_heatmap(scores, img_size, num_y, num_x)
            results["rollout"] = overlay(orig_rgb, hm, alpha)

    # ── GradCAM ───────────────────────────────────────────────────────────────
    if "gradcam" in methods:
        gcam = GradCAMViT(model)
        scores = gcam.compute(img_tensor, device=device)
        gcam.remove()
        assert len(scores) == N_patches, \
            f"Kỳ vọng {N_patches} patches, nhận {len(scores)}"
        hm = patch_scores_to_heatmap(scores, img_size, num_y, num_x)
        results["gradcam"] = overlay(orig_rgb, hm, alpha)

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_cols = 1 + len(results)   # original + methods
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5),
                             constrained_layout=True)
    fig.suptitle(Path(img_path).name, fontsize=11, fontweight="bold")

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Ảnh gốc", fontsize=10)
    axes[0].axis("off")

    titles = {
        "last_attn": "Attn lớp cuối\n(CLS→Patches)",
        "rollout"  : "Attention Rollout\n(tất cả lớp)",
        "gradcam"  : "Grad-CAM ViT\n(lớp cuối)",
    }
    for i, (method, vis) in enumerate(results.items()):
        axes[i + 1].imshow(vis)
        axes[i + 1].set_title(titles.get(method, method), fontsize=10)
        axes[i + 1].axis("off")

    # Thêm colorbar riêng cho heatmap
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[1:].tolist() if n_cols > 1 else axes,
                 fraction=0.02, pad=0.01, label="Activation (norm)")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[XAI] Saved: {save_path}")
    else:
        out = f"xai_{Path(img_path).stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[XAI] Saved: {out}")

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: Optional[str], device: str) -> object:
    from model.build import build_model
    args = _make_args_for_build()
    model = build_model(args)
    model = model.to(device)

    if checkpoint_path:
        print(f"[info] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model", ckpt)  # support both formats
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] Missing keys ({len(missing)}): {missing[:5]} ...")
        if unexpected:
            print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
        print("[info] Checkpoint loaded.")
    else:
        print("[info] Không có checkpoint – dùng raw CLIP weights.")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Batch mode: xử lý cả thư mục ảnh
# ─────────────────────────────────────────────────────────────────────────────
def process_directory(model, img_dir: str, methods: List[str], save_dir: str,
                      device: str, img_size=(384, 128)):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    files.sort()

    if not files:
        print(f"[warn] Không tìm thấy ảnh trong '{img_dir}'")
        return

    print(f"[info] Xử lý {len(files)} ảnh từ '{img_dir}' -> '{save_dir}'")
    for i, fp in enumerate(files):
        out_name = os.path.join(save_dir, f"xai_{Path(fp).stem}.png")
        try:
            visualize_image(model, fp, methods, device=device,
                            img_size=img_size, save_path=out_name)
        except Exception as e:
            print(f"[error] {fp}: {e}")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(files)} done")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="XAI (GradCAM / Attention Rollout) cho LPNC model"
    )

    # Input
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--img",     type=str, help="Đường dẫn tới 1 ảnh")
    src.add_argument("--img_dir", type=str, help="Thư mục chứa nhiều ảnh")

    # Model
    parser.add_argument("--checkpoint",    type=str, default=None,
                        help="Path tới checkpoint .pth")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Dùng raw CLIP weights, không load checkpoint")

    # Methods
    parser.add_argument(
        "--method", type=str, default="all",
        choices=["all", "last_attn", "rollout", "gradcam"],
        help="Phương pháp XAI (mặc định: all)"
    )

    # Output
    parser.add_argument("--save",     type=str, default=None,
                        help="Path lưu hình (chỉ dùng với --img)")
    parser.add_argument("--save_dir", type=str, default="xai_outputs",
                        help="Thư mục lưu khi dùng --img_dir")

    # Config
    parser.add_argument("--device",       default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--alpha",        type=float, default=0.45,
                        help="Blend ratio (0=chỉ heatmap, 1=chỉ ảnh gốc)")
    parser.add_argument("--discard_ratio", type=float, default=0.9,
                        help="Tỷ lệ loại bỏ khi Rollout (mặc định 0.9)")
    parser.add_argument("--img_h",  type=int, default=384)
    parser.add_argument("--img_w",  type=int, default=128)

    args = parser.parse_args()

    # Device fallback
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA không khả dụng, chuyển sang CPU.")
        device = "cpu"

    img_size = (args.img_h, args.img_w)

    # Methods list
    if args.method == "all":
        methods = ["last_attn", "rollout", "gradcam"]
    else:
        methods = [args.method]

    # GradCAM cần gradient -> chỉ dùng cho 1 ảnh tại một thời điểm
    # (đã xử lý bên trong visualize_image)

    # Load model
    ckpt = None if args.no_checkpoint else args.checkpoint
    model = load_model(ckpt, device)

    if args.img:
        visualize_image(
            model, args.img, methods,
            device=device, img_size=img_size,
            discard_ratio=args.discard_ratio,
            alpha=args.alpha,
            save_path=args.save,
        )
    else:
        process_directory(
            model, args.img_dir, methods,
            save_dir=args.save_dir,
            device=device, img_size=img_size,
        )


if __name__ == "__main__":
    main()

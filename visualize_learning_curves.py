"""
visualize_learning_curves.py
============================
Đọc TensorBoard event files từ thư mục output_dir và vẽ learning curves.

Cách dùng
---------
  python visualize_learning_curves.py --log_dir run_logs/baseline
  python visualize_learning_curves.py --log_dir run_logs            # so sánh nhiều run
  python visualize_learning_curves.py --log_dir run_logs/baseline --save curves.png
"""

import os
import argparse
import glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on servers)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── TensorBoard event reader (fallback: raw file) ────────────────────────────
def load_tb_scalars(log_dir: str):
    """Return dict[tag] -> (steps[], values[]) from all tfevents files under log_dir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError(
            "tensorboard không được cài. Chạy: pip install tensorboard"
        )

    scalars = {}
    event_files = sorted(
        glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    )
    if not event_files:
        # Try the dir itself
        event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))

    if not event_files:
        raise FileNotFoundError(
            f"Không tìm thấy tfevents file trong '{log_dir}'.\n"
            "Hãy chạy training với --output_dir trỏ tới thư mục này."
        )

    print(f"[info] Tìm thấy {len(event_files)} event file(s) trong '{log_dir}'")

    for ef in event_files:
        ea = EventAccumulator(ef)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            steps  = np.array([e.step  for e in events])
            vals   = np.array([e.value for e in events])
            if tag not in scalars:
                scalars[tag] = (steps, vals)
            else:
                # merge (e.g. multiple event files for same run)
                old_steps, old_vals = scalars[tag]
                steps = np.concatenate([old_steps, steps])
                vals  = np.concatenate([old_vals,  vals])
                order = np.argsort(steps)
                scalars[tag] = (steps[order], vals[order])

    return scalars


# ── Smoothing helper ──────────────────────────────────────────────────────────
def smooth(values: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """EMA smoothing (giống TensorBoard)."""
    smoothed = np.zeros_like(values)
    last = values[0]
    for i, v in enumerate(values):
        last = weight * last + (1 - weight) * v
        smoothed[i] = last
    return smoothed


# ── Main plotting function ────────────────────────────────────────────────────
def plot_run(log_dir: str, save_path: str = None, smooth_weight: float = 0.6,
             max_epoch: int = None):
    scalars = load_tb_scalars(log_dir)

    if not scalars:
        print("[warn] Không có scalar nào trong log dir này.")
        return

    print(f"[info] Tags tìm thấy: {list(scalars.keys())}")

    # ── group: losses / lr / others ──────────────────────────────────────────
    loss_tags = [t for t in scalars if "loss" in t.lower()]
    lr_tags   = [t for t in scalars if "lr" in t.lower() or "learning_rate" in t.lower()]
    eval_tags = [t for t in scalars if any(k in t.lower() for k in
                 ["r1", "rank1", "top1", "map", "temp"])]
    # Deduplicate (lr could appear in eval_tags too)
    eval_tags = [t for t in eval_tags if t not in lr_tags]

    n_panels = (1 if loss_tags else 0) + (1 if lr_tags else 0) + (1 if eval_tags else 0)
    if n_panels == 0:
        # Fallback: plot everything
        loss_tags = list(scalars.keys())
        n_panels = 1

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(10, 4 * n_panels),
        constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]

    run_name = Path(log_dir).name
    fig.suptitle(f"Learning Curves — {run_name}", fontsize=14, fontweight="bold")

    panel_idx = 0

    # ── panel 1: losses ───────────────────────────────────────────────────────
    if loss_tags:
        ax = axes[panel_idx]; panel_idx += 1
        colors = plt.cm.tab10.colors
        for ci, tag in enumerate(sorted(loss_tags)):
            steps, vals = scalars[tag]
            if max_epoch:
                mask = steps <= max_epoch
                steps, vals = steps[mask], vals[mask]
            ax.plot(steps, vals, alpha=0.25, color=colors[ci % 10], linewidth=0.8)
            ax.plot(steps, smooth(vals, smooth_weight),
                    label=tag, color=colors[ci % 10], linewidth=1.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Losses")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # ── panel 2: eval metrics ─────────────────────────────────────────────────
    if eval_tags:
        ax = axes[panel_idx]; panel_idx += 1
        colors = plt.cm.Set2.colors
        for ci, tag in enumerate(sorted(eval_tags)):
            steps, vals = scalars[tag]
            if max_epoch:
                mask = steps <= max_epoch
                steps, vals = steps[mask], vals[mask]
            ax.plot(steps, vals, "-o", markersize=3,
                    label=tag, color=colors[ci % 8], linewidth=1.6)
            best_idx = np.argmax(vals)
            ax.annotate(f"best {vals[best_idx]:.3f}@{steps[best_idx]}",
                        (steps[best_idx], vals[best_idx]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7,
                        color=colors[ci % 8])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric value")
        ax.set_title("Evaluation Metrics")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    # ── panel 3: learning rate ─────────────────────────────────────────────────
    if lr_tags:
        ax = axes[panel_idx]; panel_idx += 1
        for tag in sorted(lr_tags):
            steps, vals = scalars[tag]
            if max_epoch:
                mask = steps <= max_epoch
                steps, vals = steps[mask], vals[mask]
            ax.plot(steps, vals, linewidth=1.8, label=tag)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[info] Đã lưu: {save_path}")
    else:
        # Try interactive; fall back to saving
        try:
            matplotlib.use("TkAgg")
            plt.show()
        except Exception:
            out = "learning_curves.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[info] Đã lưu: {out}")

    plt.close(fig)


# ── Multi-run comparison ──────────────────────────────────────────────────────
def compare_runs(log_dir: str, save_path: str = None, smooth_weight: float = 0.6,
                 target_tag: str = "loss"):
    """Vẽ một tag cụ thể cho tất cả sub-directories (nhiều run) trên cùng 1 axes."""
    run_dirs = sorted([
        d for d in Path(log_dir).iterdir()
        if d.is_dir() and list(d.glob("events.out.tfevents.*"))
    ])
    if not run_dirs:
        print("[warn] Không tìm thấy sub-run nào. Dùng plot_run thay thế.")
        plot_run(log_dir, save_path=save_path, smooth_weight=smooth_weight)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    for ci, run_dir in enumerate(run_dirs):
        try:
            scalars = load_tb_scalars(str(run_dir))
        except Exception as e:
            print(f"[warn] Skip {run_dir.name}: {e}")
            continue

        # find closest matching tag
        candidates = [t for t in scalars if target_tag.lower() in t.lower()]
        tag = candidates[0] if candidates else None
        if tag is None:
            print(f"[warn] Tag '{target_tag}' không có trong {run_dir.name}")
            continue

        steps, vals = scalars[tag]
        color = colors[ci % 10]
        ax.plot(steps, vals, alpha=0.2, color=color, linewidth=0.8)
        ax.plot(steps, smooth(vals, smooth_weight),
                label=f"{run_dir.name} ({tag})", color=color, linewidth=1.8)

    ax.set_xlabel("Epoch"); ax.set_ylabel(target_tag)
    ax.set_title(f"Run Comparison — {target_tag}")
    ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[info] Đã lưu: {save_path}")
    else:
        out = "run_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[info] Đã lưu: {out}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Vẽ learning curves từ TensorBoard log"
    )
    parser.add_argument("--log_dir", default="run_logs",
                        help="Thư mục chứa tfevents (hoặc thư mục cha chứa nhiều run)")
    parser.add_argument("--save",    default=None,
                        help="Lưu hình ra file (PNG/PDF). Mặc định: hiển thị hoặc lưu vào thư mục hiện tại")
    parser.add_argument("--compare", action="store_true",
                        help="So sánh nhiều run trong cùng log_dir")
    parser.add_argument("--tag",     default="loss",
                        help="Tag để so sánh khi dùng --compare (mặc định: loss)")
    parser.add_argument("--smooth",  type=float, default=0.6,
                        help="EMA smoothing weight (0=off, 0.9=heavy, mặc định 0.6)")
    parser.add_argument("--max_epoch", type=int, default=None,
                        help="Chỉ vẽ đến epoch này")
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.log_dir, save_path=args.save,
                     smooth_weight=args.smooth, target_tag=args.tag)
    else:
        plot_run(args.log_dir, save_path=args.save,
                 smooth_weight=args.smooth, max_epoch=args.max_epoch)


if __name__ == "__main__":
    main()

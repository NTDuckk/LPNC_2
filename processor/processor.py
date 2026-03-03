import logging
import sys
import time
from pathlib import Path
import torch
from torch.cuda.amp import autocast, GradScaler

from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter

# Ensure project root is importable for standalone visualize_* scripts
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _run_post_training_viz(args, model, img_loader=None):
    """
    Tự động chạy visualize_learning_curves và visualize_xai sau khi train xong.
    Chỉ gọi trên rank 0.
    """
    logger = logging.getLogger("LPNC.train")
    output_dir = args.output_dir
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Learning curves ────────────────────────────────────────────────────
    try:
        from visualize_learning_curves import plot_run
        curves_out = str(viz_dir / "learning_curves.png")
        plot_run(log_dir=output_dir, save_path=curves_out)
        logger.info(f"[viz] Learning curves saved -> {curves_out}")
    except Exception as e:
        logger.warning(f"[viz] Learning curves failed: {e}")

    # ── 2. XAI heatmaps on a few sample images ────────────────────────────────
    try:
        from visualize_xai import visualize_image, load_model as _load_model_xai

        # Try to get one batch of images from img_loader
        sample_imgs = []
        if img_loader is not None:
            try:
                _pid, _img = next(iter(img_loader))
                # save the first image temporarily
                import torchvision.transforms.functional as TVF
                from PIL import Image as PILImage
                tmp_img_path = str(viz_dir / "_sample_img.jpg")
                # de-normalize: mean/std used in datasets
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
                std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)
                img0 = _img[0].cpu().float() * std + mean
                img0 = img0.clamp(0, 1)
                PILImage.fromarray((img0.permute(1,2,0).numpy() * 255).astype('uint8')).save(tmp_img_path)
                sample_imgs.append(tmp_img_path)
            except Exception as e:
                logger.warning(f"[viz] Could not extract sample image: {e}")

        if not sample_imgs:
            logger.info("[viz] No sample images for XAI – skipping.")
        else:
            # Load best checkpoint if it exists
            best_ckpt = Path(output_dir) / "best.pth"
            ckpt_path = str(best_ckpt) if best_ckpt.exists() else None

            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            xai_model = _load_model_xai(ckpt_path, device_str)

            for img_path in sample_imgs:
                out_path = str(viz_dir / f"xai_{Path(img_path).stem}.png")
                visualize_image(
                    xai_model, img_path,
                    methods=["last_attn", "rollout", "gradcam"],
                    device=device_str,
                    img_size=getattr(args, "img_size", (384, 128)),
                    save_path=out_path,
                )
                logger.info(f"[viz] XAI saved -> {out_path}")
    except Exception as e:
        logger.warning(f"[viz] XAI visualization failed: {e}")


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch

    arguments = {"num_epoch": num_epoch, "iteration": 0}

    logger = logging.getLogger("LPNC.train")
    logger.info("start training")

    # keep only supervised ID/contrastive/triplet components
    meters = {
        "loss": AverageMeter(),
        "supcon_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "triplet_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    scaler = GradScaler()

    best_top1 = 0.0
    now_top1 = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        # step scheduler at start of epoch (consistent with stage2)
        scheduler.step()
        for meter in meters.values():
            meter.reset()

        model.train()
        model.epoch = epoch

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # FP32 model + autocast for mixed-precision efficiency + GradScaler for backward
            with autocast():
                ret = model(batch)
                # Weighted sum of component losses using config lambdas
                lam1 = getattr(args, "lambda1_weight", 0.5)
                lam2 = getattr(args, "lambda2_weight", 1.0)
                lam3 = getattr(args, "lambda3_weight", 1.0)
                supcon_val = ret.get("supcon_loss", 0.0)
                id_val = ret.get("id_loss", 0.0)
                triplet_val = ret.get("triplet_loss", 0.0)
                total_loss = lam1 * supcon_val + lam2 * id_val + lam3 * triplet_val

            batch_size = batch["images"].shape[0]
            meters["loss"].update(float(total_loss.item()), batch_size)
            # keep per-component meters
            meters["supcon_loss"].update(float(ret.get("supcon_loss", 0.0)), batch_size)
            meters["id_loss"].update(float(ret.get("id_loss", 0.0)), batch_size)
            meters["triplet_loss"].update(float(ret.get("triplet_loss", 0.0)), batch_size)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                # per-batch component losses are reported via meters; no extra Curr(...) printing

                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
                # also print to stdout for immediate visibility
                try:
                    print(info_str, flush=True)
                except Exception:
                    pass

        # tensorboard
        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        if "temperature" in ret:
            tb_writer.add_scalar("temperature", float(ret["temperature"]), epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        # scheduler already stepped at start of epoch

        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())
                now_top1 = max(now_top1, float(top1))
                torch.cuda.empty_cache()

                if best_top1 < top1:
                    best_top1 = float(top1)
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments.get('epoch', -1)}")
        # Auto-run visualizations after training completes
        _run_post_training_viz(args, model, img_loader=evaluator.img_loader)


def do_inference(model, test_img_loader, test_txt_loader, refer_loader, args):
    logger = logging.getLogger("LPNC.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, refer_loader, args)
    _ = evaluator.eval(model.eval())
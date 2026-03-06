import logging
import sys
import time
from pathlib import Path
import torch
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter 


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
        import traceback as _tb
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
        import traceback as _tb2
        logger.warning(f"[viz] XAI visualization failed: {e}\n{_tb2.format_exc()}")

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
                scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("LPNC.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "supcon_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "triplet_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    scaler = GradScaler()

    best_top1 = 0.0
    # evaluator.eval(model.eval())
    # train
    sims = []
    now_top1 = 0
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.train()
        model.epoch = epoch
    
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch.get('index', None)

            # Mixed-precision forward
            with autocast():
                ret = model(batch)
                # sum all returned tensors whose key contains 'loss'
                total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(float(total_loss.item()), batch_size)
            meters['supcon_loss'].update(ret.get('supcon_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['triplet_loss'].update(ret.get('triplet_loss', 0), batch_size)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            synchronize()
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
    

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        if 'temperature' in ret:
            try:
                tb_writer.add_scalar('temperature', float(ret.get('temperature')), epoch)
            except Exception:
                pass
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                now_top1 = max(now_top1, top1)
                torch.cuda.empty_cache()

                # --- Compute losses on validation set (no weight updates) ---
                try:
                    eval_meters = {
                        "loss": AverageMeter(),
                        "supcon_loss": AverageMeter(),
                        "id_loss": AverageMeter(),
                        "triplet_loss": AverageMeter(),
                    }

                    # Use the underlying model if wrapped by DDP
                    effective_model = model.module if hasattr(model, 'module') else model
                    effective_model.eval()

                    device_eff = next(effective_model.parameters()).device

                    with torch.no_grad():
                        # evaluator holds val_img_loader and val_txt_loader
                        img_loader = getattr(evaluator, 'img_loader', None)
                        txt_loader = getattr(evaluator, 'txt_loader', None)
                        if img_loader is not None and txt_loader is not None:
                            # Build mapping pid -> list[captions] from txt_loader
                            caption_map = defaultdict(list)
                            txt_count = 0
                            for txt_pids, captions in txt_loader:
                                for i in range(len(txt_pids)):
                                    pid_val = int(txt_pids[i].item()) if hasattr(txt_pids[i], 'item') else int(txt_pids[i])
                                    caption_map[pid_val].append(captions[i].cpu())
                                    txt_count += 1

                            num_classes = None
                            try:
                                num_classes = effective_model.classifier_proj.out_features
                            except Exception:
                                pass

                            i_iter = 0
                            # Iterate image batches and align with one caption per image
                            for img_pids, imgs in img_loader:
                                aligned_imgs = []
                                aligned_caps = []
                                aligned_pids = []

                                for j in range(len(img_pids)):
                                    pid_img = int(img_pids[j].item()) if hasattr(img_pids[j], 'item') else int(img_pids[j])

                                    # direct match
                                    cap_list = caption_map.get(pid_img, None)

                                    # try 1-based -> 0-based if no match
                                    if cap_list is None and num_classes is not None:
                                        pid_try = pid_img - 1
                                        cap_list = caption_map.get(pid_try, None)
                                        if cap_list is not None:
                                            pid_img = pid_try

                                    if not cap_list:
                                        logger.warning(f"Missing caption for pid={pid_img}; skipping this sample in eval loss")
                                        continue

                                    cap = cap_list.pop(0)
                                    aligned_pids.append(pid_img)
                                    aligned_imgs.append(imgs[j].unsqueeze(0))
                                    aligned_caps.append(cap.unsqueeze(0) if cap.dim()==1 else cap.unsqueeze(0))

                                if not aligned_imgs:
                                    continue

                                # create batch
                                try:
                                    batch_images = torch.cat(aligned_imgs, dim=0).to(device_eff)
                                except Exception:
                                    batch_images = torch.stack([t.squeeze(0) for t in aligned_imgs], dim=0).to(device_eff)
                                batch_captions = torch.cat(aligned_caps, dim=0).to(device_eff)
                                batch_pids = torch.tensor(aligned_pids, dtype=torch.long, device=device_eff)

                                # final pid range check: try subtracting 1 if out-of-range; otherwise skip
                                if num_classes is not None:
                                    if batch_pids.max() >= num_classes or batch_pids.min() < 0:
                                        batch_pids2 = batch_pids - 1
                                        if batch_pids2.max() < num_classes and batch_pids2.min() >= 0:
                                            batch_pids = batch_pids2
                                            logger.info("Adjusted eval pids by -1 to match class range")
                                        else:
                                            logger.warning(f"Eval batch pids out-of-range (min={int(batch_pids.min())}, max={int(batch_pids.max())}); skipping batch")
                                            continue

                                batch = {
                                    'pids': batch_pids,
                                    'images': batch_images,
                                    'caption_ids': batch_captions,
                                }

                                with autocast():
                                    ret = effective_model(batch)

                                total_loss = sum([v for k, v in ret.items() if "loss" in k])

                                bsz = batch['images'].shape[0] if hasattr(batch['images'], 'shape') else 1

                                eval_meters['loss'].update(float(total_loss.item()), bsz)
                                eval_meters['supcon_loss'].update(float(ret.get('supcon_loss', 0)), bsz)
                                eval_meters['id_loss'].update(float(ret.get('id_loss', 0)), bsz)
                                eval_meters['triplet_loss'].update(float(ret.get('triplet_loss', 0)), bsz)

                                i_iter += 1
                                if (i_iter) % max(1, getattr(args, 'log_period', 50)) == 0:
                                    logger.info(
                                        f"Epoch[{epoch}] Eval Iter[{i_iter}], avg_loss: {eval_meters['loss'].avg:.4f}"
                                    )

                        else:
                            logger.warning("Evaluator does not expose paired img/txt loaders for loss computation.")

                    # Log averaged eval losses
                    logger.info(
                        "Eval losses - avg_loss: {:.4f}, supcon: {:.4f}, id: {:.4f}, triplet: {:.4f}".format(
                            eval_meters['loss'].avg, eval_meters['supcon_loss'].avg, eval_meters['id_loss'].avg, eval_meters['triplet_loss'].avg
                        )
                    )
                except Exception as _e:
                    logger.warning(f"Validation loss computation failed: {_e}")

                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")
        # Auto-run visualizations after training completes (rank 0 only)
        try:
            _run_post_training_viz(args, model, img_loader=evaluator.img_loader)
        except Exception as e:
            logger.warning(f"[viz] post-training viz failed: {e}")

                
def do_inference(model, test_img_loader, test_txt_loader, refer_loader,args):

    logger = logging.getLogger("LPNC.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, refer_loader,args)
    top1 = evaluator.eval(model.eval())

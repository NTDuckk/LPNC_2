import logging
import time
import torch

from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter


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

    best_top1 = 0.0
    now_top1 = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.train()
        model.epoch = epoch

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)

            # sum component losses
            total_loss = sum([ret[k] for k in ("supcon_loss", "id_loss", "triplet_loss") if k in ret])

            batch_size = batch["images"].shape[0]
            meters["loss"].update(float(total_loss.item()), batch_size)
            # keep per-component meters
            meters["supcon_loss"].update(float(ret.get("supcon_loss", 0.0)), batch_size)
            meters["id_loss"].update(float(ret.get("id_loss", 0.0)), batch_size)
            meters["triplet_loss"].update(float(ret.get("triplet_loss", 0.0)), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                # append current per-batch loss values (from model return dict)
                current_losses = []
                for k, val in ret.items():
                    if "loss" in k:
                        try:
                            current_losses.append(f"{k}: {float(val):.4f}")
                        except Exception:
                            pass
                if current_losses:
                    info_str += ", Curr(" + ", ".join(current_losses) + ")"

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

        scheduler.step()

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


def do_inference(model, test_img_loader, test_txt_loader, refer_loader, args):
    logger = logging.getLogger("LPNC.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, refer_loader, args)
    _ = evaluator.eval(model.eval())
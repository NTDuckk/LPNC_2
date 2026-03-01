import logging
import os
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter


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

    # AMP setup
    use_fp16 = getattr(args, 'fp16', False)
    scaler = GradScaler() if use_fp16 else None
    accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    if use_fp16:
        logger.info("AMP mixed precision training enabled")
    if accum_steps > 1:
        logger.info(f"Gradient accumulation: {accum_steps} steps (effective_bs = {args.batch_size} x {accum_steps} = {args.batch_size * accum_steps})")

    meters = {
        "loss": AverageMeter(),
        "supid_loss": AverageMeter(),
        "cotrl_loss": AverageMeter(),
        "cid_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

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
        optimizer.zero_grad()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            index = batch['index']

            with autocast(enabled=use_fp16):
                ret = model(batch)
                total_loss = sum([v for k, v in ret.items() if "loss" in k])
                if accum_steps > 1:
                    total_loss = total_loss / accum_steps

            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (n_iter + 1) % accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            batch_size = batch['images'].shape[0]
            loss_val = total_loss.item() * accum_steps if accum_steps > 1 else total_loss.item()
            meters['loss'].update(loss_val, batch_size)
            meters['supid_loss'].update(ret.get('supid_loss', 0), batch_size)
            meters['cotrl_loss'].update(ret.get('cotrl_loss', 0), batch_size)
            meters['cid_loss'].update(ret.get('cid_loss', 0), batch_size)
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
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
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
                now_top1 = max(now_top1,top1)
                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
 
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

                    

def do_inference(model, test_img_loader, test_txt_loader, refer_loader,args):

    logger = logging.getLogger("LPNC.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, refer_loader,args)
    top1 = evaluator.eval(model.eval())

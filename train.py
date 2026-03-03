import os
import os.path as op
import time
import random
import warnings

import torch
import numpy as np

from datasets import build_dataloader
from processor.processor import do_train, do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.options import get_args
from utils.comm import get_rank, synchronize

warnings.filterwarnings("ignore")


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _broadcast_str_if_ddp(s: str, src: int = 0) -> str:
    """Broadcast a python string from rank0 to all ranks (DDP-safe)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return s
    obj = [s]
    torch.distributed.broadcast_object_list(obj, src=src)
    return obj[0]


if __name__ == '__main__':
    args = get_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # seed per-rank
    set_seed(1 + get_rank())

    # build a shared output_dir name across ranks (avoid mismatch + FileExistsError)
    if get_rank() == 0:
        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    else:
        cur_time = ""
    cur_time = _broadcast_str_if_ddp(cur_time, src=0)

    name = args.name
    args.output_dir = op.join(args.output_dir, args.dataset_name, f"{cur_time}_{name}_{args.loss_names}")

    # logger (only rank0 writes handlers/files; see utils/logger.py)
    logger = setup_logger('LPNC', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))

    # only rank0 touches filesystem for configs/fig folder
    if get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        save_train_configs(args.output_dir, args)
        os.makedirs(op.join(args.output_dir, "img"), exist_ok=True)
    synchronize()

    device = "cuda"

    # TRAIN
    if args.training:
        # dataloaders: train + val (split chosen by args.val_dataset)
        train_loader, val_img_loader, val_txt_loader, refer_txt_loader, num_classes = build_dataloader(args)

        # model
        model = build_model(args, num_classes)
        logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))
        model.to(device)

        # optional load checkpoint weights (finetune path)
        if args.finetune:
            logger.info("loading {} model".format(args.finetune))
            ckpt = torch.load(args.finetune, map_location='cpu')
            param_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            new_sd = {}
            for k, v in param_dict.items():
                nk = k.replace('module.', '')
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)

        # DDP wrap
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )

        optimizer = build_optimizer(args, model)
        scheduler = build_lr_scheduler(args, optimizer)

        is_master = get_rank() == 0
        checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)

        start_epoch = 1
        if args.resume:
            checkpoint = checkpointer.resume(args.resume_ckpt_file)
            start_epoch = checkpoint.get('epoch', 1)
            logger.info(f"===================> start {start_epoch}")

        from utils.metrics import Evaluator
        evaluator = Evaluator(val_img_loader, val_txt_loader, refer_txt_loader, args)

        do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)

    # TEST / INFERENCE
    else:
        # dataloaders: test only (always uses dataset.test)
        test_img_loader, test_txt_loader, refer_txt_loader, num_classes = build_dataloader(args)

        # model
        model = build_model(args, num_classes)
        logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))
        model.to(device)

        # optional load checkpoint weights (finetune path)
        if args.finetune:
            logger.info("loading {} model".format(args.finetune))
            ckpt = torch.load(args.finetune, map_location='cpu')
            param_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
            new_sd = {}
            for k, v in param_dict.items():
                nk = k.replace('module.', '')
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)

        # DDP wrap
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )

        # only rank0 runs eval to avoid duplicated logs/output
        if args.distributed and get_rank() != 0:
            synchronize()
            raise SystemExit(0)

        effective_model = model.module if hasattr(model, "module") else model
        do_inference(effective_model, test_img_loader, test_txt_loader, refer_txt_loader, args)
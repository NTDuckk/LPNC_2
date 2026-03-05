import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    # Ensure any copied/frozen text encoder stays frozen even if base_model
    # text parameters are trainable elsewhere. This makes optimizer skip
    # parameters whose name contains 'text_encoder'.
    for n, p in model.named_parameters():
        if "text_encoder" in n:
            p.requires_grad = False

    # Two-group LR strategy:
    #   base_model (image encoder / CLIP visual backbone) -> args.lr        (e.g. 1e-6)
    #   all other trainable modules (mapping, cross-attn, head, ...)       -> args.lr * args.lr_factor (e.g. 1e-5)
    print(f'Image encoder lr: {args.lr:.2e}   |   Mapping/head lr: {args.lr * args.lr_factor:.2e}')

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = args.weight_decay

        # Image encoder (CLIP visual backbone kept in base_model)
        if "base_model" in key:
            lr = args.lr
        else:
            # randomly-initialised mapping, cross-modal, classifier heads, etc.
            lr = args.lr * args.lr_factor

        # Bias terms: slightly higher lr, lower weight decay
        if "bias" in key:
            lr = lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )

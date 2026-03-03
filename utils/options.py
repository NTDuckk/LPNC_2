import argparse


def get_args():
    parser = argparse.ArgumentParser(description="LPNC Args (PromptSG-style inference)")

    # (legacy params kept; cotrl/cid removed in code path)
    parser.add_argument("--tau", default=0.015, type=float)
    parser.add_argument("--select_ratio", default=0.4, type=float)
    parser.add_argument("--margin", default=0.1, type=float)

    # Loss weights
    parser.add_argument("--lambda1_weight", default=0.5, type=float)   # SupCon weight
    parser.add_argument("--lambda2_weight", default=1.0, type=float)   # ID(CE) weight
    parser.add_argument("--lambda3_weight", default=1.0, type=float)   # Triplet weight (new)
    parser.add_argument("--triplet_margin", default=0.3, type=float)   # Triplet margin (new)

    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="run_logs")
    parser.add_argument("--log_period", default=20, type=int)
    parser.add_argument("--eval_period", default=1, type=int)
    parser.add_argument("--val_dataset", default="test")  # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--finetune", type=str, default="", help="load weights from checkpoint for train/test")
    parser.add_argument("--pretrain", type=str, default="")

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16')
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value")
    parser.add_argument("--img_aug", default=True, action='store_true')
    parser.add_argument("--txt_aug", default=True, action='store_true')

    # cross modal transformer setting
    parser.add_argument("--cmt_depth", type=int, default=2, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=10.0, help="lr factor for random init modules")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use MLM dataset")

    ######################## loss settings ########################
    # Loss pipeline name (legacy: previously 'supid'); kept for compatibility
    parser.add_argument("--loss_names", default='combined', help="loss pipeline name (default: combined)")

    ######################## inference settings (PromptSG-style) ########################
    # simplified: fixed prompt "A photo of a person" (fast)
    # composed: per-image pseudo-token -> prompt -> text encoder (slow)
    parser.add_argument("--infer_prompt", type=str, default="composed ", choices=["simplified", "composed"])
    parser.add_argument("--fixed_prompt", type=str, default="A photo of a person")

    ######################## vision transformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)

    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 40))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="step")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="identity", help="choose sampler from [identity, random]")
    parser.add_argument("--num_instance", type=int, default=2)
    parser.add_argument("--root_dir", default="datasets/dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    # --test switches to inference mode (training=False)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()
    return args
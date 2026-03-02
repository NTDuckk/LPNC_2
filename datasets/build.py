import random
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .bases import ImageTextDataset, ImageDataset, TextDataset
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
from .preprocessing import RandomErasing


def _make_transforms(args, is_train: bool):
    # CLIP mean/std (OpenAI)
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    h, w = args.img_size  # e.g. (384, 128)

    if is_train:
        t = [
            transforms.Resize((h, w)),
            transforms.RandomHorizontalFlip(p=0.5 if getattr(args, "img_aug", True) else 0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        # Random Erasing (optional)
        if getattr(args, "img_aug", True):
            t.append(RandomErasing(probability=0.5))
        return transforms.Compose(t)

    # test/val
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _collate_dict(batch):
    """
    Batch is list[dict] from ImageTextDataset.__getitem__.
    Return dict[tensor].
    """
    out = {}
    # keys: pids, image_ids, images, caption_ids, index
    out["pids"] = torch.tensor([b["pids"] for b in batch], dtype=torch.long)
    out["image_ids"] = torch.tensor([b["image_ids"] for b in batch], dtype=torch.long)
    out["images"] = torch.stack([b["images"] for b in batch], dim=0)
    out["caption_ids"] = torch.stack([b["caption_ids"] for b in batch], dim=0)
    out["index"] = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    return out


def _parse_annos(annos):
    """
    Accepts annos as list of:
      - (pid, img_path) OR
      - (pid, image_id, img_path) OR
      - any tuple where pid is first and img_path is last
    """
    pids, paths = [], []
    for a in annos:
        if isinstance(a, dict):
            pid = a.get("pid", a.get("person_id"))
            img_path = a.get("img_path", a.get("path"))
        else:
            pid = a[0]
            img_path = a[-1]
        pids.append(pid)
        paths.append(img_path)
    return pids, paths


def _parse_captions(split):
    """
    Accepts split as:
      - dict with keys: captions, caption_pids (or pids)
      - OR list of (pid, caption)
    """
    if isinstance(split, dict):
        captions = split.get("captions", [])
        caption_pids = split.get("caption_pids", split.get("pids", []))
        return caption_pids, captions

    # list/tuple
    caption_pids, captions = [], []
    for item in split:
        pid, cap = item[0], item[1]
        caption_pids.append(pid)
        captions.append(cap)
    return caption_pids, captions


def _build_dataset(args):
    """
    Expect your repo already has dataset classes, e.g.:
      datasets/cuhkpedes.py, datasets/icfgpedes.py, datasets/rstpreid.py
    Each dataset should provide:
      - train : list of (pid, image_id, img_path, caption)
      - val_annos / test_annos : list of (pid, img_path) or (pid, image_id, img_path)
      - val / test : dict with captions + caption_pids (or list of (pid, caption))
      - train_id_container (optional) to infer num_classes
    """
    name = args.dataset_name

    if name == "CUHK-PEDES":
        from .cuhkpedes import CUHKPEDES as Dataset
    elif name == "ICFG-PEDES":
        from .icfgpedes import ICFGPEDES as Dataset
    elif name == "RSTPReid":
        from .rstpreid import RSTPReid as Dataset
    else:
        raise ValueError(f"Unknown dataset_name={name}. Please add it to datasets/build.py")

    # try a few common ctor patterns
    try:
        dataset = Dataset(args.root_dir, args)
    except TypeError:
        try:
            dataset = Dataset(args.root_dir)
        except TypeError:
            dataset = Dataset(args)

    return dataset


def build_dataloader(args) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader], int]:
    """
    Returns:
      train_loader,
      val_img_loader (gallery images),
      val_txt_loader (query texts),
      refer_txt_loader -> None (kNC removed),
      num_classes
    """
    dataset = _build_dataset(args)

    # num_classes
    if hasattr(dataset, "train_id_container"):
        num_classes = len(dataset.train_id_container)
    else:
        # fallback: infer from train pids
        pids = [x[0] for x in dataset.train]
        num_classes = int(max(pids)) + 1 if len(pids) else 0

    train_tf = _make_transforms(args, is_train=True)
    test_tf = _make_transforms(args, is_train=False)

    train_set = ImageTextDataset(dataset.train, args, transform=train_tf, text_length=args.text_length, truncate=True)

    # sampler
    if args.sampler == "identity":
        if getattr(args, "distributed", False):
            sampler = RandomIdentitySampler_DDP(train_set.dataset, batch_size=args.batch_size, num_instances=args.num_instance)
        else:
            sampler = RandomIdentitySampler(train_set.dataset, batch_size=args.batch_size, num_instances=args.num_instance)

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size if not getattr(args, "distributed", False) else args.batch_size // torch.distributed.get_world_size(),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=_collate_dict,
        )
    else:
        # random
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=_collate_dict,
        )

    # build val/test loaders
    split_name = getattr(args, "val_dataset", "test")  # "val" or "test"
    if split_name == "val":
        annos = getattr(dataset, "val_annos", None)
        split = getattr(dataset, "val", None)
    else:
        annos = getattr(dataset, "test_annos", None)
        split = getattr(dataset, "test", None)

    if annos is None or split is None:
        raise ValueError(f"Dataset object must provide '{split_name}_annos' and '{split_name}'")

    img_pids, img_paths = _parse_annos(annos)
    caption_pids, captions = _parse_captions(split)

    val_img_set = ImageDataset(img_pids, img_paths, transform=test_tf)
    val_txt_set = TextDataset(caption_pids, captions, text_length=args.text_length, truncate=True)

    val_img_loader = DataLoader(
        val_img_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_txt_loader = DataLoader(
        val_txt_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # kNC removed => no refer loader
    refer_txt_loader = None

    return train_loader, val_img_loader, val_txt_loader, refer_txt_loader, num_classes
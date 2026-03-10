import logging
import random

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .market1501 import Market1501


__factory = {
    'CUHK-PEDES': CUHKPEDES,
    'ICFG-PEDES': ICFGPEDES,
    'RSTPReid': RSTPReid,
    'Market1501': Market1501
}


def build_transforms(img_size=(384, 128), aug=False, is_train=True,
                     flip_prob=0.5, pad_size=10, re_prob=0.5):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(flip_prob),
            T.Pad(pad_size),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=re_prob, scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], (int, np.int64)):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


def _build_refer_loader(dataset, args, num_workers):
    random.seed(42)

    all_text = []
    all_id = []

    # dataset.train is expected as list of tuples: (pid, image_id, img_path, caption)
    for pid, _, _, text in dataset.train:
        all_id.append(pid)
        all_text.append(text)

    if len(all_text) == 0:
        raise RuntimeError("No training captions found to build refer_txt_loader.")

    sample_size = max(1, int(len(all_text) * 0.2))
    sample_size = min(sample_size, len(all_text))
    random_indices = random.sample(range(len(all_text)), sample_size)

    sampled_ids = [all_id[i] for i in random_indices]
    sampled_texts = [all_text[i] for i in random_indices]

    refer = TextDataset(
        sampled_ids,
        sampled_texts,
        text_length=args.text_length
    )

    refer_txt_loader = DataLoader(
        refer,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return refer_txt_loader


def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    refer_txt_loader = _build_refer_loader(dataset, args, num_workers)

    if args.training:
        train_transforms = build_transforms(
            img_size=args.img_size,
            aug=args.img_aug,
            is_train=True,
            flip_prob=args.flip_prob,
            pad_size=args.pad_size,
            re_prob=args.re_prob
        )
        val_transforms = build_transforms(
            img_size=args.img_size,
            is_train=False
        )

        train_set = ImageTextDataset(
            dataset.train,
            args,
            train_transforms,
            text_length=args.text_length
        )

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')

                mini_batch_size = args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance
                )
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True
                )
                train_loader = DataLoader(
                    train_set,
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                    collate_fn=collate
                )
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, '
                    f'id: {args.batch_size // args.num_instance}, '
                    f'instance: {args.num_instance}'
                )
                train_loader = DataLoader(
                    train_set,
                    batch_size=args.batch_size,
                    sampler=RandomIdentitySampler(
                        dataset.train, args.batch_size, args.num_instance
                    ),
                    num_workers=num_workers,
                    collate_fn=collate
                )

        elif args.sampler == 'random':
            logger.info('using random sampler')
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate
            )
        else:
            raise ValueError(
                f"unsupported sampler! expected identity or random but got {args.sampler}"
            )

        # For Market1501:
        #   dataset.val  -> query captions/images
        #   dataset.test -> gallery captions/images
        ds = dataset.val if args.val_dataset == 'val' else dataset.test

        val_img_set = ImageDataset(
            ds['image_pids'],
            ds['img_paths'],
            val_transforms
        )
        val_txt_set = TextDataset(
            ds['caption_pids'],
            ds['captions'],
            text_length=args.text_length
        )

        val_img_loader = DataLoader(
            val_img_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        val_txt_loader = DataLoader(
            val_txt_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_img_loader, val_txt_loader, refer_txt_loader, num_classes

    else:
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(
                img_size=args.img_size,
                is_train=False
            )

        ds = dataset.test

        test_img_set = ImageDataset(
            ds['image_pids'],
            ds['img_paths'],
            test_transforms
        )
        test_txt_set = TextDataset(
            ds['caption_pids'],
            ds['captions'],
            text_length=args.text_length
        )

        test_img_loader = DataLoader(
            test_img_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        test_txt_loader = DataLoader(
            test_txt_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return test_img_loader, test_txt_loader, refer_txt_loader, num_classes
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from PIL import Image, ImageFile
import errno
import json
import pickle as pkl
import os
import os.path as osp
import yaml
from easydict import EasyDict as edict

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def mkdir_if_missing(directory):
    if directory == '':
        return
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        obj = pkl.load(f)
    return obj


def save_pickle(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_train_configs(path, args):
    """DDP-safe save configs.

    - Avoid race condition when multiple ranks create the same folder.
    - Only rank 0 writes configs.yaml.
    - If distributed is initialized, synchronize with a barrier.
    """
    import torch

    # Always make sure directory exists (exist_ok avoids FileExistsError race)
    os.makedirs(path, exist_ok=True)

    # Determine rank robustly (works even if process group not initialized yet)
    rank = int(os.environ.get("RANK", "0"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    if rank == 0:
        with open(os.path.join(path, "configs.yaml"), "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_train_configs(path):
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)
# encoding: utf-8
import glob
import json
import os.path as op
import re
from typing import Dict, List, Any

from .bases import BaseDataset


class Market1501(BaseDataset):
    """
    Market1501 with captions for text-image retrieval style loading.

    Expected image folders:
        {root}/Market-1501-v15.09.15/bounding_box_train
        {root}/Market-1501-v15.09.15/query
        {root}/Market-1501-v15.09.15/bounding_box_test

    Expected caption files (placed directly under {root}):
        market1501_bounding_box_train[.jsonl|.json]
        market1501_query[.jsonl|.json]
        market1501_bounding_box_test[.jsonl|.json]

    Supported caption-file styles:
    1) JSONL: one record per line
    2) JSON: list[dict]
    3) JSON: dict mapping filename -> caption(s)

    Each record can use common keys like:
        image key: img_path / image_path / file_path / filename / file_name / image / img / path / name
        caption key: captions / caption / texts / text / description / descriptions
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root='', verbose=True, **kwargs):
        super(Market1501, self).__init__()

        self.root = root
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.train_dir = op.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = op.join(self.dataset_dir, 'query')
        self.gallery_dir = op.join(self.dataset_dir, 'bounding_box_test')

        self.train_anno_path = self._resolve_caption_file('market1501_bounding_box_train')
        self.query_anno_path = self._resolve_caption_file('market1501_query')
        self.gallery_anno_path = self._resolve_caption_file('market1501_bounding_box_test')

        self._check_before_run()

        self.train, self.train_id_container = self._process_split(
            img_dir=self.train_dir,
            anno_path=self.train_anno_path,
            training=True,
            relabel=True
        )
        self.val, self.val_id_container = self._process_split(
            img_dir=self.query_dir,
            anno_path=self.query_anno_path,
            training=False,
            relabel=False
        )
        self.test, self.test_id_container = self._process_split(
            img_dir=self.gallery_dir,
            anno_path=self.gallery_anno_path,
            training=False,
            relabel=False
        )

        # Optional aliases if some other code wants query/gallery names
        self.query = self.val
        self.gallery = self.test

        if verbose:
            self.logger.info("=> Market1501 Images and Captions are loaded")
            self.show_dataset_info()

    def _resolve_caption_file(self, stem: str) -> str:
        candidates = [
            op.join(self.root, stem),
            op.join(self.root, stem + '.jsonl'),
            op.join(self.root, stem + '.json'),
        ]
        for p in candidates:
            if op.exists(p):
                return p
        raise RuntimeError(
            f"Caption file for '{stem}' not found. Tried: {candidates}"
        )

    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not op.exists(self.train_dir):
            raise RuntimeError(f"'{self.train_dir}' is not available")
        if not op.exists(self.query_dir):
            raise RuntimeError(f"'{self.query_dir}' is not available")
        if not op.exists(self.gallery_dir):
            raise RuntimeError(f"'{self.gallery_dir}' is not available")
        if not op.exists(self.train_anno_path):
            raise RuntimeError(f"'{self.train_anno_path}' is not available")
        if not op.exists(self.query_anno_path):
            raise RuntimeError(f"'{self.query_anno_path}' is not available")
        if not op.exists(self.gallery_anno_path):
            raise RuntimeError(f"'{self.gallery_anno_path}' is not available")

    def _extract_pid_camid(self, img_path: str):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        m = pattern.search(op.basename(img_path))
        if m is None:
            raise RuntimeError(f"Cannot parse pid/camid from: {img_path}")
        pid, camid = map(int, m.groups())
        return pid, camid - 1  # camid to 0-based

    def _load_annotation_records(self, anno_path: str) -> List[Any]:
        with open(anno_path, 'r', encoding='utf-8') as f:
            raw = f.read().strip()

        if not raw:
            return []

        # Try normal JSON first
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                if 'annotations' in obj and isinstance(obj['annotations'], list):
                    return obj['annotations']
                if 'data' in obj and isinstance(obj['data'], list):
                    return obj['data']
                # dict mapping filename -> caption(s)
                records = []
                for k, v in obj.items():
                    records.append({
                        'img_path': k,
                        'captions': v
                    })
                return records
        except json.JSONDecodeError:
            pass

        # Fallback: JSONL
        records = []
        with open(anno_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _extract_image_name(self, rec: Any) -> str:
        if isinstance(rec, str):
            return op.basename(rec)

        if not isinstance(rec, dict):
            return None

        image_keys = [
            'img_path', 'image_path', 'file_path', 'filename',
            'file_name', 'image', 'img', 'path', 'name'
        ]
        for k in image_keys:
            if k in rec and rec[k]:
                return op.basename(str(rec[k]))

        return None

    def _extract_captions(self, rec: Any) -> List[str]:
        if isinstance(rec, str):
            return [rec.strip()] if rec.strip() else []

        if not isinstance(rec, dict):
            return []

        caption_keys = [
            'captions', 'caption', 'texts', 'text',
            'description', 'descriptions'
        ]

        value = None
        for k in caption_keys:
            if k in rec:
                value = rec[k]
                break

        if value is None:
            return []

        if isinstance(value, str):
            value = [value]
        elif not isinstance(value, list):
            value = [value]

        captions = []
        for x in value:
            if isinstance(x, str):
                x = x.strip()
                if x:
                    captions.append(x)
            elif x is not None:
                x = str(x).strip()
                if x:
                    captions.append(x)

        return captions

    def _load_caption_map(self, anno_path: str) -> Dict[str, List[str]]:
        records = self._load_annotation_records(anno_path)
        caption_map = {}

        for rec in records:
            img_name = self._extract_image_name(rec)
            captions = self._extract_captions(rec)

            if img_name is None:
                continue
            if len(captions) == 0:
                continue

            if img_name not in caption_map:
                caption_map[img_name] = []
            caption_map[img_name].extend(captions)

        return caption_map

    def _process_split(self, img_dir: str, anno_path: str, training=False, relabel=False):
        img_paths = sorted(glob.glob(op.join(img_dir, '*.jpg')))
        caption_map = self._load_caption_map(anno_path)

        # collect valid image paths first
        valid_img_paths = []
        original_pid_container = set()

        for img_path in img_paths:
            pid, _ = self._extract_pid_camid(img_path)
            if pid == -1:
                continue
            original_pid_container.add(pid)
            valid_img_paths.append(img_path)

        pid2label = None
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(sorted(original_pid_container))}

        missing_caption_images = []

        if training:
            dataset = []
            image_id = 0

            for img_path in valid_img_paths:
                img_name = op.basename(img_path)
                pid, _ = self._extract_pid_camid(img_path)
                captions = caption_map.get(img_name, [])

                if len(captions) == 0:
                    missing_caption_images.append(img_name)
                    image_id += 1
                    continue

                pid_out = pid2label[pid] if relabel else pid

                for caption in captions:
                    dataset.append((pid_out, image_id, img_path, caption))

                image_id += 1

            pid_container = set(pid2label.values()) if relabel else set(original_pid_container)

            if relabel:
                expected = set(range(len(pid_container)))
                assert pid_container == expected, \
                    f"Train relabel is not contiguous: got {sorted(pid_container)[:10]} ..."

        else:
            dataset = {
                "image_pids": [],
                "img_paths": [],
                "caption_pids": [],
                "captions": []
            }

            for img_path in valid_img_paths:
                img_name = op.basename(img_path)
                pid, _ = self._extract_pid_camid(img_path)
                captions = caption_map.get(img_name, [])

                dataset["image_pids"].append(pid)
                dataset["img_paths"].append(img_path)

                if len(captions) == 0:
                    missing_caption_images.append(img_name)
                    continue

                for caption in captions:
                    dataset["captions"].append(caption)
                    dataset["caption_pids"].append(pid)

            pid_container = set(original_pid_container)

        if len(missing_caption_images) > 0:
            self.logger.warning(
                f"[Market1501] {len(missing_caption_images)} images in '{op.basename(img_dir)}' "
                f"have no matched captions from '{op.basename(anno_path)}'. "
                f"First 10: {missing_caption_images[:10]}"
            )

        return dataset, pid_container
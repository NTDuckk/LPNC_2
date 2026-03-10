import os.path as op
import re
import json

from .bases import BaseDataset


class Market1501(BaseDataset):
    """
    Market1501 with text captions for text-to-image person re-identification.

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)

    Caption annotations are read from JSONL files located alongside this module.
    Each line: {"image_path": "XXXX_cXsX_XXXXXX_XX.jpg", "caption": "..."}
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root='', verbose=True):
        super(Market1501, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.train_dir = op.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = op.join(self.dataset_dir, 'query')
        self.gallery_dir = op.join(self.dataset_dir, 'bounding_box_test')

        # JSONL caption files located alongside this Python file
        jsonl_dir = op.dirname(op.abspath(__file__))
        self.train_jsonl = op.join(jsonl_dir, 'market1501_bounding_box_train.jsonl')
        self.test_jsonl = op.join(jsonl_dir, 'market1501_bounding_box_test.jsonl')
        self.query_jsonl = op.join(jsonl_dir, 'market1501_query.jsonl')

        self._check_before_run()

        # Read annotations from JSONL
        train_annos = self._read_jsonl(self.train_jsonl, self.train_dir)
        gallery_annos = self._read_jsonl(self.test_jsonl, self.gallery_dir)
        query_annos = self._read_jsonl(self.query_jsonl, self.query_dir)

        self.train_annos = train_annos
        self.test_annos = gallery_annos
        self.val_annos = self.test_annos

        # Process splits
        self.train, self.train_id_container = self._process_train(train_annos)
        self.test, self.test_id_container = self._process_test(query_annos, gallery_annos)
        self.val = self.test
        self.val_id_container = self.test_id_container

        if verbose:
            self.logger.info("=> Market1501 Images and Captions are loaded")
            self.show_dataset_info()

    def _read_jsonl(self, jsonl_path, img_dir):
        """Read JSONL caption file and return list of annotation dicts.
        Skips junk images (pid == -1).
        """
        annos = []
        pattern = re.compile(r'([-\d]+)_c(\d)')
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                image_name = entry['image_path']
                caption = entry['caption']

                match = pattern.search(image_name)
                pid = int(match.group(1))
                camid = int(match.group(2))

                if pid == -1:
                    continue  # skip junk images

                img_path = op.join(img_dir, image_name)
                annos.append({
                    'pid': pid,
                    'camid': camid,
                    'img_path': img_path,
                    'caption': caption,
                })
        return annos

    def _process_train(self, annos):
        """Process training annotations into (pid, image_id, img_path, caption) tuples.
        PIDs are relabeled to be contiguous starting from 0.
        """
        pid_container = set()
        for anno in annos:
            pid_container.add(anno['pid'])

        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        image_id = 0
        for anno in annos:
            pid = pid2label[anno['pid']]
            dataset.append((pid, image_id, anno['img_path'], anno['caption']))
            image_id += 1

        relabeled_pids = set(pid2label.values())
        return dataset, relabeled_pids

    def _process_test(self, query_annos, gallery_annos):
        """Process test data: gallery images as image set, query captions as text set."""
        pid_container = set()

        img_paths = []
        image_pids = []
        captions = []
        caption_pids = []

        # Gallery images
        for anno in gallery_annos:
            pid = anno['pid']
            pid_container.add(pid)
            img_paths.append(anno['img_path'])
            image_pids.append(pid)

        # Query captions
        for anno in query_annos:
            pid = anno['pid']
            pid_container.add(pid)
            captions.append(anno['caption'])
            caption_pids.append(pid)

        dataset = {
            "image_pids": image_pids,
            "img_paths": img_paths,
            "caption_pids": caption_pids,
            "captions": captions,
        }
        return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper."""
        for path in [self.dataset_dir, self.train_dir, self.query_dir,
                     self.gallery_dir, self.train_jsonl, self.test_jsonl,
                     self.query_jsonl]:
            if not op.exists(path):
                raise RuntimeError(f"'{path}' is not available")

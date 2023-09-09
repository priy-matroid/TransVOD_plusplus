# Modified by Qianyu Zhou and Lu He
# ------------------------------------------------------------------------
# TransVOD++
# Copyright (c) 2022 Shanghai Jiao Tong University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) SenseTime. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_single as T
from torch.utils.data.dataset import ConcatDataset
import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image

class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, ann_ignores, transforms, is_train, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.is_train = is_train
        if self.is_train:
            self.ids_train = []
            img_infos = self.coco.loadImgs(self.ids)
            for img_info in img_infos:
                if img_info['is_vid_train_frame']:
                    self.ids_train.append(img_info['id'])
        if ann_ignores:
            self.coco_ignores = COCO(ann_ignores)
        else:
            self.coco_ignores = None


    def __len__(self):
        if self.is_train:
            return len(self.ids_train)
        else:
            return len(self.ids)

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        
        coco = self.coco
        if self.is_train:
            img_id = self.ids_train[idx]
        else:
            img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)

        image_id = img_id
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)
        if self.coco_ignores:
            # print('ignoring!!')
            ann_ids_ignore = self.coco_ignores.getAnnIds(imgIds=[img_id], iscrowd=None)
            anns_ignore = self.coco_ignores.loadAnns(ann_ids_ignore)
            img_array = np.array(img)

            for ignore in anns_ignore:
                bbox = ignore['bbox']
                class_id, x1, y1, width, height = ignore['category_id'], bbox[0], bbox[1], bbox[2], bbox[3]
                x_min, y_min, x_max, y_max = x1, y1, x1+width, y1+height
                img_array[y_min:y_max, x_min:x_max] = [0, 0, 0]
            img = Image.fromarray(img_array, 'RGB')

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, img_side):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train_vid' or image_set == "train_det" or image_set == "train_joint":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize([img_side], max_size=1000),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([img_side], max_size=1000),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    # root = Path(args.vid_path)
    # assert root.exists(), f'provided COCO path {root} does not exist'
    # mode = 'instances'
    # PATHS = {
    #     # "train_joint": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json'), (root / "Data" / "VID", root / "annotations_10true" / 'imagenet_vid_train.json')],
    #     "train_det": [(root / "Data" / "DET", root / "annotations" / 'imagenet_det_30plus1cls_vid_train.json')],
    #     "train_vid": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_train.json')],
    #     "train_joint": [(root / "Data" , root / "annotations" / 'imagenet_vid_train_joint_30.json')],
    #     "val": [(root / "Data" / "VID", root / "annotations" / 'imagenet_vid_val.json')],
    # }
    args.vid_path = args.data_root
    root = Path(args.vid_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    # PATHS = {
    #     "train_vid": [(root, os.path.join(root, 'UAV_train_every10.json'), True, os.path.join(root, 'UAV_train_every10_ignores.json'))],
    #     "val": [(root, os.path.join(root, 'UAV_val_every10.json'), False, os.path.join(root, 'UAV_val_every10_ignores.json'))]
    # }

    # PATHS = {
    #     "train_vid": [(root, os.path.join(root, 'VisDrone_VID_train_overfit.json'), True, None)],
    #     "val": [(root, os.path.join(root, 'VisDrone_VID_train_overfit.json'), True, None)]
    # }

    PATHS = {
        "train_vid": [(root, os.path.join(root, 'CBP_coco_train.json'), True, None), (root, os.path.join(root, 'train_DET.json'), True, None)],
        "val": [(root, os.path.join(root, 'CBP_coco_val.json'), True, None)]
    }

    datasets = []
    print(args.img_side)
    for (img_folder, ann_file, is_train, ann_ignores) in PATHS[image_set]:
        dataset = CocoDetection(img_folder, ann_file, ann_ignores, transforms=make_coco_transforms(image_set, args.img_side), is_train=is_train, return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

    

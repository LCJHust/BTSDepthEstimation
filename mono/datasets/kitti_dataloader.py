#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from .preprocess import get_transform

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def readPathFiles(file_path, rgb_dir, gt_dir, location='dm6'):
    im_gt_paths = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if location == 'dm6':
                im_path = os.path.join(rgb_dir, line.split()[0])
            else:
                im_path = os.path.join(rgb_dir, line.split()[0].replace('png', 'jpg'))
            gt_path = os.path.join(gt_dir, line.split()[1])
            im_gt_paths.append((im_path, gt_path))
    return im_gt_paths

def pil_loader(path, rgb=True):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if rgb:
            return img.convert('RGB')
        else:
            return img.convert('I')

class KittiRaw(Dataset):
    def __init__(self, cfg, type, transform=None):
        self.rgb_root = cfg['img_root']
        self.gt_root = cfg['gt_root']
        self.mode = type
        self.location = cfg['location']
        self.transform = transform
        self.im_gt_paths = None
        self.pairs_root = cfg['pairs_root']
        self.input_shape = cfg.data['output_size']

        if self.mode == 'train':
            self.im_gt_paths = readPathFiles(os.path.join(self.pairs_root, 'kitti_train_pairs.txt'),
                                             self.rgb_root, self.gt_root, self.location)
        elif self.mode == 'test':
            self.im_gt_paths = readPathFiles(os.path.join(self.pairs_root, 'kitti_test_pairs.txt'),
                                             self.rgb_root, self.gt_root, self.location)

        elif self.mode == 'eval':
            self.im_gt_paths = readPathFiles(os.path.join(self.pairs_root, 'kitti_test_pairs.txt'),
                                             self.rgb_root, self.gt_root, self.location)

        elif self.mode == 'vis':
            self.im_gt_paths = readPathFiles(os.path.join(self.pairs_root, 'kitti_test_pairs.txt'),
                                             self.rgb_root, self.gt_root, self.location)[::20]

        else:
            print('no mode named as ', self.mode)

        # FIXME: due to mmdetection
        self.flag = np.zeros(len(self.im_gt_paths), dtype=np.int64)

    def __len__(self):
        return len(self.im_gt_paths)

    def __getitem__(self, idx):
        im_path, gt_path = self.im_gt_paths[idx]

        if self.mode == 'train':
            im_path = os.path.join(self.rgb_root, im_path)
            gt_path = os.path.join(self.gt_root, gt_path)
        elif self.mode == 'eval':
            im_path = os.path.join(self.rgb_root, im_path)
            gt_path = os.path.join(self.gt_root, gt_path)
        elif self.mode == 'vis':
            im_path = os.path.join(self.rgb_root, im_path)
            gt_path = os.path.join(self.gt_root, gt_path)
        else:
            raise NotImplementedError

        im = pil_loader(im_path.replace('png', 'jpg'))
        gt = pil_loader(gt_path, rgb=False)

        if self.mode == 'train':
            w, h = im.size
            th, tw = self.input_shape[0], self.input_shape[1]
            assert th < h, tw < w

            top_margin = int(round((h - th)))
            left_margin = int(round((w - tw) / 2.))

            im = im.crop((left_margin, top_margin, left_margin + tw, top_margin + th))

            gt = np.ascontiguousarray(gt, dtype=np.float32) / 256.0
            gt = gt.reshape((h, w))
            gt = gt[top_margin:top_margin + th, left_margin:left_margin + tw]

            processed = get_transform(augment=False)
            im = processed(im)

        sample = {}

        sample['leftImage'] = im
        sample['left_gt'] = gt

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def name(self):
        return 'KittiRaw'

if __name__ == "__main__":
    from mmcv import Config
    cfg = '../../config/bts_cfg.py'
    cfg = Config.fromfile(cfg)
    kitti_dataset = KittiRaw(cfg, 'train')
    print(kitti_dataset[0]['left_gt'].max())
    print("*")
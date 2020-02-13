#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)
# Get the img and depth

import os
import os.path
import numpy as np
from PIL import Image

import torch.utils.data as data
import random
from . import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_kitti_files(data_root, gt_root):

    left_folder = 'rawdata/2011_09_26/2011_09_26_drive_0084_sync/image_02/data'
    depth_folder = 'train/2011_09_26_drive_0084_sync/proj_depth/groundtruth/image_02'
    # depth_folder = 'depth_gt/train_gt/'

    all_left_img = []
    depth_img = []
    left_img_folder = os.path.join(data_root, left_folder)
    depth_img_folder = os.path.join(gt_root, depth_folder)

    for im in sorted(os.listdir(depth_img_folder)):
        depth_img.append(os.path.join(depth_img_folder, im))
        all_left_img.append(os.path.join(left_img_folder, im))

    return all_left_img, depth_img, all_left_img[100:], depth_img[100:] # train, val


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


class KITTIRaw(data.Dataset):
    def __init__(self, cfg, training, loader=default_loader, dploader=disparity_loader):

        self.data_root = cfg['data_root']
        self.gt_root = cfg['gt_root']
        self.input_shape = cfg.get('input_shape', (260, 513))

        left, depth, \
        test_left, test_depth = get_kitti_files(self.data_root, self.gt_root)

        if training:
            self.left = left
            self.depth = depth
        else:
            self.left = test_left
            self.depth = test_depth

        self.loader = loader
        self.dploader = dploader
        self.training = training

        #FIXME: due to mmdetection
        self.flag = np.zeros(len(self.left), dtype=np.int64)

    def __getitem__(self, index):
        left = self.left[index]
        depth = self.depth[index]

        left_img = self.loader(left)
        dataL = self.dploader(depth)

        if self.training:
            w, h = left_img.size
            th, tw = self.input_shape[0], self.input_shape[1]
            assert th < h, tw < w

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            dataL = dataL.reshape((h, w))
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
        else:
            w, h = left_img.size

            # crop_h = 320
            # crop_w = 1024

            crop_h = 260
            crop_w = 513

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))

            dataL = dataL.crop((w - crop_w, h - crop_h, w, h))
            dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)

        sample = {}
        sample['img_left'] = left_img
        sample['data_L'] = dataL

        return sample

    def __len__(self):
        return len(self.left)

    @property
    def name(self):
        return 'KITTIRaw'

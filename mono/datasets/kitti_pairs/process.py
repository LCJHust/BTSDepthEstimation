#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

with open('kitti_train_pairs.txt', 'r') as f:
    lines = f.readlines()

with open('kitti_train_pairs.txt', 'w') as f:
    for line in lines:
        rgb, gt = line.split(" ")
        # rgb = rgb.replace('/data4/kitti_raw/rawdata/', '')
        # gt = gt.replace('/data4/kitti_raw/', '')
        f.writelines(rgb[25:]+ " " + gt[17:])

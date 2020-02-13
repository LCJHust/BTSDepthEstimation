#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

from .kitti_dataloader import KittiRaw


def get_dataset(cfg, training=True):
    return KittiRaw(cfg, 'train')

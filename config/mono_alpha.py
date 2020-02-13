#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

model = dict(
    n_channels = 3,
    n_classes = 1
)

dataset_type='kitti_raw'

# data_root = '/node01/data5/kitti_raw/'
# gt_root = '/node01/data1/kitti/kitti_raw_depth/'

data_root = '/data4/kitti_raw/'
gt_root = '/data4/kitti_raw/'

import os
print(os.path.exists(data_root))
print(os.path.dirname(__file__))

data = dict(
    imgs_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        gt_root=gt_root,
        input_shape=[260, 513],
    ),
    val=dict(
        type='raw_15',
        data_root = '/node01/odo/raw_15'
    )
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 30
validate_interval=1
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

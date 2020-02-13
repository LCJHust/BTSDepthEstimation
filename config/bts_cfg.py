#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)
#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Caojia

import numpy as np

# Dataset Settings
dataset_type = 'KittiRaw'
img_root = '/data5/kitti_raw/'
gt_root = '/data1/kitti/kitti_raw_depth/'
location = 'dm6'
pairs_root = '/home/caojia/Projects/DepthEstimation/mono/datasets/kitti_pairs/'
pretrained_model = '/home/caojia/densenet161.pth'

data = dict(
    shift=1.0,
    alpha=1,
    beta=90.0,
    K=80.0,
    output_size=(352, 1216),
    rgb_dir=img_root,
    gt_dir=gt_root,
    K_inv=np.linalg.inv(np.array([[721.0, 0, 605.0],
                                     [0, 721.0, 176.0],
                                     [0,     0,     1]])),
    train=dict(
        type='train',
        rd_rotation=np.random.uniform(-5.0, 5.0),
        rd_flip=np.random.uniform(0.0, 1.0) < 0.5,
        # color_jitter=(0.4, 0.4, 0.4),
    ),
    eval=dict(
        type='eval'
    ),
    imgs_per_gpu=4,
    workers_per_gpu=8,
)


model = dict(
        type='LPGNet',
        depth_num_layers = 161,
        input_shape = [352, 1216],
        max_depth = 80,
        fxy = [721.0]
    )


optimizer = dict(type='Adam', lr=0.01, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 2,
    step=[8, 11])
checkpoint_config = dict(interval=1)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'), # log the screen output into txt file
        dict(type='TensorboardLoggerHook'),
    ])

apex = dict( # https://nvidia.github.io/apex/amp.html
    synced_bn=True, # whether to use apex.synced_bn
    use_mixed_precision=False, # whether to use apex for mixed precision training
    type="float32", # the model weight type: float16 or float32
    loss_scale=512, # the factor when apex scales the loss value
)


total_epochs = 20
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
validate = True
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '/home/caojia/Projects/DepthEstimation/tmp/'
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Caojia


from __future__ import division

from mono.datasets.get_dataset import get_dataset
from mono.model import LPGNet
from mono.core.evaluation.eval_hooks import DistEvalHook
from mmcv import Config
import torch
import numpy as np
import math

config = 'config/bts_cfg.py'
cfg = Config.fromfile(config)

model_path = './tmp/epoch_30.pth'
val_dataset = get_dataset(cfg, training=False)

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the element of x."""


class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        output = output.float()
        target = target.float()
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output.float() - target.float()).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())
        

model = LPGNet(cfg)






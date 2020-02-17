#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image

from infer import BTSInfer

import matplotlib.pyplot as plt

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    gt_mask = gt > 0
    gt = gt[gt_mask]
    pred = pred[gt_mask]

    pred[pred<MIN_DEPTH] = MIN_DEPTH
    pred[pred>MAX_DEPTH] = MAX_DEPTH

    thresh = np.maximum((gt / pred), (pred / gt))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "a1": delta1,
        "a2": delta2,
        "a3": delta3
    }

class Evaluator():
    def __init__(self, cfg):
        self.cfg = cfg
        self.bts_infer = BTSInfer(cfg)

    def __transfrom(self, input_gt):
        original_width, original_height = input_gt.size
        output_width, output_height = self.cfg.data['output_size'][0], self.cfg.data['output_size'][1]

        input_gt = np.array(input_gt).astype(np.float32) / 255.0

        top_margin = int(original_height - output_width)
        left_margin = int((original_width - output_height) / 2)

        input_gt = input_gt[top_margin:top_margin + output_width, left_margin:left_margin + output_height]

        return input_gt

    def eval(self, img_folder, gt_depth_folder, exist_prediction=True):
        gt_files = os.listdir(gt_depth_folder)[:50]
        img_files = os.listdir(img_folder)[:50]

        abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3 = 0, 0, 0, 0, 0, 0, 0
        for (pred_f, gt_f) in zip(img_files, gt_files):
            pred = Image.open(os.path.join(img_folder, pred_f)).convert('RGB')
            gt = Image.open(os.path.join(gt_depth_folder, gt_f)).convert('I')


            if not exist_prediction:
                pred = self.bts_infer.predict(pred)
                pred = pred.squeeze().cpu().numpy()

            gt = self.__transfrom(gt)

            eval_scores = compute_errors(gt, pred)
            abs_rel += eval_scores['abs_rel']
            sq_rel += eval_scores['sq_rel']
            rmse += eval_scores['rmse']
            rmse_log += eval_scores['rmse_log']
            delta1 += eval_scores['a1']
            delta2 += eval_scores['a2']
            delta3 += eval_scores['a3']

        return {
            "abs_rel": abs_rel / len(img_files),
            "sq_rel": sq_rel / len(img_files),
            "rmse": rmse / len(img_files),
            "rmse_log": rmse_log / len(img_files),
            "delta1": delta1 / len(img_files),
            "delta2": delta2 / len(img_files),
            "delta3": delta3 / len(img_files)
        }



if __name__ == "__main__":
    from mmcv import ConfigDict
    import matplotlib.pyplot as plt

    infer_cfg = dict(
        model_path='./tmp/epoch_16.pth',
        pretrained_model='/home/caojia/densenet161.pth',

        data=dict(
            output_size=(352, 1216),
            imgs_per_gpu=1
        ),
        model=dict(
            type='LPGNet',
            depth_num_layers=161,
            input_shape=[352, 1216],
            max_depth=80,
            fxy=[721.0]
        )
    )

    infer_cfg = ConfigDict(infer_cfg)

    evaluator = Evaluator(infer_cfg)
    img_folder = '/home/caojia/kitti_eigen_test/image_02/'
    gt_folder = '/home/caojia/kitti_eigen_test/groundtruth/'

    print(evaluator.eval(img_folder, gt_folder, False))



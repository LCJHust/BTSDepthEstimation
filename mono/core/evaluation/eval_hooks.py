#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Author: Duanzhixiang(zhixiangduan@deepmotion.ai)

import os
import os.path as osp
import cv2

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import Hook
from mmcv.parallel import scatter, collate
from torch.utils.data import Dataset
from .pixel_error import *

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

def change_input_variable(data):
    for k, v in data.items():
        data[k] = torch.as_tensor(v).float()
    return data

def unsqueeze_input_variable(data):
    for k, v in data.items():
        data[k] = torch.unsqueeze(v, dim=0)
    return data

class NonDistEvalHook(Hook):
    def __init__(self, dataset, cfg):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.interval = cfg.get('interval', 1)
        self.out_path = cfg.get('work_dir', './')

    def after_train_epoch(self, runner):
        print('evaluation..............................................')

        abs_rel = AverageMeter()
        sq_rel = AverageMeter()
        rmse = AverageMeter()
        rmse_log = AverageMeter()
        a1 = AverageMeter()
        a2 = AverageMeter()
        a3 = AverageMeter()

        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()

        for idx in range(self.dataset.__len__()):
            data = self.dataset[idx]
            data = change_input_variable(data)
            data = unsqueeze_input_variable(data)
            with torch.no_grad():
                result = runner.model(data)

            pred_disp, _ = disp_to_depth(result[("disp", 0)])
            pred_disp = pred_disp.cpu()[0, 0].numpy()

            gt_depth = data['gt_depth'].cpu()[0].numpy()
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_, a3_ = \
                compute_errors(gt_depth, pred_depth)

            abs_rel.update(abs_rel_)
            sq_rel.update(sq_rel_)
            rmse.update(rmse_)
            rmse_log.update(rmse_log_)
            a1.update(a1_)
            a2.update(a2_)
            a3.update(a3_)
            print('a1_ is ', a1_)

        print('a1 is ', a1.avg)

class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.interval = interval

    def after_train_epoch(self, runner):
        print('evaluation..............................................')

        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data = change_input_variable(data)

            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(data_gpu)

            pred_disp, _ = disp_to_depth(result[("disp", 0)])
            pred_disp = pred_disp.cpu()[0, 0].numpy()

            gt_depth = data['gt_depth'].cpu().numpy()
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_, a3_ = \
                compute_errors(gt_depth, pred_depth)


            result = {}
            result['abs_rel'] = abs_rel_
            result['sq_rel'] = sq_rel_
            result['rmse'] = rmse_
            result['rmse_log'] = rmse_log_
            result['a1'] = a1_
            result['a2'] = a2_
            result['a3'] = a3_
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError

class DistEvalMonoHook(DistEvalHook):

    def evaluate(self, runner, results):
        if mmcv.is_str(results):
            assert results.endswith('.pkl')
            results = mmcv.load(results)
        elif not isinstance(results, list):
            raise TypeError(
                'results must be a list of numpy arrays or a filename, not {}'.
                    format(type(results)))

        abs_rel = AverageMeter()
        sq_rel = AverageMeter()
        rmse = AverageMeter()
        rmse_log = AverageMeter()
        a1 = AverageMeter()
        a2 = AverageMeter()
        a3 = AverageMeter()

        print('results len is ', results.__len__())

        for result in results:
            abs_rel.update(result['abs_rel'])
            sq_rel.update(result['sq_rel'])
            rmse.update(result['rmse'])
            rmse_log.update(result['rmse_log'])
            a1.update(result['a1'])
            a2.update(result['a2'])
            a3.update(result['a3'])

        runner.log_buffer.output['abs_rel'] = abs_rel.avg
        runner.log_buffer.output['sq_rel'] = sq_rel.avg
        runner.log_buffer.output['rmse'] = rmse.avg
        runner.log_buffer.output['rmse_log'] = rmse_log.avg
        runner.log_buffer.output['a1'] = a1.avg
        runner.log_buffer.output['a2'] = a2.avg
        runner.log_buffer.output['a3'] = a3.avg
        runner.log_buffer.ready = True

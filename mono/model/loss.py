#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

from __future__ import absolute_import, division, print_function

import torch

def weighted_multi_label_classification_loss(softmax_output, information_matrix, gt):
    """
    :param softmax_output: NxC
    :param information_matrix: CxC
    :param gt: N, max is C
    :return:
    """
    weighted_gt = information_matrix[gt]
    loss = torch.sum(torch.mul(-torch.log(softmax_output), weighted_gt))
    loss /= softmax_output.size()[0]

    return loss

def scale_invariant_error_loss(pred, gt):
    pred = pred.squeeze()
    gt = gt.squeeze()
    mask = gt > 1.0
    mask.detach_()

    mask_pred = pred > 1
    d = torch.log(pred[mask & mask_pred]) - torch.log(gt[mask & mask_pred])
    loss = (torch.mean(d ** 2) - 0.85 * (torch.mean(d) ** 2)) * 10.0

    return loss
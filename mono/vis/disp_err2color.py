#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

from __future__ import absolute_import, division, print_function

import numpy as np

def disp_err_image(disp_est, disp_gt):
    """
    Calculate the error map between disparity estimation and disparity ground-truth
    hot color -> big error, cold color -> small error
    Inputs:
        disp_est: numpy array, disparity estimation map in (Height, Width) layout, range [0,255]
        disp_gt:  numpy array, disparity ground-truth map in (Height, Width) layout, range [0,255]
    Outputs:
        disp_err: numpy array, disparity error map in (Height, Width, 3) layout, range [0,255]
    """
    disp_shape = disp_gt.shape

    # error color map with interval (0, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24, 48, inf)/3.0
    # different interval corresponds to different 3-channel projection
    cols = np.array([
        [0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
        [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
        [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
        [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
        [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
        [3 / 3.0, 6 / 3.0, 254, 224, 144],
        [6 / 3.0, 12 / 3.0, 253, 174, 97],
        [12 / 3.0, 24 / 3.0, 244, 109, 67],
        [24 / 3.0, 48 / 3.0, 215, 48, 39],
        [48 / 3.0, float("inf"), 165, 0, 38]
    ])

    # get the error (<3px or <5%) map
    tau = np.array([3.0, 0.05])

    E = np.abs(disp_est - disp_gt)

    t1 = E / tau[0]
    t2 = E
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if disp_gt[i, j] > 0:
                t2[i, j] = E[i, j] / disp_gt[i, j] / tau[1]
    E = np.minimum(t1, t2)

    # based on error color map, project the E within [cols[i,0], cols[i,1]] into 3-channel color image
    disp_err = np.zeros((disp_shape[0], disp_shape[1], 3), dtype='uint8')
    for c_i in range(cols.shape[0]):
        for i in range(disp_shape[0]):
            for j in range(disp_shape[1]):
                # disp_gt[i,j]>0 &  cols[c_i,0] <= E[i,j] <= cols[c_i,1]
                if disp_gt[i, j] != 0 and E[i, j] >= cols[c_i, 0] and E[i, j] <= cols[c_i, 1]:
                    disp_err[i, j, 0] = int(cols[c_i, 2])
                    disp_err[i, j, 1] = int(cols[c_i, 3])
                    disp_err[i, j, 2] = int(cols[c_i, 4])

    return disp_err
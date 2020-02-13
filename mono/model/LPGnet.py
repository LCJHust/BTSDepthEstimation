#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os

from .depth_encoder import DepthEncoder
from .utils import PlaneDepthModule
from .loss import scale_invariant_error_loss

from ..vis.disp2color import disp_to_color
from ..vis.disp_err2color import disp_err_image

class LPGNet(nn.Module):
    def __init__(self, cfg):
        super(LPGNet, self).__init__()

        self.cfg = cfg
        self.DepthEncoder = DepthEncoder(cfg.model['depth_num_layers'], cfg['pretrained_model'])
        input_shape = cfg.model['input_shape']
        self.max_depth = cfg.model['max_depth']
        self.fxy = cfg.model['fxy']
        self.bs = cfg.data['imgs_per_gpu']
        # self.calib_matrix_inv = torch.FloatTensor(cfg.data['K_inv']).cuda()

        self.lpg8 = PlaneDepthModule((input_shape[0]//8, input_shape[1]//8), 64, 8, self.max_depth)
        self.lpg4 = PlaneDepthModule((input_shape[0]//4, input_shape[1]//4), 32, 4, self.max_depth)
        self.lpg2 = PlaneDepthModule((input_shape[0]//2, input_shape[1]//2), 16, 2, self.max_depth)

        self.global_step = 0

        self.upconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(2208)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(2592, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(256)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(448, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.atrous_conv_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.atrous_conv_6 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=6, padding=6),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.atrous_conv_12 = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=12, padding=12),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.atrous_conv_18 = nn.Sequential(
            nn.Conv2d(640, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=18, padding=18),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.atrous_conv_24 = nn.Sequential(
            nn.Conv2d(704, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=24, padding=24),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv_daspp = nn.Sequential(
            nn.Conv2d(448, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(64)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(161, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(129, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.BatchNorm2d(16)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(19, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        self.global_step += 1

        dense_features, skips = self.DepthEncoder(inputs['leftImage'])
        focal = self.fxy * self.bs

        feature_16 = skips[3]
        feature_8  = skips[2]
        feature_4  = skips[1]
        feature_2  = skips[0]

        upconv5 = self.upconv5(dense_features)
        concat5 = torch.cat([upconv5, feature_16], dim=1)
        iconv5 = self.conv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat([upconv4, feature_8], dim=1)
        iconv4 = self.conv4(concat4)

        daspp_3 = self.atrous_conv_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.atrous_conv_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.atrous_conv_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.atrous_conv_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.atrous_conv_24(concat4_5)

        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.conv_daspp(concat4_daspp)

        depth_8x8_scaled, depth_8x8_scaled_ds = self.lpg8(daspp_feat, focal, 4)

        upconv3 = self.upconv3(daspp_feat)

        concat3 = torch.cat([upconv3, feature_4, depth_8x8_scaled_ds], dim=1)
        iconv3 = self.conv3(concat3)

        depth_4x4_scaled, depth_4x4_scaled_ds = self.lpg4(iconv3, focal, 2)
        upconv2 = self.upconv2(iconv3)

        concat2 = torch.cat([upconv2, feature_2, depth_4x4_scaled_ds], dim=1)
        iconv2 = self.conv2(concat2)

        depth_2x2_scaled, depth_2x2_scaled_ds = self.lpg2(iconv2, focal, 1)

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat([upconv1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled], dim=1)
        iconv1 = self.conv1(concat1)

        ds = iconv1 * self.max_depth

        output = {}
        output['ds'] = ds

        if self.training:
            gt = inputs['left_gt']
            if self.global_step % 1000 == 0:
                gt_mask = (gt[0] > 0).clone().data.cpu().numpy()
                vis_ds = ds.clone()[0, 0].data.cpu().numpy()
                vis_ds_half = depth_2x2_scaled.clone()[0, 0].data.cpu().numpy()
                vis_ds_forth = depth_4x4_scaled.clone()[0, 0].data.cpu().numpy()
                vis_ds_eigth = depth_8x8_scaled.clone()[0, 0].data.cpu().numpy()

                vis_ds[gt_mask == 0] = 0
                vis_ds_half[gt_mask == 0] = 0
                vis_ds_forth[gt_mask == 0] = 0
                vis_ds_eigth[gt_mask == 0] = 0

                vis = np.concatenate([disp_to_color(gt[0].squeeze().detach().cpu().numpy()),
                                      disp_to_color(vis_ds),
                                      disp_err_image(ds[0].squeeze().detach().cpu().numpy(),
                                                     gt[0].squeeze().detach().cpu().numpy()),
                                      disp_to_color(vis_ds_half),
                                      disp_to_color(vis_ds_forth),
                                      disp_to_color(vis_ds_eigth)], axis=0)
                img = Image.fromarray(vis.astype(np.uint8))
                img.save(os.path.join(self.cfg['work_dir'], 'vis_img_bts.jpg'))

            loss_SIE = scale_invariant_error_loss(ds, gt)
            losses = {
                'loss_SIE': loss_SIE
            }

            return losses
        else:
            return ds

        return ds


if __name__ == "__main__":
    import time
    from mmcv import ConfigDict

    cfg = dict(
        pretrained_model='/home/caojia/densenet161.pth',
        model = dict(
            depth_num_layers=161,

            input_shape=[608, 960],
            max_depth=80,
            fxy=[631.0]
        ),
        data = dict(
            imgs_per_gpu=2
        )
    )
    cfg = ConfigDict(cfg)

    net = LPGNet(cfg).cuda().eval()
    x = torch.randn((2, 3, 608, 960)).cuda()
    focal = [712.] * x.size()[0]  # camera focal_fxy, the length should be equal to input batch_size
    inputs = dict(
        leftImage=x,
        left_gt=x[:, 0, :, :]
    )
    torch.cuda.synchronize()
    s_t = time.time()

    y = net(inputs)
    torch.cuda.synchronize()
    print('inference time is ', time.time() - s_t)






#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlaneDepthModule(nn.Module):
    """https://arxiv.org/pdf/1907.10326.pdf equation(1)"""
    def __init__(self, input_size, channel, upratio, max_depth=80):
        super().__init__()

        self.input_size = input_size
        self.upratio = upratio
        self.max_depth = max_depth
        up_h = input_size[0] * upratio
        up_w = input_size[1] * upratio

        up_vu = torch.meshgrid(torch.arange(up_h), torch.arange(up_w))
        self.up_v_index = up_vu[0].contiguous().view(-1).cuda()
        self.up_u_index = up_vu[1].contiguous().view(-1).cuda()

        c = channel
        plane_conv = []
        while c >= 4:
            if c < 8:
                plane_conv.append(torch.nn.Conv2d(c, 4, 1))
                c = 4
                break
            else:
                out = c // 2
                plane_conv.append(torch.nn.Conv2d(c, out, 1))
                plane_conv.append(nn.BatchNorm2d(out))
                plane_conv.append(nn.ReLU())

            c = c // 2
        assert c == 4
        self.plane_conv = nn.Sequential(*plane_conv)

    def forward(self, x, focal, downratio):
        assert x.size()[-2:] == self.input_size, ('math size should be ', self.input_size,
                                             'your size is ', x.size()[-2:])
        org_h = x.size()[2]
        org_w = x.size()[3]

        x = self.plane_conv(x)
        x = torch.cat(
            [torch.tanh(x[:, 0:2, :, :]),
             torch.unsqueeze(torch.sigmoid(x[:, 2, :, :]), 1),
             torch.unsqueeze(torch.sigmoid(x[:, 3, :, :])*self.max_depth, dim=1)], 1
        )

        plane_params_norm = F.normalize(x[:, 0:3, :, :], p=2, dim=1)
        plane_params_dist = x[:, 3:4, :, :]
        plane_params = torch.cat([plane_params_norm, plane_params_dist], 1)
        depth_scaled = self.compute_depth(plane_params, focal) / self.max_depth
        depth_scaled_ds = F.interpolate(depth_scaled, scale_factor=1/downratio, mode='bilinear', align_corners=True)

        return depth_scaled, depth_scaled_ds

    def compute_depth(self, plane_equation, focal):
        """
        :param plane_equation: (B, C, H, W)
        :param upratio: 8, 4, 2
        :param focal: camera_focal_xy, assume fx == fy
        :return:
        """
        b = plane_equation.size()[0] # batchsize
        self.estimation_depth = torch.ones(b, 1, plane_equation.size()[2]*self.upratio,
                                           plane_equation.size()[3]*self.upratio).type_as(plane_equation)

        #TODO get rid of for loop to accelerate
        for i in range(b):
            # the focal may range among batches
            v = (self.up_v_index.float() % self.upratio - (self.upratio - 1)/2.0)/focal[i]
            u = (self.up_u_index.float() % self.upratio - (self.upratio - 1)/2.0)/focal[i]

            org_v = self.up_v_index // self.upratio
            org_u = self.up_u_index // self.upratio
            plane_params = plane_equation[i, :, org_v, org_u]
            a, b, c, d = plane_params[0, :], plane_params[1, :], plane_params[2, :], plane_params[3, :]

            numerator = d * torch.sqrt((u*u + v*v + 1.0))
            denominator = a*u + b*v + c
            self.estimation_depth[i, 0, self.up_v_index, self.up_u_index] = numerator/denominator

        return self.estimation_depth


if __name__ == "__main__":
    import time
    batch_size = 2
    x = torch.randn((batch_size, 32, 200, 320)).float().cuda()
    upratio = 2
    focal = [721.0] * batch_size

    module = PlaneDepthModule((200, 320), 32, upratio)
    module = module.cuda()

    torch.cuda.synchronize()
    start_time = time.time()

    estimate_depth = module(x, focal, 2)[0]
    torch.cuda.synchronize()
    print('time is: ', time.time() - start_time)
    print(estimate_depth.shape)
    print("*")


#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .densenet_backbone import densenet121, densenet161, densenet169, densenet201

class DepthEncoder(nn.Module):
    def __init__(self, num_layers, pretrained_path=None):
        super(DepthEncoder, self).__init__()

        densenets = {
            121: densenet121,
            161: densenet161,
            169: densenet169,
            201: densenet201
        }

        if num_layers not in densenets:
            raise ValueError("{} is not a valid number of densenet layers.".format(num_layers))

        self.encoder = densenets[num_layers]()

        if pretrained_path is not None:
            checkpoint = torch.load(pretrained_path)
            self.encoder.load_state_dict(checkpoint)

    def forward(self, x):
        self.features = []
        out = self.encoder.features.conv0(x)
        out = self.encoder.features.norm0(out)
        out = self.encoder.features.relu0(out)
        self.features.append(out)                 # 1/2

        out = self.encoder.features.pool0(out)
        self.features.append(out)                 # 1/4

        out = self.encoder.features.transition1(self.encoder.features.denseblock1(out))
        self.features.append(out)                 # 1/8

        out = self.encoder.features.transition2(self.encoder.features.denseblock2(out))
        self.features.append(out)                 # 1/16

        out = self.encoder.features.transition3(self.encoder.features.denseblock3(out))
        out = self.encoder.features.norm5(self.encoder.features.denseblock4(out))
        out = F.relu(out)

        return out, self.features

if __name__ == "__main__":
    import torchvision.models as models
    model = models.densenet161(True)



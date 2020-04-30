#!/usr/bin/env python
# -*-coding: utf-8 -*-
# Author: Liang Caojia(caojialiang@deepmotion.ai)

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from torchvision.transforms import Compose

import numpy as np

from mono.model import LPGNet
from mmcv.runner import load_checkpoint

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.255]

# torch.cuda.set_device(1)

class BTSInfer():
    def __init__(self, cfg):
        model = LPGNet(cfg)
        load_checkpoint(model, cfg['model_path'])
        self.model = model.cuda().eval()
        self.cfg = cfg

    def __infer_transform(self):
        infer_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
            ]
        )
        return infer_transforms

    def __transfrom(self, input_image):
        original_width, original_height = input_image.size
        output_width, output_height = self.cfg.data['output_size'][0], self.cfg.data['output_size'][1]

        input_image = np.array(input_image).astype(np.float32) / 255.0

        top_margin = int(original_height - output_width)
        left_margin = int((original_width - output_height) / 2)

        input_image = input_image[top_margin:top_margin + output_width, left_margin:left_margin + output_height, :]

        return input_image

    def predict(self, input_image):
        with torch.no_grad():
            input_image = self.__transfrom(input_image)
            trans = self.__infer_transform()
            input_image = trans(input_image)
            input_image = input_image.unsqueeze(0).cuda()
            input_data = {
                'leftImage': input_image
            }
            pred_depth = self.model(input_data)

        return pred_depth


if __name__ == "__main__":


    from mmcv import ConfigDict
    from PIL import Image
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

    inferBTS = BTSInfer(infer_cfg)
    input_image_path = '/data5/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000699.jpg'
    input_image = Image.open(input_image_path).convert('RGB')
    infer_depth = inferBTS.predict(input_image)

    print(infer_depth.shape)

    plt.imshow(infer_depth.squeeze().data.cpu().numpy())
    plt.show()
'''
Modified VGGNet in Pytorch.

References
[1] https://arxiv.org/pdf/1409.1556.pdf
[2] https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .base import BaseBackbone, BaseHead
from .utils import init_normalization, init_activation
from abc import *


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(BaseBackbone):
    name = 'vgg'

    def __init__(self,
                 net_type,
                 in_shape,
                 norm_type,
                 dropout_prob):

        super(VGG, self).__init__(in_shape)
        c, h, w = in_shape
        self.in_channel = c
        self.norm_type = norm_type
        self.dropout_prob = dropout_prob

        cfg_segments = self._divide_cfg_into_segments(cfg[net_type])

        self.layer1 = self._make_layer(cfg_segments[0])
        self.layer2 = self._make_layer(cfg_segments[1])
        self.layer3 = self._make_layer(cfg_segments[2])
        self.layer4 = self._make_layer(cfg_segments[3])
        self.layer5 = self._make_layer(cfg_segments[4])
        self.layer1[0].is_input_layer = True

    def _divide_cfg_into_segments(self, cfg):  # cfg.split('M')
        segments = []
        current_segment = []

        for v in cfg:
            current_segment.append(v)
            if v == 'M':  # each segments ends with max pooling
                segments.append(current_segment)
                current_segment = []
        if current_segment:  # last segment
            segments.append(current_segment)

        return segments

    def _make_layer(self, cfg):
        layers = []

        for out_channel in cfg:
            if out_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channel, out_channel, kernel_size=3, padding=1),
                           init_normalization(channels=out_channel, norm_type=self.norm_type),
                           nn.ReLU(),
                           nn.Dropout(p=self.dropout_prob)]
                self.in_channel = out_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x  # vgg11 - cifar10 (4, 4, 512)


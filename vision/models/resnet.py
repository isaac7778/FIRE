'''
Modified ResNet in PyTorch.

Modifications
[1] Group-Normalization
[2] Learnable Spatial Embedding (no global average pooling)

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from abc import *
from .base import BaseBackbone, BaseHead
from .utils import init_normalization, init_activation

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, norm_type, dropout):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = init_normalization(channels=planes, norm_type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm2 = init_normalization(channels=planes, norm_type=norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                init_normalization(channels=self.expansion * planes, norm_type=norm_type)
            )
        self.dropout = nn.Dropout(dropout)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = self.dropout(self.act2(out))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, norm_type):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = init_normalization(channels=planes, norm_type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.norm2 = init_normalization(channels=planes, norm_type=norm_type)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.norm3 = init_normalization(channels=self.expansion * planes, norm_type=norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                init_normalization(channels=self.expansion * planes, norm_type=norm_type)
            )

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()

    def forward(self, x):
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(BaseBackbone):
    name = 'resnet'

    def __init__(self,
                 net_type,
                 in_shape,
                 downsample,
                 norm_type,
                 dropout):

        super(ResNet, self).__init__(in_shape)
        c, h, w = in_shape
        self.in_channel = c

        if downsample:
            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                init_normalization(channels=64, norm_type=norm_type),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(self.in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
                init_normalization(channels=64, norm_type=norm_type),
                nn.ReLU(inplace=True),
            )
        self.stem[0].is_input_layer = True

        self.in_planes = 64
        if net_type == 'resnet18':
            block = BasicBlock
            num_blocks = [2, 2, 2, 2]

        elif net_type == 'resnet34':
            block = BasicBlock
            num_blocks = [3, 4, 6, 3]

        elif net_type == 'resnet50':
            block = Bottleneck
            num_blocks = [3, 4, 6, 3]

        elif net_type == 'resnet101':
            block = Bottleneck
            num_blocks = [3, 4, 23, 3]

        elif net_type == 'resnet152':
            block = Bottleneck
            num_blocks = [3, 8, 36, 3]

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_type=norm_type, dropout=dropout)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_type=norm_type, dropout=dropout)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_type=norm_type, dropout=dropout)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_type=norm_type, dropout=dropout)

    def _make_layer(self, block, planes, num_blocks, stride, norm_type, dropout):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_type, dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


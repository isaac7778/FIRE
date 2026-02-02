import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.utils import _standard_normal
from einops import rearrange

def init_normalization(channels, norm_type="bn", one_d=False):
    assert norm_type in ["bn", "bn_nt", "ln", "ln_nt", "gn", 'none']
    if norm_type == "bn":
        if one_d:
            return nn.BatchNorm1d(channels, affine=True)
        else:
            return nn.BatchNorm2d(channels, affine=True)

    elif norm_type == "bn_nt":
        if one_d:
            return nn.BatchNorm1d(channels, affine=False)
        else:
            return nn.BatchNorm2d(channels, affine=False)

    elif norm_type == "ln":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=True)
        else:
            return nn.GroupNorm(1, channels, affine=True)

    elif norm_type == "ln_nt":
        if one_d:
            return nn.LayerNorm(channels, elementwise_affine=False)
        else:
            return nn.GroupNorm(1, channels, affine=False)

    elif norm_type == 'gn':
        return nn.GroupNorm(4, channels, affine=False)

    elif norm_type == 'none':
        return nn.Identity()


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)


def init_activation(activ_type="relu"):
    if activ_type == 'relu':
        return nn.ReLU(inplace=True)

    elif activ_type == 'crelu':
        return CReLU()

    elif activ_type == 'none':
        return nn.Identity()

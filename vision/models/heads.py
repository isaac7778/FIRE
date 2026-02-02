from abc import *
import torch
import torch.nn as nn
from .base import BaseHead
from einops import rearrange
from .utils import init_normalization, init_activation

class GAPHead(BaseHead):
    name = 'gap'

    def __init__(self,
                 in_shape,
                 output_size,
                 norm_type,
                 activ_type,
                 drop_prob,
                 hidden_dims):

        super().__init__(in_shape, output_size)
        c, h, w = self.in_shape
        self.pool = nn.AvgPool2d((h, w))
        self.norm = init_normalization(channels=c, norm_type=norm_type)
        self.activ = init_activation(activ_type=activ_type)
        in_channel = c

        layers = []
        hidden_dims = [in_channel] + hidden_dims
        for idx in range(len(hidden_dims) - 1):
            h_in = hidden_dims[idx]
            h_out = hidden_dims[idx + 1]
            if activ_type == 'crelu':
                h_out = h_out // 2
            layers.append(nn.Linear(h_in, h_out))
            layers.append(self.activ)
            layers.append(nn.Dropout(p=drop_prob))

        layers.append(nn.Linear(hidden_dims[-1], output_size))
        layers[-1].is_output_layer = True
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        x = self.pool(x)

        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)
        x = self.fc(x)

        return x

    def get_feature(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        x = self.pool(x)

        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)
        x = self.fc[:-1](x)

        return x

class MLPHead(BaseHead):
    name = 'mlp'

    def __init__(self,
                 in_shape,
                 output_size,
                 norm_type,
                 activ_type,
                 drop_prob,
                 hidden_dims):

        super().__init__(in_shape, output_size)
        c, h, w = self.in_shape
        self.norm = init_normalization(channels=c, norm_type=norm_type)
        self.activ = init_activation(activ_type=activ_type)
        in_channel = c * h * w

        layers = []
        hidden_dims = [in_channel] + hidden_dims
        for idx in range(len(hidden_dims) - 1):
            h_in = hidden_dims[idx]
            h_out = hidden_dims[idx + 1]
            if activ_type == 'crelu':
                h_out = h_out // 2
            layers.append(nn.Linear(h_in, h_out))
            layers.append(self.activ)
            layers.append(nn.Dropout(p=drop_prob))

        layers.append(nn.Linear(hidden_dims[-1], output_size))
        layers[-1].is_output_layer = True
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)
        x = self.fc(x)

        return x

    def get_feature(self, x):
        """
        [params] x (torch.Tensor: (n, c, h, w))
        [returns] x (torch.Tensor: (n, d))
        """
        n, c, h, w = x.shape
        x = rearrange(x, 'n c h w -> n (c h w)')
        x = self.norm(x)
        x = self.fc[:-1](x)

        return x
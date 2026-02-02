from .heads import *
from .resnet import ResNet
from .vgg import VGG
from .vit import ViT
from .base import Model
import torch

def get_resnet18_CIFAR10():
    in_shape = (3, 32, 32)
    output_size = 10
    backbone = ResNet(
        net_type='resnet18',
        in_shape=in_shape,
        downsample=False,
        norm_type='bn',
        dropout=0,
    )
    fake_obs = torch.zeros((2, *in_shape))
    out = backbone(fake_obs)
    head = GAPHead(
        out.shape[1:],
        output_size,
        norm_type='none',
        activ_type='relu',
        drop_prob=0.0,
        hidden_dims=[512],
    )
    model = Model(backbone=backbone, head=head)
    return model

def get_TinyViT_CIFAR100():
    in_shape = (3, 32, 32)
    output_size = 100
    backbone = ViT(
        net_type='vit_tiny',
        in_shape=in_shape,
        patch_size=4,
        dim=192,
        heads=3,
        dim_head=64,
        depth=12,
        mlp_dim=768,
        pool='cls',
        dropout=0,
        emb_dropout=0,
    )
    fake_obs = torch.zeros((2, *in_shape))
    out = backbone(fake_obs)
    head = MLPHead(
        out.shape[1:],
        output_size,
        norm_type='none',
        activ_type='relu',
        drop_prob=0.0,
        hidden_dims=[],
    )
    model = Model(backbone=backbone, head=head)
    return model

def get_VGG16_TinyImageNet():
    in_shape = (3, 64, 64)
    output_size = 200
    backbone = VGG(
        net_type='vgg16',
        in_shape=in_shape,
        norm_type='bn',
        dropout_prob=0,
    )
    fake_obs = torch.zeros((2, *in_shape))
    out = backbone(fake_obs)
    head = MLPHead(
        out.shape[1:],
        output_size,
        norm_type='none',
        activ_type='relu',
        drop_prob=0.0,
        hidden_dims=[1024, 1024],
    )
    model = Model(backbone=backbone, head=head)
    return model
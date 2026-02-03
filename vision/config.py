class Config:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, dictionary):
        self.__dict__.update(dictionary)

    def __repr__(self):
        return f"Config({self.__dict__})"

    def print(self):
        print("-"*30)
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("-" * 30)

CONFIG = {
    'regen': {
        'coef': 0.,  # l2 init regularization coefficient, 0.1 ~ 1e-5
    },
    'snp': {
        'coef': 0. # Shrink and perturb, 0.2, 0.4, 0.6, 0.8, 1.0
    },
    'dash': {
        'alpha': 0.,  
        'lambda': 0.,  
    },
    'full_reset': {
        'enable': False,
    },
    'parseval_reg': {
        'enable': False,
        'coef': 1e-3,
    },
    'fire': {
        'enable': False,
        'iter_num': 10,
    },
    'cbp': {
        'enable': False,
        'replacement_rate': 1e-5,
        'maturity_threshold': 1000,
    },
    'redo': {
        'enable': False,
        'period': 1,
        'threshold': 0.01,
    },
    'snr': {
        'enable': False,
        'tau_percentile': 0.98,
        'tau_reset_freq': 16,
    },
    'optimizer': 'adam',  # adam, muon

    # 'lr': 0.01,  # Learning rate
    # 'lr': 1e-4,  # Learning rate
    'lr': 1e-3,  # Default Learning rate

    'clip_grad_norm': 0.5,  # clip grad norm, 0 for disable

    'task': 'TinyImageNet',  # Task name: CIFAR10, CIFAR100, TinyImageNet
    'model': 'VGG16',  # RESNET18, TinyViT, VGG16
    
    'benchmark': 'warm_start',  # warm_start, continual, class_incremental
    'warm_start_subset_ratio': 10,

    'log_every': 1,
    'seed': 0,  # Random seed
    'batch_size': 256,  # Batch size
    'disable_wandb': False,
}

import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config():
    parser = argparse.ArgumentParser(description='Train Config')
    
    # Flatten dictionary for argparse
    def add_arguments(parser, config, prefix=''):
        for key, value in config.items():
            arg_name = f"--{prefix}{key}".replace('_', '-')
            if isinstance(value, dict):
                add_arguments(parser, value, prefix=f"{prefix}{key}-")
            else:
                arg_type = type(value)
                if arg_type == bool:
                    parser.add_argument(arg_name, type=str2bool, default=value, help=f"Default: {value}")
                elif value is None:
                     parser.add_argument(arg_name, default=value, help=f"Default: {value}")
                else:
                    parser.add_argument(arg_name, type=arg_type, default=value, help=f"Default: {value}")
    
    # Create a deep copy to avoid modifying valid defaults based on previous runs if this function is called multiple times
    from copy import deepcopy
    default_config = deepcopy(CONFIG)
    add_arguments(parser, default_config)
    
    args = parser.parse_args()
    
    # Update config with args
    def update_config(config, args, prefix=''):
        for key, value in config.items():
            arg_name = f"{prefix}{key}" # args attribute name uses underscores, not hyphens
            if isinstance(value, dict):
                update_config(value, args, prefix=f"{prefix}{key}_")
            else:
                if hasattr(args, arg_name):
                    config[key] = getattr(args, arg_name)
    
    update_config(default_config, args)
    
    return Config(default_config)
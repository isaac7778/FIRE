from abc import *
import torch.nn as nn
import torch


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = in_shape

    @classmethod
    def get_name(cls):
        return cls.name

    @abstractmethod
    def forward(self, x):
        """
        [param] x (torch.Tensor): (n, c, h, w)
        [return] x (torch.Tensor): (n, c, h, w)
        """
        pass

    @property
    def output_dim(self):
        pass

class BaseHead(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_shape, output_size):
        super().__init__()
        self.in_shape = in_shape
        self.output_size = output_size

    @classmethod
    def get_name(cls):
        return cls.name

class Model(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

        self.norm_param_names = self.get_norm_param_names(self.backbone, 'backbone') + self.get_norm_param_names(self.head, 'head')

        # hook functions for analysis
        self.hook_enabled = False
        self._activations = {}
        self._register_hooks()

        self.init_stds = {}
        self.init_norms = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                self.init_stds[name] = param.data.std().item()
                self.init_norms[name] = param.data.norm().item()

    def encode(self, x):
        b = self.backbone(x)

        return b

    def forward(self, x):
        """
        [backbone]: (n,c,h,w)-> Tuple((n,c,h,w), info)
        [head]: (n,c,h,w)-> Tuple((n,d), info)
        """
        b = self.backbone(x)
        h = self.head(b)

        return h

    def get_feature(self, x):
        '''

        '''
        b = self.backbone(x)
        feature = self.head.get_feature(b)

        return feature


    def get_activations(self):
        """
        return activations after a forward pass
        """
        return self._activations

    def _register_hooks(self):
        def hook_fn(module, input, output, name):
            if self.hook_enabled:
                self._activations[name] = output

        def register_hook(module, prefix=''):
            for name, sub_module in module.named_children():
                sub_prefix = f'{prefix}_{name}' if prefix else name
                if list(sub_module.children()):  # Check if it is a composite module
                    register_hook(sub_module, sub_prefix)
                else:
                    hook_name = f'{prefix}_{name}' if prefix else name
                    sub_module.register_forward_hook(
                        lambda module, input, output, name=hook_name: hook_fn(module, input, output, name)
                    )

        register_hook(self.backbone, 'backbone')
        register_hook(self.head, 'head')
        self.backbone.register_forward_hook(
            lambda module, input, output, name='backbone_output': hook_fn(module, input, output, name)
        )

    def enable_hooks(self):
        self.hook_enabled = True
        self._activations.clear()

    def disable_hooks(self):
        self.hook_enabled = False
        self._activations.clear()

    def is_norm_layer_parameter(self, param_name:str):
        if param_name in self.norm_param_names:
            return True
        return False

    def get_norm_param_names(self, model, fname):
        norm_param_names = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                for param_name, _ in module.named_parameters():
                    full_name = f"{fname}.{name}.{param_name}" if name else param_name
                    norm_param_names.append(full_name)
        return norm_param_names

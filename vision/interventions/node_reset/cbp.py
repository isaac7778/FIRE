from typing import Dict
import math
import torch
from torch import nn
from torch.utils.data import DataLoader

from interventions.node_reset.reset_base import ResetBase


def get_layer_bound(layer, init, gain):
    if isinstance(layer, nn.Conv2d):
        return math.sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, nn.Linear):
        if init == 'default':
            bound = math.sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * math.sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = math.sqrt(3 / layer.in_features)
        else:
            bound = gain * math.sqrt(3 / layer.in_features)
        return bound


def get_layer_std(layer, gain):
    if isinstance(layer, nn.Conv2d):
        return gain * math.sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, nn.Linear):
        return gain * math.sqrt(1 / layer.in_features)


class CBP(ResetBase):
    """
    Paper: https://www.nature.com/articles/s41586-024-07711-7
    Code: https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/algos/res_gnt.py
          https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/algos/cbp_conv.py
    """

    def __init__(
        self,
        model: nn.Module,
        period: int,
        replacement_rate: float,
        maturity_threshold: int,
        decay_rate: float = 0.99,
        n_analysis_samples: int = 512,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(model, period)
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.decay_rate = decay_rate
        self.n_analysis_samples = n_analysis_samples
        self.device = device

        # Target layers
        self.in_layers: Dict[str, nn.Module] = {}
        self.out_layers: Dict[str, nn.Module] = {}
        self.bn_layers: Dict[str, nn.Module] = {}
        self.pool_layers: Dict[str, nn.Module] = {}
        self.get_weight_layers()

        # State variables
        self.util: Dict[str, torch.Tensor] = {}
        self.ages: Dict[str, torch.Tensor] = {}
        self.mean_feature_mag: Dict[str, torch.Tensor] = {}
        self.accumulated_num_features_to_replace: Dict[str, int] = {}
        self.stds: Dict[str, float] = {}
        self.num_new_features_to_replace: Dict[str, int] = {}
        for name, layer in self.in_layers.items():
            next_layer = self.out_layers[name] if name in self.out_layers else None
            if isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Conv2d):
                out_dim = layer.out_channels
            elif isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
                out_dim = next_layer.in_features
            elif isinstance(layer, nn.Linear):
                out_dim = layer.out_features
            else:
                raise RuntimeError
            self.util[name] = torch.zeros(out_dim, dtype=torch.float32, device=self.device)
            self.ages[name] = torch.zeros(out_dim, dtype=torch.float32, device=self.device)
            self.mean_feature_mag[name] = torch.zeros(out_dim, dtype=torch.float32, device=self.device)
            self.accumulated_num_features_to_replace[name] = 0
            self.stds[name] = get_layer_std(layer, gain=nn.init.calculate_gain('relu'))
            self.num_new_features_to_replace[name] = replacement_rate * out_dim

        # Register forward hook to capture features
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks = []
        for name, layer in self.in_layers.items():
            self._hooks.append(layer.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook_fn(_, __, out):
            previous = self._features.get(name, torch.empty(0, device=out.device))
            if name in self.pool_layers:
                pool_layer = self.pool_layers[name]
                current = pool_layer(out).detach()
            else:
                current = out.detach()
            feature = torch.cat([previous, current], dim=0)
            self._features[name] = feature[-self.n_analysis_samples:].detach()
        return hook_fn

    @torch.no_grad()
    def _apply_fn(self, dataloader: DataLoader):
        features_to_replace_input_indices, features_to_replace_output_indices = self.get_features_to_reinit(features=self._features)
        self.reinit_features(features_to_replace_input_indices, features_to_replace_output_indices)

    def get_weight_layers(self):
        prev_name = None
        _layers = {}
        for name, m in self.model.named_modules():
            is_attn = any(t in name for t in ["to_q", "to_k", "to_v"])
            if 'shortcut' in name:  # Skip shortcut layers
                continue
            elif isinstance(m, (nn.Linear, nn.Conv2d)):
                self.in_layers[name] = m
                if not is_attn:  # No output layer for qkv in transformers
                    if prev_name is not None:
                        self.out_layers[prev_name] = m
                    prev_name = name
                else: prev_name = None
            elif isinstance(m, nn.BatchNorm2d):
                self.bn_layers[prev_name] = m
            elif isinstance(m, nn.MaxPool2d):
                self.pool_layers[prev_name] = m

        # Remove the last layer from in_layers as it has no outgoing weights
        self.in_layers.pop(prev_name)

    def get_features_to_reinit(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        features_to_replace_input_indices = {name: torch.empty(0, dtype=torch.long) for name in self.in_layers.keys()}
        features_to_replace_output_indices = {name: torch.empty(0, dtype=torch.long) for name in self.out_layers.keys()}
        if self.replacement_rate == 0:
            return features_to_replace_input_indices, features_to_replace_output_indices

        for name in self.in_layers.keys():
            in_layer = self.in_layers[name]
            out_layer = self.out_layers[name] if name in self.out_layers else None

            self.ages[name] += 1
            """
            Update feature stats
            """
            if isinstance(in_layer, nn.Linear):
                if features[name].ndim == 3:
                    reshape_ft = features[name].reshape(-1, features[name].shape[-1])
                    self.mean_feature_mag[name] += (1 - self.decay_rate) * reshape_ft.abs().mean(dim=0)
                elif features[name].ndim == 2:
                    self.mean_feature_mag[name] += (1 - self.decay_rate) * features[name].abs().mean(dim=0)
                else:
                    raise NotImplementedError
            elif isinstance(in_layer, nn.Conv2d) and isinstance(out_layer, nn.Conv2d):
                self.mean_feature_mag[name] += (1 - self.decay_rate) * features[name].abs().mean(dim=(0, 2, 3))
            elif isinstance(in_layer, nn.Conv2d) and isinstance(out_layer, nn.Linear):
                flattened_features = features[name].reshape((-1, out_layer.weight.size(1)))
                self.mean_feature_mag[name] += (1 - self.decay_rate) * flattened_features.abs().mean(dim=0)
            """
            Find the no. of features to replace
            """
            eligible_feature_indices = torch.where(self.ages[name] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            self.accumulated_num_features_to_replace[name] += self.num_new_features_to_replace[name]

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace[name])
            self.accumulated_num_features_to_replace[name] -= num_new_features_to_replace

            if num_new_features_to_replace == 0: continue

            """
            Calculate utility
            """
            if out_layer is not None:
                if isinstance(out_layer, nn.Linear):
                    output_weight_mag = out_layer.weight.data.abs().mean(dim=0)
                elif isinstance(out_layer, nn.Conv2d):
                    output_weight_mag = out_layer.weight.data.abs().mean(dim=(0, 2, 3))
                self.util[name] = output_weight_mag * self.mean_feature_mag[name]
            else:
                self.util[name] = self.mean_feature_mag[name] # use only feature for attn layers since output weight is ambiguous

            """
            Find features to replace in the current layer
            """
            new_features_to_replace = torch.topk(-self.util[name][eligible_feature_indices], num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            """
            Initialize utility for new features
            """
            self.util[name][new_features_to_replace] = 0

            features_to_replace_input_indices[name] = new_features_to_replace
            features_to_replace_output_indices[name] = new_features_to_replace

            if isinstance(in_layer, torch.nn.Conv2d) and isinstance(out_layer, torch.nn.Linear):
                spatial_size = out_layer.in_features // in_layer.out_channels
                conv_channel_indices = new_features_to_replace // spatial_size
                conv_channel_indices = torch.unique(conv_channel_indices)
                features_to_replace_input_indices[name] = conv_channel_indices
        return features_to_replace_input_indices, features_to_replace_output_indices

    def reinit_features(self, features_to_replace_input_indices, features_to_replace_output_indices):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        for name in self.in_layers.keys():
            num_features_to_replace = features_to_replace_input_indices[name].shape[0]
            if num_features_to_replace == 0:
                continue
            in_layer = self.in_layers[name]
            out_layer = self.out_layers[name] if name in self.out_layers else None

            in_layer.weight.data[features_to_replace_input_indices[name], :] *= 0.0
            in_layer.weight.data[features_to_replace_input_indices[name], :] += \
                torch.empty([num_features_to_replace] + list(in_layer.weight.shape[1:]), device=self.device).normal_(std=self.stds[name])

            if in_layer.bias is not None:
                in_layer.bias.data[features_to_replace_input_indices[name]] *= 0.0
            """
            Set the outgoing weights and ages to zero
            """
            if out_layer is not None:
                out_layer.weight.data[:, features_to_replace_output_indices[name]] = 0
            self.ages[name][features_to_replace_input_indices[name]] = 0
            """
            Reset the corresponding batchnorm layers
            """
            if name in self.bn_layers:
                self.bn_layers[name].bias[features_to_replace_input_indices[name]] *= 0.0
                self.bn_layers[name].weight[features_to_replace_input_indices[name]] *= 0.0
                self.bn_layers[name].weight[features_to_replace_input_indices[name]] += 1.0
                self.bn_layers[name].running_mean[features_to_replace_input_indices[name]] *= 0.0
                self.bn_layers[name].running_var[features_to_replace_input_indices[name]] *= 0.0
                self.bn_layers[name].running_var[features_to_replace_input_indices[name]] += 1.0
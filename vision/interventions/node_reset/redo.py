from typing import Dict
from copy import deepcopy
import torch
from torch import nn
from torch.utils.data import DataLoader

from interventions.node_reset.reset_base import ResetBase


@torch.no_grad()
def get_redo_masks(activations: dict, tau: float) -> dict:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """
    masks = {}

    # Last activation are the q-values, which are never reset
    for name, activation in list(activations.items())[:-1]:
        # Taking the mean here conforms to the expectation under D in the main paper's formula
        if activation.ndim == 4:  # Conv layer
            score = activation.abs().mean(dim=(0, 2, 3))
        elif activation.ndim == 3:  # Linear layer in transformers
            score = activation.abs().mean(dim=(0, 1))
        else:  # Linear layer
            score = activation.abs().mean(dim=0)

        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
        if tau > 0.0:
            layer_mask[normalized_score <= tau] = 1
        else:
            layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
        masks[name] = layer_mask

    return masks


@torch.no_grad()
def reset_dormant_neurons(model: nn.Module, masks: dict):
    """Re-initializes the dormant neurons of a model."""

    layers = {name: module for name, module in model.named_modules()}

    _layers = []
    _masks = []
    for name, mask in masks.items():
        if name in layers.keys():
            _layers.append(layers[name])
            _masks.append(mask)

    masks = _masks[:-1]
    ingoing_layers = _layers[:-1]
    outgoing_layers = _layers[1:]

    # Sanity checks
    assert (
        len(ingoing_layers) == len(outgoing_layers) == len(masks)
    ), "The number of layers and masks should match the number of masks."

    # Reset the ingoing weights
    reset_indices = {}
    for layer, mask in zip(ingoing_layers, masks):
        if torch.all(~mask):
            continue
        elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            # Reinitialize the ingoing weights
            new_layer = deepcopy(layer)
            new_layer.reset_parameters()
            layer.weight.data[mask, ...] = new_layer.weight.data[mask, ...]
            reset_indices[layer.weight] = (mask, ...)
            # Reset the bias if exists
            if layer.bias is not None:
                layer.bias.data[mask, ...] = new_layer.bias.data[mask, ...]
                reset_indices[layer.bias] = (mask, ...)

    # Set the outgoing weights to 0
    for layer, next_layer, mask in zip(ingoing_layers, outgoing_layers, masks):
        if torch.all(~mask):
            continue
        elif isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
            # Reset the outgoing weights to 0
            num_repeatition = next_layer.weight.data.shape[1] // mask.shape[0]
            if num_repeatition >= 1:
                linear_mask = torch.repeat_interleave(mask, num_repeatition)
            else:
                linear_mask = mask.reshape(next_layer.weight.data.shape[1], -1).any(dim=1)
            next_layer.weight.data[:, linear_mask].data.fill_(0)
            reset_indices[next_layer.weight] = (slice(None), linear_mask)
        elif ((isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Conv2d)) or
              (isinstance(layer, nn.Linear) and isinstance(next_layer, nn.Linear))):
            # Reset the outgoing weights to 0
            if next_layer.weight.data.shape[1] == mask.shape[0]:
                next_layer.weight.data[:, mask, ...].data.fill_(0)
                reset_indices[next_layer.weight] = (slice(None), mask, ...)
            elif mask.shape[0] % next_layer.weight.data.shape[1] == 0:
                contracted_mask = mask.reshape(next_layer.weight.data.shape[1], -1).any(dim=1)
                next_layer.weight.data[:, contracted_mask, ...].data.fill_(0)
                reset_indices[next_layer.weight] = (slice(None), contracted_mask, ...)
        else:
            continue

    return reset_indices


@torch.no_grad()
def reset_opt_state(optimizer: torch.optim.Optimizer, reset_indices: dict):
    """Reset optimizer state with a given set of indices for each parameter."""

    for param, indices in reset_indices.items():
        state = optimizer.state.get(param, {})
        for key in state.keys():
            if key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                buf = state.get(key, None)
                if buf is None:
                    continue
                buf[indices].zero_()
            elif key == 'step':
                state['step'].zero_()

    return optimizer


class ReDO(ResetBase):
    """
    Paper: https://arxiv.org/abs/2302.12902
    Code: https://github.com/timoklein/redo/blob/main/src/redo.py
    """

    def __init__(
        self,
        model: nn.Module,
        period: int,
        threshold: float,
        n_analysis_samples: int = 512,
    ):
        super().__init__(model, period)
        self.threshold = threshold
        self.n_analysis_samples = n_analysis_samples

        # Target layers
        self._layers: Dict[str, nn.Module] = {}
        self.get_weight_layers()

        # Register forward hook to capture activations
        self._acts: Dict[str, torch.Tensor] = {}
        self._hooks = []
        for name, layer in self._layers.items():
            self._hooks.append(layer.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook_fn(_, __, out):
            previous = self._acts.get(name, torch.empty(0, device=out.device))
            is_attn = any(t in name for t in ["to_q", "to_k", "to_v"])
            if not is_attn:
                current = nn.functional.relu(out).detach()  # Consider only ReLU activations
            else:
                current = out.detach()
            act = torch.cat([previous, current], dim=0)
            self._acts[name] = act[-self.n_analysis_samples:].detach()
        return hook_fn

    def apply(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        info = {}
        if self._step % self.period == 0:
            # dataloader = deepcopy(dataloader)
            info = self._apply_fn(dataloader, optimizer)
        self._step += 1
        return info

    def _apply_fn(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        masks = get_redo_masks(self._acts, self.threshold)
        reset_indices = reset_dormant_neurons(self.model, masks)
        reset_opt_state(optimizer, reset_indices)

        # Log dormant ratios
        info = {}
        total_params = 0
        total_dormant_params = 0
        for k, v in masks.items():
            num_params = v.numel()
            num_dormant = v.sum().item()
            info[f'ReDO_dormant_ratios/{k}'] = num_dormant / num_params if num_params > 0 else 0.0
            total_params += num_params
            total_dormant_params += num_dormant
        info['ReDO_dormant_ratios/overall'] = total_dormant_params / total_params if total_params > 0 else 0.0
        return info

    def get_weight_layers(self):
        for name, m in self.model.named_modules():
            if 'shortcut' in name:  # Skip shortcut layers
                continue
            elif isinstance(m, (nn.Linear, nn.Conv2d)):
                self._layers[name] = m
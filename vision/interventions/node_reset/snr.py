from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from interventions.node_reset.reset_base import ResetBase
from interventions.node_reset.redo import reset_dormant_neurons, reset_opt_state


class SNR(ResetBase):
    """
    Paper: https://arxiv.org/abs/2410.20098
    Code: https://github.com/ajozefiak/SelfNormalizedResets/blob/main/src/algorithms/snr.py
    """

    def __init__(
        self,
        model: nn.Module,
        period: int,
        threshold_reset_freq: int,
        threshold_percentile: float = 0.9,
        threshold_expansion_factor: float = 2.0,
        tau_max: int = 20000,
        n_analysis_samples: int = 512,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(model, period)
        self.threshold_reset_freq = threshold_reset_freq
        self.threshold_percentile = threshold_percentile
        self.threshold_expansion_factor = threshold_expansion_factor
        self.tau_max = tau_max
        self.n_analysis_samples = n_analysis_samples
        self.device = device

        # State variables
        self.ages: Dict[str, torch.Tensor] = {}
        self.hist: Dict[str, torch.Tensor] = {}
        self.tau: Dict[str, torch.Tensor] = {}

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
            current = nn.functional.relu(out).detach()  # Consider only ReLU activations
            act = torch.cat([previous, current], dim=0)
            self._acts[name] = act[-self.n_analysis_samples:].detach()
        return hook_fn

    def apply(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        if self._step % self.period == 0:
            # dataloader = deepcopy(dataloader)
            self._apply_fn(dataloader, optimizer)
        self._step += 1

        # Log tau values
        info = {}
        if self._step % self.threshold_reset_freq == 0:
            for k, v in self.tau.items():
                info[f'SNR_thresholds/{k}'] = v
        return info

    @torch.no_grad()
    def _apply_fn(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        # Initialize variables
        layer_names = list(self._layers.keys())

        # Update ages and histograms
        for name in layer_names:
            # Get activations
            activation = self._acts[name]
            batch_size = activation.shape[0]
            if activation.ndim == 4:  # Conv layer: (B, C, H, W) -> (B*H*W, C)
                act = activation.permute(0, 2, 3, 1).reshape(-1, activation.shape[1])
            elif activation.ndim == 3:  # Linear layer: (B, T, N) -> (B*T, N)
                act = activation.reshape(-1, activation.shape[-1])
            else:  # Linear layer: (B, N)
                act = activation
            num_neurons = act.shape[1]

            # Ensure state variables
            if name not in self.ages:
                self.ages[name] = torch.zeros(num_neurons, dtype=torch.long, device=self.device)
            if name not in self.hist:
                self.hist[name] = torch.zeros((num_neurons, self.tau_max+1), dtype=torch.long, device=self.device)
            if name not in self.tau:
                self.tau[name] = torch.full((num_neurons,), max(1, self.tau_max//2), dtype=torch.long, device=self.device)

            # Calculate firing mask
            fire = (act > 0).any(dim=0)  # (N,)

            # Update ages: firing -> 0, non-firing -> +B
            ages = self.ages[name]
            col_mask = torch.where(fire, ages.clamp_max(self.tau_max), 0).unsqueeze(1)  # (N,1)
            row_mask = fire.unsqueeze(1).long()  # (N,1)
            indices = torch.arange(self.tau_max+1, device=self.device).unsqueeze(0)  # (1, tau_max+1)
            self.hist[name] += torch.where(indices == col_mask, row_mask, 0)
            self.ages[name] = torch.where(fire, 0, ages + batch_size)

        # Update tau based on histograms
        if self._step % self.threshold_reset_freq == 0:
            for name in layer_names:
                activation = self._acts[name]
                batch_size = activation.shape[0]
                hist = self.hist[name]  # (N, tau_max+1)
                threshold = self.tau[name]  # (N,)

                # Turn each neuron's histogram into a CDF and compute its threshold_percentile index
                cdf = torch.cumsum(hist[:, 1:], dim=1)
                cdf_normalized = cdf / (torch.sum(hist[:, 1:], dim=1, keepdim=True) + 1e-8)
                threshold_percentile_index = torch.argmax((cdf_normalized >= self.threshold_percentile).long(), dim=1) + 1

                # If the threshold_percentile index is below tau, contract tau to the threshold_percentile index
                # other expand tau by threshold_expansion_factor
                new_threshold_layer = torch.where(
                    threshold_percentile_index < threshold,
                    threshold_percentile_index,
                    self.threshold_expansion_factor*threshold,
                )

                # Ensure that we do not exceed the tau_max nor hit 0
                new_threshold = torch.maximum(
                    torch.full_like(new_threshold_layer, batch_size),
                    torch.minimum(new_threshold_layer, torch.full_like(new_threshold_layer, self.tau_max)),
                )
                self.tau[name].data.copy_(new_threshold)

                # Reset neuron_ages_hist
                self.hist[name].zero_()

        # Get masks for all layers
        masks = {}
        for name in layer_names:
            masks[name] = (self.ages[name] >= self.tau[name])

        # Update histograms
        for name in layer_names:
            ages = self.ages[name]
            col_mask = torch.where(masks[name], ages.clamp_max(self.tau_max), 0).unsqueeze(1)  # (N, 1)
            row_mask = masks[name].unsqueeze(1).long()  # (N, 1)
            indices = torch.arange(self.tau_max+1, device=self.device).unsqueeze(0)  # (1, tau_max+1)
            self.hist[name] += torch.where(indices == col_mask, row_mask, 0)

        # Set zero ages
        for name in layer_names:
            self.ages[name] *= (1 - masks[name].long())

        reset_indices = reset_dormant_neurons(self.model, masks)
        reset_opt_state(optimizer, reset_indices)

    def get_weight_layers(self):
        for name, m in self.model.named_modules():
            if 'shortcut' in name:  # Skip shortcut layers
                continue
            elif isinstance(m, (nn.Linear, nn.Conv2d)):
                self._layers[name] = m
from abc import ABC, abstractmethod
from copy import deepcopy
import torch.nn as nn
from torch.utils.data import DataLoader


class ResetBase(ABC):
    def __init__(self, model: nn.Module, period: int):
        self._step = 0
        self.model = model
        self.period = period

    def update_period(self, period: int):
        # self._step = 0
        self._step = 1
        self.period = period

    def apply(self, dataloader: DataLoader):
        if self._step % self.period == 0:
            dataloader = deepcopy(dataloader)
            self._apply_fn(dataloader)
        self._step += 1
        return {}

    @abstractmethod
    def _apply_fn(self, dataloader: DataLoader):
        pass
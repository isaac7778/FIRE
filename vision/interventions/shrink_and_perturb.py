import torch
from torch import nn

def shrink_and_perturb(model: nn.Module, init_model: nn.Module, shrink_coef: float):
    with torch.no_grad():
        for param, init_param in zip(model.parameters(), init_model.parameters()):
            param.data = (1 - shrink_coef) * param.data.clone() + shrink_coef * init_param.data.clone()
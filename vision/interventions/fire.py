import torch
from torch import nn
import numpy as np

@torch.no_grad()
def fire(model, alpha, iteration, is_vit=False):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            param = m.weight

            if is_vit:
                m_type = name.split(".")[-1]
                if m_type not in ["to_q", "to_k"]: # only perform on mlp for q and k
                    continue

            weight_matrix = param.data.detach().clone()
            if weight_matrix.ndim == 4:
                ortho_weight_matrix = torch.zeros_like(weight_matrix)
                for i in range(weight_matrix.shape[2]):
                    for j in range(weight_matrix.shape[3]):
                        ortho_weight_matrix[:,:,i,j] = newton_schulz(weight_matrix[:,:,i,j], num_iters=iteration)
            else:
                ortho_weight_matrix = newton_schulz(weight_matrix, num_iters=iteration)

            # scale = sqrt(d_out/d_in) / kernel_size
            kernel_size = weight_matrix.shape[2]*weight_matrix.shape[3] if weight_matrix.ndim==4 else 1.0
            scale = np.sqrt(weight_matrix.shape[0]/weight_matrix.shape[1]) / kernel_size
            ortho_weight_matrix *= scale
            final_matrix = alpha * ortho_weight_matrix + (1 - alpha) * weight_matrix
            param.data = final_matrix

def newton_schulz(matrix, num_iters=10):
    a, b = (1.5, -0.5)
    assert matrix.ndim == 2
    do_transpose = matrix.size(1) > matrix.size(0)

    X = matrix
    if do_transpose:
        X = X.T

    X = X / X.norm()
    for _ in range(num_iters):
        A = X.T @ X
        X = a * X + b * X @ A

    if do_transpose:
        X = X.T
    return X

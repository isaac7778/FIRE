import torch
from torch import nn
import numpy as np

@torch.no_grad()
def fire(model, iteration):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m_type = name.split(".")[-1]

            if m_type == "c_attn":
                # apply qkv separately
                param = m.weight

                _, n_embed = param.shape

                q_p = param.data[:n_embed]
                k_p = param.data[n_embed:n_embed*2]
                v_p = param.data[-n_embed:]
                out = []
                for p in [q_p, k_p, v_p]:
                    weight_matrix = p.data.detach().clone()
                    ortho_weight_matrix = newton_schulz(weight_matrix, num_iters=iteration)

                    # scale = sqrt(d_out/d_in) / kernel_size
                    kernel_size = weight_matrix.shape[2]*weight_matrix.shape[3] if weight_matrix.ndim==4 else 1.0
                    scale = np.sqrt(weight_matrix.shape[0]/weight_matrix.shape[1]) / kernel_size
                    ortho_weight_matrix *= scale
                    out.append(ortho_weight_matrix)

                param.data[:n_embed], param.data[n_embed:n_embed * 2], param.data[-n_embed:] = out
            else:
                param = m.weight
                weight_matrix = param.data.detach().clone()
                if weight_matrix.ndim == 4:
                    ortho_weight_matrix = torch.zeros_like(weight_matrix)
                    for i in range(weight_matrix.shape[2]):
                        for j in range(weight_matrix.shape[3]):
                            ortho_weight_matrix[:, :, i, j] = newton_schulz(weight_matrix[:, :, i, j])
                else:
                    ortho_weight_matrix = newton_schulz(weight_matrix, num_iters=iteration)

                # scale = sqrt(d_out/d_in) / kernel_size
                kernel_size = weight_matrix.shape[2] * weight_matrix.shape[3] if weight_matrix.ndim == 4 else 1.0
                scale = np.sqrt(weight_matrix.shape[0] / weight_matrix.shape[1]) / kernel_size
                ortho_weight_matrix *= scale
                param.data = ortho_weight_matrix

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
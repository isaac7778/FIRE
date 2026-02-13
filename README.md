
# 🔥 FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff (ICLR'26 Oral)

This repository contains the code for the paper:  
**"FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff"**  
Authors: 
[Isaac Han](https://isaac7778.github.io/), 
Sangyeon Park, 
Seungwon Oh, 
[Donghu Kim](https://i-am-proto.github.io/),
[Hojoon Lee*](https://joonleesky.github.io/), 
[Kyung-Joong Kim*](https://cilab.gist.ac.kr/hp/current-member/)

Accepted at ICLR 2026 (Oral presentation - top ~1.5% of submissions)

<p align="left">
  <img src="assets/iclr2026fire.png" width="300">
</p>

For more information, please see our [project webpage](https://isaac7778.github.io/fire/) and [paper](https://arxiv.org/abs/2602.08040)


## 📖 Codebase

As we conducted experiments in diverse domains (vision, language and RL), we used different settings for each of them. Please refer below to set up and run experiments:

#### Continual Visual Learning (Fig 2) > [vision/README.md](vision/README.md)

#### Continual Pretraining of LLMs (Fig 3) > [language/README.md](language/README.md)

#### Reinforcement Learning (Fig 4) > [rl/dqn/README.md](rl/dqn/README.md) and [rl/sac/README.md](rl/sac/README.md)

## 🔥FIRE implementation
Stop worrying about plasticity loss, just apply FIRE before training on new data.
```python
import torch
from torch import nn
import numpy as np

@torch.no_grad()
def fire(model, iteration=10):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            param = m.weight
            weight_matrix = param.data.detach().clone()
            if weight_matrix.ndim == 4: # cnn
                ortho_weight_matrix = torch.zeros_like(weight_matrix)
                for i in range(weight_matrix.shape[2]):
                    for j in range(weight_matrix.shape[3]):
                        ortho_weight_matrix[:,:,i,j] = newton_schulz(weight_matrix[:,:,i,j], num_iters=iteration)
            else: # linear
                ortho_weight_matrix = newton_schulz(weight_matrix, num_iters=iteration)

            # scale = sqrt(d_out/d_in) / kernel_size
            kernel_size = weight_matrix.shape[2]*weight_matrix.shape[3] if weight_matrix.ndim==4 else 1.0
            scale = np.sqrt(weight_matrix.shape[0]/weight_matrix.shape[1]) / kernel_size
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
```
---

## 📄 Citation
If you find our work useful, please consider citing the paper as follows:
```
@article{han2026fire,
	title={FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff},
	author={Isaac Han and Sangyeon Park and Seungwon Oh and Donghu Kim and Hojoon Lee and Kyung-Joong Kim},
	journal={International Conference on Learning Representations (ICLR)},
	year={2026}
}	
```

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from scale_rl.networks.utils import he_normal_init, orthogonal_init


class MLPBlock(nn.Module):
    hidden_dim: int
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # sqrt(2) is recommended when using with ReLU activation.
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal_init(jnp.sqrt(2)),
            dtype=self.dtype,
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal_init(jnp.sqrt(2)),
            dtype=self.dtype,
        )(x)
        x = nn.relu(x)
        return x


class ResidualBlock(nn.Module):
    hidden_dim: int
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Dense(
            self.hidden_dim * 4, kernel_init=he_normal_init(), dtype=self.dtype
        )(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=he_normal_init(), dtype=self.dtype)(x)
        return res + x


class SimNorm(nn.Module):
    levels: int
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        L = self.levels
        shape = x.shape
        x = jnp.reshape(x, (*shape[:-1], L, -1))
        x = nn.softmax(x, -1)
        x = jnp.reshape(x, (shape))
        return x

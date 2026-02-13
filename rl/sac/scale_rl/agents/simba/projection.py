import jax
import jax.numpy as jnp
import numpy as np
from scale_rl.networks.utils import tree_map_until_match
from scale_rl.networks.trainer import PRNGKey, Trainer

# NOTE: not using jnp.asarray((3.4445, -4.7750, 2.0315)) & 5 steps
#       turns out using this does not results in norm=1 columns (or rows)
def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array = jnp.asarray((1.5,-0.5,0.0)), 
    ns_steps: int = 10,
    eps: float = 1e-8,
) -> jax.Array:
    r"""Orthogonalize via Newton-Schulz iteration.

    We opt to use a quintic iteration whose coefficients are selected to maximize
    the slope at zero. For the purpose of minimizing steps, it turns out to be
    empirically effective to keep increasing the slope at zero even beyond the
    point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather
    something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
    which turns out not to hurt model performance at all relative to UV^T, where
    USV^T = G is the SVD.

    Args:
    x: A matrix to orthogonalize.
    ns_coeffs: Coefficients for the Newton-schulz iterators.
        Must have shape (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
        Ignored if `ns_coeffs` is a 2D array.
    eps: Term added to denominators to improve numerical stability.

    Returns:
    The orthogonalized matrix.
    """
    if x.ndim != 2:
        raise ValueError(f'Input must have shape (m, n), got {x.shape}')
    if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
        raise ValueError(
            'Newton-Schulz coefficients must have shape (3,) or (n, 3), '
            f'got {ns_coeffs.shape}'
    )
    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
        a = x @ x.T
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x + b @ x
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    x /= jnp.linalg.norm(x) + eps  # Ensure spectral norm is at most 1
    ns_coeffs = ns_coeffs.astype(x.dtype)
    if ns_coeffs.ndim == 1:
        x = jax.lax.fori_loop(
            0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x
        )
    else:
        x, _ = jax.lax.scan(
            lambda x, abc: (newton_schulz_iterator(x, abc), None), x, ns_coeffs
        )
    if transposed:
        x = x.T
    return x


def orthogonal_project_layer(tree, ns_steps=10, scaler_type='default'):
    """
    apply l2-normalization to the all leaf nodes
    """

    def ortho_fn(x):
        assert x.ndim == 2
        d_in, d_out = x.shape[0], x.shape[1]
        if scaler_type == 'muon':
            scale = jnp.sqrt(d_out/d_in)
        elif scaler_type == 'ortho_init':
            scale = 1.0
        else:
            raise ValueError
        return scale * orthogonalize_via_newton_schulz(
            x,
            ns_steps = ns_steps,
        )

    if tree["kernel"].ndim == 2:
        tree["kernel"] = ortho_fn(tree["kernel"])
    elif tree["kernel"].ndim == 3:
        tree["kernel"] = jax.vmap(ortho_fn)(tree["kernel"])
    else:
        raise ValueError

    return tree


def orthogonal_project_network(
    network: Trainer,
    ns_steps: int,
    scaler_type: str,
    regex: str = "Dense.*",
) -> Trainer:
    params = network.params
    new_params = tree_map_until_match(
        f=lambda x: orthogonal_project_layer(x, ns_steps, scaler_type),
        tree=params,
        target_re=regex,
        keep_values=True,
    )
    return network.replace(params=new_params)

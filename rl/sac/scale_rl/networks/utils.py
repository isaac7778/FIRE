import re
from typing import List

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))


###########################
###   Initialization    ###
###########################


def orthogonal_init(scale: float = jnp.sqrt(2), axis: int = -1):
    return nn.initializers.orthogonal(scale, column_axis=axis)


def xavier_normal_init():
    return nn.initializers.glorot_normal()


def xavier_uniform_init():
    return nn.initializers.glorot_uniform()


def he_normal_init():
    return nn.initializers.he_normal()


def he_uniform_init():
    return nn.initializers.he_uniform()


###########################


def noisy_sample(dist: tfd.Distribution, action_noise: List[jnp.ndarray]):
    """
    reference: https://github.com/martius-lab/pink-noise-rl/blob/main/pink/sb3.py
    """
    if isinstance(dist, tfp.distributions.TransformedDistribution):
        dist = dist.distribution
    mean = dist.loc
    scale_diag = dist.stddev()
    actions = mean + scale_diag * jnp.stack(action_noise)
    return nn.tanh(actions)


def tree_map_until_match(
    f, tree, target_re, *rest, keep_structure=True, keep_values=False
):
    """
    Similar to `jax.tree_util.tree_map_with_path`, but `is_leaf` is a regex condition.
    args:
        f: A function to map the discovered nodes (i.e., dict key matches `target_re`).
           Inputs to f will be (1) the discovered node and (2) the corresponding nodes in `*rest``.
        target_re: A regex string condition that triggers `f`.
        tree: A pytree to be searched by `target_re` and mapped by `f`.
        *rest: List of pytrees that are at least 'almost' identical structure to `tree`.
               'Almost', since the substructure of matching nodes don't have to be identical.
               i.e., The tree structure of `tree` and `*rest` should be identical only up to the matching nodes.
        keep_structure: If false, the returned tree will only contain subtrees that lead to the matching nodes.
        keep_values: If false, unmatched leaves will become `None`. Assumes `keep_structure=True`.
    """

    if not isinstance(tree, dict):
        return tree if keep_values else None

    ret_tree = {}
    for k, v in tree.items():
        v_rest = [r[k] for r in rest]
        if re.fullmatch(target_re, k):
            ret_tree[k] = f(v, *v_rest)
        else:
            subtree = tree_map_until_match(
                f,
                v,
                target_re,
                *v_rest,
                keep_structure=keep_structure,
                keep_values=keep_values,
            )
            if keep_structure or subtree:
                ret_tree[k] = subtree

    return ret_tree


def tree_filter(f, tree, target_re="scaler"):
    if isinstance(tree, dict):
        # Keep only "target_re" keys in the dictionary
        filtered_tree = {}
        for k, v in tree.items():
            if re.fullmatch(target_re, k):
                filtered_tree[k] = tree_filter(f, v, target_re="scaler")
            elif isinstance(v, dict):  # Recursively check nested dictionaries
                filtered_value = tree_filter(f, v, target_re="scaler")
                if filtered_value:  # Only keep non-empty dictionaries
                    filtered_tree[k] = filtered_value
        return filtered_tree
    else:
        # If not a dictionary, return the tree as is (typically a leaf node)
        return f(tree)

"""
Stochastically rounded operations between JAX tensors.

This code was written by Nestor Demeure and is licensed under the Apache 2.0 license.
You can find an up-to-date source and full description here: https://github.com/nestordemeure/jochastic
"""
import jax
import jax.numpy as jnp


def stochastic_round_bf16(key, x):
    finfo = jnp.finfo(jnp.bfloat16)

    # round x in both directions
    x_closer = x.astype(jnp.bfloat16)
    error = x - x_closer.astype(jnp.float32)
    direction = jnp.where(error > 0, finfo.max, finfo.min)
    x_farther = jnp.nextafter(x_closer, direction)

    # randomly pick rounding direction to preserve orig. value in expectation 
    ulp = jnp.abs(x_farther.astype(jnp.float32) - x_closer.astype(jnp.float32))
    rand_unif = jax.random.uniform(key=key, shape=x.shape)
    use_closer = rand_unif * ulp > jnp.abs(error)
    x_stoch = jnp.where(use_closer, x_closer, x_farther)
    
    return x_stoch.astype(jnp.float32)


def _random_split_like_tree(key, tree):
    """
    Takes a random number generator key and a tree, splits the key into a properly structured tree.
    credit: https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    """
    tree_structure = jax.tree.structure(tree)
    key_leaves = jax.random.split(key, tree_structure.num_leaves)
    return jax.tree.unflatten(tree_structure, key_leaves)


def tree_stochastic_round_bf16(key, tree):
    key_tree = _random_split_like_tree(key, tree)
    return jax.tree.map(stochastic_round_bf16, key_tree, tree)

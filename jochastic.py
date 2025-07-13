"""
Stochastically rounded operations between JAX tensors.

This code was written by Nestor Demeure and is licensed under the Apache 2.0 license.
You can find an up-to-date source and full description here: https://github.com/nestordemeure/jochastic
"""
import jax
import jax.numpy as jnp


def _random_split_like_tree(prngKey, tree):
    """
    Takes a random number generator key and a tree, splits the key into a properly structured tree.
    credit: https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    """
    tree_structure = jax.tree.structure(tree)
    key_leaves = jax.random.split(prngKey, tree_structure.num_leaves)
    return jax.tree.unflatten(tree_structure, key_leaves)


def _misround_result(result, error):
    """
    Given the result of a floating point operation and the numerical error introduced during that operation
    returns the floating point number on the other side of the interval containing the analytical result of the operation.

    NOTE: the output of this function will be of the type of result, the type of error does not matter.
    """
    # computes the direction in which the misrounded result lies
    finfo = jnp.finfo(result.dtype)
    direction = jnp.where(error > 0, finfo.max, finfo.min)
    # goes one ULP in that direction
    return jnp.nextafter(result, direction)


def _pseudorandom_bool(prngKey, result, alternative_result, error):
    """
    Takes  the result of a floating point operation, 
    the floating point number on the other side of the interval containing the analytical result of the operation
    and the numerical error introduced during that operation
    returns a randomly generated boolean.
    """
    # gets a random number in [0;1]
    random_unitary_float = jax.random.uniform(key=prngKey, shape=result.shape, dtype=result.dtype)
    # draws a boolean randomly, biasing the draw as a function of the ratio of the error and one ULP
    ulp = jnp.abs(alternative_result - result)
    abs_error = jnp.abs(error)
    result = random_unitary_float * ulp > abs_error
    return result


def stochastic_round_bf16(key, x_fp32):
    x_bf16 = x_fp32.astype(jnp.bfloat16)
    error = x_fp32 - x_bf16.astype(jnp.float32)
    x_bf16_alt = _misround_result(x_bf16, error)
    use_orig = _pseudorandom_bool(key, x_bf16, x_bf16_alt, error)
    x_bf16_stoch = jnp.where(use_orig, x_bf16, x_bf16_alt)
    return x_bf16_stoch.astype(jnp.float32)


def tree_stochastic_round_bf16(key, tree):
    key_tree = _random_split_like_tree(key, tree)
    return jax.tree.map(stochastic_round_bf16, key_tree, tree)

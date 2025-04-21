import jax
import jax.numpy as jnp
from flax import nnx
from collections.abc import Mapping


def flatten_dict(d, prefix=None):
    if isinstance(d, Mapping):
        out = {}
        for k, v in d.items():
            nested_prefix = k if prefix is None else f'{prefix}.{k}'
            out |= flatten_dict(v, nested_prefix)
        return out
    else:
        return {prefix: d}


def get_num_model_params(model: nnx.Module):
    graphdef, params = nnx.split(model, nnx.Param)
    n_params = jax.tree.reduce(lambda x, y: x + jnp.size(y), params, 0)
    return n_params


def halflife_to_decay(t_token, n_batch=1):
    """
    notation:
    - t_token: halflife measured in number of tokens
    - t_steps: halflife measured in number of steps
    - n_batch: number of tokens per batch
    - d: decay coefficient
    """
    t_steps = t_token / n_batch # halflife (measured in number of steps)
    d = (1/2)**(1/t_steps)
    return d


def decay_to_halflife(d, n_batch=1):
    """
    notation:
    - t_token: halflife measured in number of tokens
    - t_steps: halflife measured in number of steps
    - n_batch: number of tokens per batch
    - d: decay coefficient
    """
    # note: d**t_steps = 1/2
    t_steps = jnp.log(1/2) / jnp.log(d)
    t_token = t_steps * n_batch
    return t_token


def pad_mask(batch, eos_token_id=1):
    B, L = batch.shape

    # get idx of last EOS token
    # if there is no EOS token, equals L-1
    idx_last_eos_token = (L - 1) - jnp.argmax(batch[:, ::-1] == eos_token_id, axis=1)
    
    # only use tokens before the last EOS token
    mask = jnp.arange(L)[None, :] <= idx_last_eos_token[:, None]

    return mask # [B, L]

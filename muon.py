"""
based on https://github.com/google-deepmind/optax/blob/main/optax/contrib/_muon.py
"""

from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp

from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
    if x.ndim <= 2:
        raise ValueError(f'Input must have ≥2 dims, got {x.shape}')
    if ns_coeffs_.shape != (3,):
        raise ValueError(f'ns_coeffs must have shape (3,), got {ns_coeffs_.shape}')
    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
        a = x @ x.T
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x + b @ x
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    x /= jnp.linalg.norm(x) + eps # Ensure spectral norm is at most 1
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x)
    if transposed: x = x.T
    return x


class MuonState(NamedTuple):
    """State for the Adam algorithm."""
    count: chex.Array # shape=(), dtype=jnp.int32.
    mu: base.Updates
    ns_coeffs: chex.Array # shape=(), dtype=jnp.int32.


def scale_by_muon(
    ns_coeffs: tuple = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    adaptive: bool = False,
) -> base.GradientTransformation:

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype) # First moment
        return MuonState(jnp.zeros([], jnp.int32), mu, jnp.asarray(ns_coeffs))

    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, beta, 1)
        count_inc = numerics.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, beta, count_inc)
        # Apply Newton-schulz orthogonalization.
        updates = jax.tree.map(lambda x: orthogonalize_via_newton_schulz(x, state.ns_coeffs, ns_steps, eps), mu_hat)
        updates = jax.tree.map(lambda x: jnp.sqrt(jnp.maximum(1, x.shape[-1] / x.shape[-2])) * x, updates)
        mu = otu.tree_cast(mu, mu_dtype)
        return updates, MuonState(count_inc, mu, state.ns_coeffs)
    return base.GradientTransformation(init_fn, update_fn)

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
from optax._src import transform
from optax._src import combine


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
    if x.ndim < 2:
        raise ValueError(f'Input must have >= 2 dims, got {x.shape}')
    if ns_coeffs.shape != (3,):
        raise ValueError(f'ns_coeffs must have shape (3,), got {ns_coeffs}')
    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
        x_mT = jnp.swapaxes(x, -2, -1) # <-- changed (matrix transpose last 2 dims)
        a = x @ x_mT # <-- changed (use matrix transpose)
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x + b @ x
    transposed = False
    if x.shape[-2] > x.shape[-1]: # <-- changed (check last 2 dims)
        x = jnp.swapaxes(x, -2, -1) # <-- changed (transpose last 2 dims)
        transposed = True
    # Original line: x /= jnp.linalg.norm(x) + eps
    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps) # <-- changed (normalize each matrix slice)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x)
    if transposed: x = jnp.swapaxes(x, -2, -1) # <-- changed (transpose last 2 dims)
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
) -> base.GradientTransformation:

    def init_fn(params):
        mu = otu.tree_zeros_like(params) # First moment
        return MuonState(jnp.zeros([], jnp.int32), mu, jnp.asarray(ns_coeffs))

    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, beta, 1)
        count_inc = numerics.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, beta, count_inc)
        # Apply Newton-schulz orthogonalization.
        updates = jax.tree.map(lambda x: orthogonalize_via_newton_schulz(x, state.ns_coeffs, ns_steps, eps), mu_hat)
        updates = jax.tree.map(lambda x: jnp.sqrt(jnp.maximum(1, x.shape[-1] / x.shape[-2])) * x, updates)
        return updates, MuonState(count_inc, mu, state.ns_coeffs)
    return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: tuple = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_muon(ns_coeffs, ns_steps, beta, eps),
        transform.scale_by_learning_rate(learning_rate),
    )

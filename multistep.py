import functools
import chex
import jax
import jax.numpy as jnp
from optax import tree_utils as otu
from optax._src import base
from collections.abc import Callable
from typing import Any, NamedTuple, Optional


class SingleStepsState(NamedTuple):
    inner_opt_state: Any # The state of the wrapped optimizer.


class MultiStepsState(NamedTuple):
    mini_step: chex.Array # Current mini-step counter. At an update, this either increases by 1 or is reset to 0.
    inner_opt_state: Any # The state of the wrapped optimizer.
    grad_mean: Any # Accumulated gradients over multiple mini-steps.


def singlesteps(inner_factory, steps):

    @functools.wraps(inner_factory)
    def wrapped_transform(*args, **kwargs):
        opt = inner_factory(*args, **kwargs)

        def init_fn(params: Any) -> SingleStepsState:
            init_state = SingleStepsState(opt.init(params))
            return init_state

        def update_fn(
            updates: base.Updates,
            state: MultiStepsState,
            params: Optional[base.Params] = None,
            **kwargs,
        ):
            updates, inner_state = opt.update(updates, state.inner_opt_state, params=params, **kwargs)
            state = SingleStepsState(inner_state)
            return updates, state

        return base.GradientTransformation(init_fn, update_fn)

    return wrapped_transform


def multisteps(inner_factory, steps):

    @functools.wraps(inner_factory)
    def wrapped_transform(*args, **kwargs):
        opt = inner_factory(*args, **kwargs)

        def init_fn(params: Any) -> MultiStepsState:
            init_state = MultiStepsState(
                mini_step=jnp.zeros([], dtype=jnp.int32),
                inner_opt_state=opt.init(params),
                grad_mean=otu.tree_zeros_like(params)
            )
            return init_state

        def update_fn(
            updates: base.Updates,
            state: MultiStepsState,
            params: Optional[base.Params] = None,
        ):
            emit = state.mini_step >= (steps - 1)

            # accumulate grads
            grad_mean = jax.tree.map(lambda m, g: (state.mini_step*m + g) / (state.mini_step+1), state.grad_mean, updates)

            # if emit, do optimzier step
            # otherwise, return zero updates
            updates, inner_state = jax.lax.cond(emit,
                lambda: opt.update(grad_mean, state.inner_opt_state, params=params),
                lambda: (otu.tree_zeros_like(updates), state.inner_opt_state),
            )

            # if emit, reset accumulated gradients
            grad_mean = jax.tree.map(lambda g: (1-emit)*g, grad_mean)

            # update state
            state = MultiStepsState(
                mini_step=(state.mini_step + 1) * (1-emit),
                inner_opt_state=inner_state,
                grad_mean=grad_mean,
            )

            return updates, state

        return base.GradientTransformation(init_fn, update_fn)

    return wrapped_transform

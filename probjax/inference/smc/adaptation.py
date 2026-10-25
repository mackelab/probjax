"""Adaptors that tune an SMC move between steps.

The population after each step is a sample from the current target, so its
spread is a good estimate of the next target's geometry. Both adaptors update
the move's params after every step and keep no state.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from probjax.inference.adaptation import get_param, replace_params
from probjax.inference.base import Adaptor

__all__ = ["acceptance_rate_adaptor", "particle_adaptor"]


def _stateless(tune) -> Adaptor:
    def update(state: Any, info: Any, adaptor_state: Any, params: Any):
        return adaptor_state, tune(state, info, params), None

    return Adaptor(lambda state, params: (), update, lambda _, params: (params, None))


def particle_adaptor(field: str = "inverse_mass_matrix") -> Adaptor:
    """Set ``params[field]`` to the covariance of the particles.

    A one-dimensional field gets the variances, a two-dimensional one the full
    covariance, matching what HMC-style moves expect of an inverse mass matrix.
    """

    def tune(state: Any, _info: Any, params: Any) -> Any:
        flat = jax.vmap(lambda x: ravel_pytree(x)[0])(state.particles)
        current = jnp.asarray(get_param(params, field))
        if current.ndim == 2:
            value = jnp.atleast_2d(jnp.cov(flat, rowvar=False))
        else:
            value = jnp.var(flat, axis=0)
        return replace_params(params, **{field: value.astype(current.dtype)})

    return _stateless(tune)


def acceptance_rate_adaptor(target: float = 0.65, field: str = "step_size") -> Adaptor:
    """Scale ``params[field]`` by ``exp(acceptance_rate - target)``.

    Too many rejections shrink the step, too few grow it. Steps that report no
    acceptance rate (NaN) leave it unchanged.
    """

    def tune(_state: Any, info: Any, params: Any) -> Any:
        rate = info.acceptance_rate
        factor = jnp.where(jnp.isnan(rate), 1.0, jnp.exp(rate - target))
        return replace_params(params, **{field: get_param(params, field) * factor})

    return _stateless(tune)

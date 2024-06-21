from chex import PRNGKey
from jaxtyping import Array, ArrayLike, PyTree


from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jax.scipy.stats as stats
import numpy as np

from functools import partial
import matplotlib.pyplot as plt


import blackjax

from blackjax.base import State, Info

from probjax.inference.kernels.base import MCMCKernel
from blackjax.base import SamplingAlgorithm


class GibbsState(NamedTuple):
    position: Dict[str, ArrayLike]
    inner_state: Dict[str, State]


class GibbsInfo(NamedTuple):
    inner_info: Dict[str, Info]


def init(
    position: Dict[str, ArrayLike],
    logdensity_fn: Callable,
    inner_kernel: Dict,
) -> GibbsState:

    inner_state = {}
    for k in position.keys():

        def logdensity_k(value):
            kwargs = position.copy()
            kwargs[k] = position[k]
            return logdensity_fn(**kwargs)

        inner_state[k] = inner_kernel[k].init(
            position=position[k],
            logdensity_fn=logdensity_k,
        )

    return GibbsState(position=position, inner_state=inner_state)


def build_kernel(
    inner_kernel: Dict,
    inner_kernel_kwargs: Optional[Dict[str, Any]] = None,
    inner_kernel_steps: Optional[Dict[str, int]] = None,
) -> Callable:

    _kernels = {k: inner_kernel[k].build_kernel() for k in inner_kernel.keys()}

    def kernel(
        rng_key: PRNGKey,
        state: GibbsState,
        logdensity_fn: Callable,
        **kwargs,
    ) -> Tuple[GibbsState, GibbsInfo]:

        inner_info = {}
        inner_state = {}
        new_position = state.position.copy()

        for k in state.position.keys():
            if inner_kernel_steps is None:
                num_steps = 1
            else:
                num_steps = inner_kernel_steps[k] or 1

            rng_key, *rng_keys = jax.random.split(rng_key, num_steps + 1)

            def logdensity_k(value):
                kwargs = state.position.copy()
                kwargs[k] = value
                return logdensity_fn(**kwargs)

            if inner_kernel_kwargs is None:
                kwargs = {}
            else:
                kwargs = inner_kernel_kwargs[k] or {}

            new_inner_state, new_inner_info = _kernels[k](
                rng_keys[0], state.inner_state[k], logdensity_k, **kwargs
            )

            if num_steps > 1:
                carry = (new_inner_state, new_inner_info)

                def one_step(carry, key):
                    print(key)
                    state, info = carry
                    state, info = _kernels[k](key, state, logdensity_k, **kwargs)
                    return (state, info), None

                (new_inner_state, new_inner_info), _ = jax.lax.scan(
                    one_step, carry, jnp.array(rng_keys[1:])
                )

            new_position[k] = new_inner_state.position
            inner_state[k] = new_inner_state
            inner_info[k] = new_inner_info

        return GibbsState(position=new_position, inner_state=inner_state), GibbsInfo(
            inner_info=inner_info
        )

    return kernel


class gibbs:
    """Implements a Gibbs sampler."""

    init = staticmethod(init)
    build_kernel = staticmethod(build_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        inner_kernel: Dict,
        inner_kernel_kwargs: Optional[Dict] = None,
        inner_kernel_steps: Optional[Dict[str, int]] = None,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(inner_kernel, inner_kernel_kwargs, inner_kernel_steps)

        def init_fn(position: Array, rng_key=None):
            del rng_key
            return cls.init(
                position,
                logdensity_fn,
                inner_kernel,
            )

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                inner_kernel_kwargs=inner_kernel_kwargs,
                inner_kernel_steps=inner_kernel_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)

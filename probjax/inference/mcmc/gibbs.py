from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from blackjax.base import Info, SamplingAlgorithm, State
from probjax.utils.typing import Array, ArrayLike, RngKey


class GibbsState(NamedTuple):
    position: Dict[str, ArrayLike]
    inner_state: Dict[str, State]


class GibbsInfo(NamedTuple):
    inner_info: Dict[str, Info]


def init(
    position: Dict[str, ArrayLike],
    logdensity_fn: Callable,
    inner_kernel: Dict,
    rng_key: Optional[RngKey] = None,
) -> GibbsState:
    inner_state = {}

    def logdensity_condtional(k, value):
        kwargs = position.copy()
        kwargs[k] = value
        return logdensity_fn(**kwargs)

    for k in position:
        logdensity_k = partial(logdensity_condtional, k)

        # inspect for keyword argument "rng_key"
        if "rng_key" in inner_kernel[k].init.__code__.co_varnames:
            inner_state[k] = inner_kernel[k].init(
                position[k], logdensity_k, rng_key=rng_key
            )
        else:
            inner_state[k] = inner_kernel[k].init(position[k], logdensity_k)

    return GibbsState(position=position, inner_state=inner_state)


def build_kernel(
    inner_kernel: Dict,
    inner_kernel_kwargs: Optional[Dict[str, Any]] = None,
    inner_kernel_steps: Optional[Dict[str, int]] = None,
) -> Callable:
    _kernel_builders = {}
    for k in inner_kernel:
        if hasattr(inner_kernel[k], "build_step"):
            _kernel_builders[k] = inner_kernel[k].build_step
        else:
            raise ValueError(
                "Each inner kernel must expose build_step so it can receive a "
                "conditional logdensity_fn."
            )

    def kernel(
        rng_key: RngKey,
        state: GibbsState,
        logdensity_fn: Callable,
        **kwargs,
    ) -> Tuple[GibbsState, GibbsInfo]:
        inner_info = {}
        inner_state = {}
        new_position = state.position.copy()

        def logdensity_conditional(k, value):
            kwargs = new_position.copy()
            kwargs[k] = value
            return logdensity_fn(**kwargs)

        for k in state.position:
            num_steps = 1 if inner_kernel_steps is None else inner_kernel_steps[k] or 1

            rng_key, *rng_keys = jax.random.split(rng_key, num_steps + 1)

            logdensity_k = partial(logdensity_conditional, k)
            kwargs = {} if inner_kernel_kwargs is None else inner_kernel_kwargs[k] or {}

            kernel_step = _kernel_builders[k](logdensity_k, **kwargs)
            params = (
                inner_kernel[k].init_params(state.inner_state[k])
                if hasattr(inner_kernel[k], "init_params")
                else None
            )

            try:
                new_inner_state, new_inner_info = kernel_step(
                    rng_keys[0], state.inner_state[k], params
                )
            except TypeError:
                new_inner_state, new_inner_info = kernel_step(
                    rng_keys[0], state.inner_state[k]
                )

            if num_steps > 1:
                carry = (new_inner_state, new_inner_info)

                def one_step(carry, key):
                    state, info = carry
                    kernel_step = _kernel_builders[k](logdensity_k, **kwargs)
                    params = (
                        inner_kernel[k].init_params(state)
                        if hasattr(inner_kernel[k], "init_params")
                        else None
                    )
                    try:
                        state, info = kernel_step(key, state, params)  # noqa: B023
                    except TypeError:
                        state, info = kernel_step(key, state)  # noqa: B023
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
            return cls.init(
                position,
                logdensity_fn,
                inner_kernel,
                rng_key=rng_key,
            )

        def step_fn(rng_key: RngKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                inner_kernel_kwargs=inner_kernel_kwargs,
                inner_kernel_steps=inner_kernel_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)

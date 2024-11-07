from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey
from jax.typing import ArrayLike
from jaxtyping import PyTree

from probjax.inference.kernels.adaptation import step_size_adaption
from probjax.inference.kernels.base import MarkovKernelAPI

# Some utility functions for creating 1D slices through an N-dimensional space


def sample_random_direction(key: PRNGKey, position: ArrayLike) -> Array:
    # Sample a random direction
    direction = jax.random.normal(key, shape=position.shape)
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    return direction


def linear_slice_fn(position: ArrayLike, theta: ArrayLike) -> Callable:
    # Linear slice from position in direction theta
    def linear_slice(t: float):
        return position + t * theta

    return linear_slice


def sample_random_polynomial(
    key: PRNGKey, position: ArrayLike, degree: int = 3
) -> Array:
    # Sample a random polynomial directions
    a = jax.random.normal(key, (degree,) + position.shape)
    a = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a


def polynomial_slice_fn(
    position: ArrayLike, theta: ArrayLike, degree: int = 3
) -> Callable:
    # A polynomial slice from position in directions thetas with degrees
    degrees = jnp.arange(1, degree + 1)
    # Factorial of degrees
    factorial = jnp.cumprod(degrees)
    # Factors
    factors = 1.0 / factorial

    def polynomial_slice_fn(t: float):
        t_powers = jnp.power(t, degrees) * factors
        return position + jnp.sum(theta * t_powers[:, None], axis=0)

    return polynomial_slice_fn


def sample_random_index(key: PRNGKey, position: ArrayLike) -> Array:
    idx = jax.random.randint(key, (), minval=0, maxval=position.shape[0])
    return idx


def axis_slice_fn(position: ArrayLike, theta: ArrayLike) -> Callable:
    # Slice along an axis
    def axis_slice_fn(t: float):
        new_position = position.at[theta].add(t)
        return new_position

    return axis_slice_fn


# Slice sampler


class SliceParams(NamedTuple):
    step_size: float


class SliceState(NamedTuple):
    position: Array
    logdensity: Array
    random_arg_slice: Array


class SliceInfo(NamedTuple):
    num_evals: int
    proposal: SliceState


def init(position: ArrayLike, logdensity_fn: Callable, rng_key: PRNGKey) -> SliceState:
    log_density = logdensity_fn(position)
    return SliceState(position, log_density, rng_key)


def init_params(position: ArrayLike, step_size: float = 0.5) -> SliceParams:
    return SliceParams(step_size)


def build_kernel(
    slice_fn_builder: Callable = linear_slice_fn,
    slice_fn_arg: Callable = sample_random_direction,
):
    def kernel(
        rng_key: PRNGKey,
        state: SliceState,
        log_density_fn: Callable,
        max_evals: int = 100,
        step_size: float = 0.5,
        **kwargs,
    ) -> Tuple[SliceState, SliceInfo]:
        rng_key, key_slice, key_rejections = jax.random.split(rng_key, 3)
        direction = slice_fn_arg(key_slice, state.position, **kwargs)
        log_density = state.logdensity
        u = jax.random.uniform(key_rejections, shape=())
        y = jnp.squeeze(jnp.log(u) + log_density)

        slice_fn = slice_fn_builder(state.position, direction)

        t_lower, t_upper, evals = lower_upper_bracket(
            log_density_fn, slice_fn, y, step_size, max_evals
        )

        x_new, log_density, evals_reject = accept_reject_slice(
            log_density_fn, slice_fn, key_rejections, t_lower, t_upper, y, max_evals
        )

        new_state = SliceState(x_new, log_density, rng_key)
        info = SliceInfo(evals + evals_reject, state)
        return new_state, info

    return kernel


def build_step(
    logdensity_fn: Callable,
    max_evals: int = 100,
    slice_fn="linear",
    slice_fn_arg: Optional[Callable] = None,
    **kwargs,
) -> Callable:
    if isinstance(slice_fn, str):
        if slice_fn == "linear":
            slice_fn = linear_slice_fn
            slice_fn_arg = sample_random_direction
        elif slice_fn == "poly":
            slice_fn = polynomial_slice_fn
            slice_fn_arg = partial(sample_random_polynomial, **kwargs)
        elif slice_fn == "axis":
            slice_fn = axis_slice_fn
            slice_fn_arg = sample_random_index
        elif slice_fn == "none":
            slice_fn = lambda pos, theta: lambda t: pos + t
            slice_fn_arg = lambda key, pos: None
        else:
            raise ValueError("Invalid slice function")

    kernel = build_kernel(slice_fn, slice_fn_arg)

    def step_fn(rng_key: PRNGKey, state: SliceState, params: SliceParams):
        return kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size=params.step_size,
            max_evals=max_evals,
        )

    return step_fn


def build_adaptation(
    logdensity_fn: Callable,
    max_evals: int = 100,
    slice_fn="linear",
    slice_fn_arg: Optional[Callable] = None,
) -> Callable:
    def adapt_parms(
        key: PRNGKey,
        position: PyTree,
        params: SliceParams,
        num_steps: int = 100,
        target_num_evals: int = 10,
        method: str = "step_size",
        t0: int = 10,
        gamma: float = 0.05,
        kappa: float = 0.75,
    ) -> Tuple[SliceState, SliceParams]:
        if method == "step_size":
            key, rng_init = jax.random.split(key)
            adaption_alg = step_size_adaption(
                Slice,
                logdensity_fn,
                params,
                target=float(target_num_evals) / max_evals,
                target_from_info_fn=lambda info: info.num_evals / max_evals,
                t0=t0,
                gamma=gamma,
                kappa=kappa,
                init_kwargs={"rng_key": rng_init},
                algorithm_kwargs={
                    "max_evals": max_evals,
                    "slice_fn": slice_fn,
                    "slice_fn_arg": slice_fn_arg,
                },
            )
        else:
            raise ValueError("Invalid method")

        out, _ = adaption_alg.run(key, position, num_steps)
        return out.state, out.parameters

    return adapt_parms


class Slice(MarkovKernelAPI):
    init = init
    build_step = build_step
    init_params = init_params
    build_adaptation = build_adaptation


def lower_upper_bracket(
    log_density_fn: Callable,
    slice_fn: Callable,
    log_density_bound: ArrayLike,
    step_size: float,
    max_steps: int,
) -> Tuple[Array, Array, int]:
    # Bracket expansion phase
    def cond_fn(carry):
        i, _, _, mask = carry
        return jnp.any(mask) & (i < max_steps)

    def body_fn(carry):
        i, t, step_size, mask = carry

        t += step_size
        x = slice_fn(t)
        potential = log_density_fn(x)
        mask = potential >= log_density_bound

        return (i + 1, t, step_size, mask)

    evals1, t_upper, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (0, jnp.array(step_size), step_size, True)
    )
    evals2, t_lower, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (0, jnp.array(-step_size), -step_size, True)
    )
    total_evals = evals1 + evals2

    return t_lower, t_upper, total_evals


def accept_reject_slice(
    log_density_fn: Callable,
    slice_fn: Callable,
    key: PRNGKey,
    t_lower: ArrayLike,
    t_upper: ArrayLike,
    log_density_bound: ArrayLike,
    max_steps: int,
) -> Tuple[Array, Array, int]:
    def cond_fn_reject(carry):
        i, _, _, _, _, _, mask_reject = carry
        return jnp.any(mask_reject) & (i < max_steps)

    def body_fn_reject(carry):
        i, key, t_lower, t_upper, _, h, mask_reject = carry

        key, key_reject = jax.random.split(key)
        t_new = jax.random.uniform(key_reject, shape=(), minval=t_lower, maxval=t_upper)
        x_new = slice_fn(t_new)
        h = log_density_fn(x_new)

        mask_reject = h <= log_density_bound

        new_t_lower = jax.lax.cond(t_new < 0, lambda: t_new, lambda: t_lower)
        new_t_upper = jax.lax.cond(t_new > 0, lambda: t_new, lambda: t_upper)

        return (i + 1, key, new_t_lower, new_t_upper, x_new, h, mask_reject)

    key, key_reject = jax.random.split(key)
    t_new = jax.random.uniform(key_reject, shape=(), minval=t_lower, maxval=t_upper)
    x_new = slice_fn(t_new)
    h = log_density_fn(x_new)
    mask_reject = h <= log_density_bound
    new_t_lower = jax.lax.cond(t_new <= 0, lambda: t_new, lambda: t_lower)
    new_t_upper = jax.lax.cond(t_new >= 0, lambda: t_new, lambda: t_upper)

    evals, _, _, _, x_new, h, _ = jax.lax.while_loop(
        cond_fn_reject,
        body_fn_reject,
        (1, key, new_t_lower, new_t_upper, x_new, h, mask_reject),
    )

    return x_new, h, evals

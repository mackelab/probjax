from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from probjax.inference.mcmc.adaptation import slice_step_size_adaptor
from probjax.inference.mcmc.base import MarkovKernelAPI, make_step_from_kernel
from probjax.utils.typing import Array, ArrayLike, Scalar

# Some utility functions for creating 1D slices through an N-dimensional space


def sample_random_direction(rng, position: Array) -> Array:
    """Sample a random unit direction in the space of ``position``.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key used to draw the direction.
    position : array_like
        Point used to infer the shape of the direction.

    Returns
    -------
    Array
        Unit vector with the same shape as ``position``.
    """
    # Sample a random direction
    direction = jax.random.normal(rng, shape=position.shape)
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    return direction


def linear_slice_fn(
    position: ArrayLike, theta: ArrayLike
) -> Callable[[float], ArrayLike]:
    """Build a linear slice through ``position`` in direction ``theta``.

    Parameters
    ----------
    position : array_like
        Base point of the slice.
    theta : array_like
        Direction of the slice.

    Returns
    -------
    Callable
        A function mapping a scalar step ``t`` to ``position + t * theta``.
    """

    # Linear slice from position in direction theta
    def linear_slice(t: float) -> ArrayLike:
        return position + t * theta

    return linear_slice


def sample_random_polynomial(rng, position: Array, degree: int = 3) -> Array:
    """Sample random polynomial directions with the shape of ``position``.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key used to draw the directions.
    position : array_like
        Point used to infer the spatial shape.
    degree : int, optional
        Number of polynomial directions to draw. Default is 3.

    Returns
    -------
    Array
        Array of normalized directions with leading dimension ``degree``.
    """
    # Sample a random polynomial directions
    a = jax.random.normal(rng, (degree,) + position.shape)
    a = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a


def polynomial_slice_fn(
    position: ArrayLike, theta: ArrayLike, degree: int = 3
) -> Callable[[float], ArrayLike]:
    """Build a polynomial slice through ``position`` in directions ``theta``.

    Parameters
    ----------
    position : array_like
        Base point of the slice.
    theta : array_like
        Polynomial direction coefficients, indexed by degree.
    degree : int, optional
        Degree of the polynomial slice. Default is 3.

    Returns
    -------
    Callable
        A function mapping a scalar step ``t`` to the polynomial slice point.
    """
    # A polynomial slice from position in directions thetas with degrees
    degrees = jnp.arange(1, degree + 1)
    # Factorial of degrees
    factorial = jnp.cumprod(degrees)
    # Factors
    factors = 1.0 / factorial

    def polynomial_slice_fn(t: float) -> ArrayLike:
        t_powers = jnp.power(t, degrees) * factors
        return position + jnp.sum(theta * t_powers[:, None], axis=0)

    return polynomial_slice_fn


def sample_random_index(rng, position: Array) -> Array:
    """Sample a random coordinate index in the first axis of ``position``.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key used to draw the index.
    position : array_like
        Point used to bound the sampled index.

    Returns
    -------
    Array
        Scalar integer index in ``[0, position.shape[0])``.
    """
    idx = jax.random.randint(rng, (), minval=0, maxval=position.shape[0])
    return idx


def axis_slice_fn(position: Array, theta: Array) -> Callable[[float], Array]:
    """Build a slice that moves ``position`` along a single coordinate.

    Parameters
    ----------
    position : array_like
        Base point of the slice.
    theta : array_like
        Index of the coordinate to move along.

    Returns
    -------
    Callable
        A function mapping a scalar step ``t`` to the shifted point.
    """

    # Slice along an axis
    def axis_slice_fn(t: float) -> Array:
        new_position = position.at[theta].add(t)
        return new_position

    return axis_slice_fn


# Slice sampler


class SliceParams(NamedTuple):
    """Static parameters of the slice sampler.

    Attributes
    ----------
    step_size : float
        Initial bracket expansion step size.
    """

    step_size: Scalar


class SliceState(NamedTuple):
    """Markov chain state of the slice sampler.

    Attributes
    ----------
    position : Array
        Current position of the chain.
    logdensity : Array
        Log density evaluated at ``position``.
    random_arg_slice : Array
        Random argument (e.g. direction) used to build the slice.
    """

    position: Array
    logdensity: Array
    random_arg_slice: Array


class SliceInfo(NamedTuple):
    """Auxiliary information returned by a slice kernel step.

    Attributes
    ----------
    num_evals : int
        Number of density evaluations used by the step.
    num_expansions : int
        How many of them stepped the bracket out.
    proposal : SliceState
        State before the step, kept for diagnostics.
    """

    num_evals: int
    num_expansions: int
    proposal: SliceState


def init(position: Array, logdensity_fn: Callable, rng) -> SliceState:
    """Initialize the slice sampler state.

    Parameters
    ----------
    position : array_like
        Initial position of the chain.
    logdensity_fn : callable
        Function returning the (unnormalized) log density at a position.
    rng : jax.random.PRNGKey
        Random key stored for the initial slice argument.

    Returns
    -------
    SliceState
        Initial sampler state.
    """
    log_density = logdensity_fn(position)
    return SliceState(position, log_density, rng)


def init_params(state: Any, step_size: float = 0.5) -> SliceParams:
    """Initialize the static parameters of the slice sampler.

    Parameters
    ----------
    state : array_like
        Current state; accepted for API compatibility and unused.
    step_size : float, optional
        Initial bracket expansion step size. Default is 0.5.

    Returns
    -------
    SliceParams
        Sampler parameters.
    """
    return SliceParams(step_size)


def build_kernel(
    slice_fn_builder: Callable = linear_slice_fn,
    slice_fn_arg: Callable = sample_random_direction,
):
    """Build the raw slice-sampling kernel.

    Parameters
    ----------
    slice_fn_builder : callable, optional
        Factory building the slice function from a position and direction.
        Defaults to :func:`linear_slice_fn`.
    slice_fn_arg : callable, optional
        Function sampling the slice argument (e.g. direction). Defaults to
        :func:`sample_random_direction`.

    Returns
    -------
    callable
        Kernel mapping ``(rng, state, log_density_fn, ...)`` to a new
        ``(SliceState, SliceInfo)`` pair.
    """

    def kernel(
        rng,
        state: SliceState,
        log_density_fn: Callable,
        max_evals: int = 100,
        step_size: float = 0.5,
        **kwargs: Any,
    ) -> tuple[SliceState, SliceInfo]:
        rng_state, rng_slice, rng_rejections = jax.random.split(rng, 3)
        direction = slice_fn_arg(rng_slice, state.position, **kwargs)
        log_density = state.logdensity
        u = jax.random.uniform(rng_rejections, shape=())
        y = jnp.squeeze(jnp.log(u) + log_density)

        slice_fn = slice_fn_builder(state.position, direction)

        t_lower, t_upper, evals = lower_upper_bracket(
            log_density_fn, slice_fn, y, step_size, max_evals
        )

        x_new, log_density, evals_reject, accepted = accept_reject_slice(
            log_density_fn, slice_fn, rng_rejections, t_lower, t_upper, y, max_evals
        )
        # Out of evaluations without an accepted point: stay put.
        x_new = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old), x_new, state.position
        )
        log_density = jnp.where(accepted, log_density, state.logdensity)

        new_state = SliceState(x_new, log_density, rng_state)
        info = SliceInfo(evals + evals_reject, evals, state)
        return new_state, info

    return kernel


def build_step(
    logdensity_fn: Callable,
    max_evals: int = 100,
    slice_fn: str | Callable = "linear",
    slice_fn_arg: Callable | None = None,
    **kwargs: Any,
) -> Callable:
    """Build the slice-sampling step; see :class:`slice` for the arguments."""
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

    if slice_fn_arg is None:
        raise ValueError("A slice function needs a slice argument sampler.")
    kernel_builder = lambda: build_kernel(slice_fn, slice_fn_arg)
    return make_step_from_kernel(
        logdensity_fn,
        kernel_builder,
        call_defaults={"max_evals": max_evals},
    )


class slice(MarkovKernelAPI):
    """Slice sampling along random one-dimensional slices.

    Each step brackets the slice by stepping out with ``step_size`` and then
    shrinks it until a point is accepted. Tunable parameters come from
    ``kernel.init_params(state, step_size=0.5)`` (see :class:`SliceParams`).
    Default adaptation: :func:`slice_step_size_adaptor`, balancing bracket
    expansions and contractions.

    Args:
        logdensity_fn: Log density of the target.
        max_evals: Maximum number of density evaluations per step.
        slice_fn: Slice geometry: ``"linear"`` (random direction), ``"poly"``
            (random polynomial curve), ``"axis"`` (random coordinate),
            ``"none"`` or a custom ``(position, arg) -> (t -> position)``
            builder.
        slice_fn_arg: ``(key, position) -> arg`` sampling the slice argument;
            inferred when ``slice_fn`` is a string.
        **kwargs: Options for the ``"poly"`` slice-argument sampler.
    """

    init = staticmethod(init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
    default_adaptation_fn = staticmethod(slice_step_size_adaptor)


def lower_upper_bracket(
    log_density_fn: Callable,
    slice_fn: Callable,
    log_density_bound: ArrayLike,
    step_size: float,
    max_steps: int,
) -> tuple[Array, Array, int]:
    """Expand a bracket around a slice until it leaves the target level set.

    Parameters
    ----------
    log_density_fn : callable
        Function returning the (unnormalized) log density at a position.
    slice_fn : callable
        Slice function mapping a scalar step to a point in parameter space.
    log_density_bound : array_like
        Log-density threshold defining the slice level set.
    step_size : float
        Initial bracket expansion step size.
    max_steps : int
        Maximum number of expansion steps in each direction.

    Returns
    -------
    tuple
        ``(t_lower, t_upper, total_evals)`` giving the bracket endpoints in
        slice coordinates and the number of density evaluations performed.
    """

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

    step_size = jnp.asarray(step_size, dtype=float)
    evals1, t_upper, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (0, step_size, step_size, True)
    )
    evals2, t_lower, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (0, -step_size, -step_size, True)
    )
    total_evals = evals1 + evals2

    return t_lower, t_upper, total_evals


def accept_reject_slice(
    log_density_fn: Callable,
    slice_fn: Callable,
    rng,
    t_lower: ArrayLike,
    t_upper: ArrayLike,
    log_density_bound: ArrayLike,
    max_steps: int,
) -> tuple[Array, Array, int, Array]:
    """Draw a slice sample by rejection within a bracket.

    Parameters
    ----------
    log_density_fn : callable
        Function returning the (unnormalized) log density at a position.
    slice_fn : callable
        Slice function mapping a scalar step to a point in parameter space.
    key : jax.random.PRNGKey
        Random key used to sample candidate points.
    t_lower : array_like
        Lower bracket endpoint in slice coordinates.
    t_upper : array_like
        Upper bracket endpoint in slice coordinates.
    log_density_bound : array_like
        Log-density threshold defining the slice level set.
    max_steps : int
        Maximum number of rejection iterations.

    Returns
    -------
    tuple
        ``(x_new, h, evals, accepted)``: the last candidate, its log density,
        the number of evaluations and whether the candidate lies in the slice
        (``False`` when ``max_steps`` ran out).
    """

    def cond_fn_reject(carry):
        i, _, _, _, _, _, mask_reject = carry
        return jnp.any(mask_reject) & (i < max_steps)

    def body_fn_reject(carry):
        i, key, t_lower, t_upper, _, h, mask_reject = carry

        key, key_reject = jax.random.split(key)
        t_new = jax.random.uniform(key_reject, shape=(), minval=t_lower, maxval=t_upper)
        x_new = slice_fn(t_new)
        h = log_density_fn(x_new)

        # Written so that a NaN density is rejected.
        mask_reject = ~(h > log_density_bound)

        new_t_lower = jax.lax.cond(t_new < 0, lambda: t_new, lambda: t_lower)
        new_t_upper = jax.lax.cond(t_new > 0, lambda: t_new, lambda: t_upper)

        return (i + 1, key, new_t_lower, new_t_upper, x_new, h, mask_reject)

    key, key_reject = jax.random.split(rng)
    t_new = jax.random.uniform(key_reject, shape=(), minval=t_lower, maxval=t_upper)
    x_new = slice_fn(t_new)
    h = log_density_fn(x_new)
    mask_reject = ~(h > log_density_bound)
    new_t_lower = jax.lax.cond(t_new <= 0, lambda: t_new, lambda: t_lower)
    new_t_upper = jax.lax.cond(t_new >= 0, lambda: t_new, lambda: t_upper)

    evals, _, _, _, x_new, h, mask_reject = jax.lax.while_loop(
        cond_fn_reject,
        body_fn_reject,
        (1, key, new_t_lower, new_t_upper, x_new, h, mask_reject),
    )

    return x_new, h, evals, ~mask_reject

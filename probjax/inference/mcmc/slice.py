from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from probjax.inference.mcmc.base import make_kernel_api, make_step_from_kernel
from probjax.utils.typing import Array, ArrayLike, RngKey

# Some utility functions for creating 1D slices through an N-dimensional space


def sample_random_direction(key: RngKey, position: ArrayLike) -> Array:
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
    direction = jax.random.normal(key, shape=position.shape)
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    return direction


def linear_slice_fn(position: ArrayLike, theta: ArrayLike) -> Callable:
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
    def linear_slice(t: float):
        return position + t * theta

    return linear_slice


def sample_random_polynomial(
    key: RngKey, position: ArrayLike, degree: int = 3
) -> Array:
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
    a = jax.random.normal(key, (degree,) + position.shape)
    a = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    return a


def polynomial_slice_fn(
    position: ArrayLike, theta: ArrayLike, degree: int = 3
) -> Callable:
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

    def polynomial_slice_fn(t: float):
        t_powers = jnp.power(t, degrees) * factors
        return position + jnp.sum(theta * t_powers[:, None], axis=0)

    return polynomial_slice_fn


def sample_random_index(key: RngKey, position: ArrayLike) -> Array:
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
    idx = jax.random.randint(key, (), minval=0, maxval=position.shape[0])
    return idx


def axis_slice_fn(position: ArrayLike, theta: ArrayLike) -> Callable:
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
    def axis_slice_fn(t: float):
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

    step_size: float


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
    proposal : SliceState
        State before the step, kept for diagnostics.
    """

    num_evals: int
    proposal: SliceState


def init(position: ArrayLike, logdensity_fn: Callable, rng_key: RngKey) -> SliceState:
    """Initialize the slice sampler state.

    Parameters
    ----------
    position : array_like
        Initial position of the chain.
    logdensity_fn : callable
        Function returning the (unnormalized) log density at a position.
    rng_key : jax.random.PRNGKey
        Random key stored for the initial slice argument.

    Returns
    -------
    SliceState
        Initial sampler state.
    """
    log_density = logdensity_fn(position)
    return SliceState(position, log_density, rng_key)


def init_params(state: ArrayLike, step_size: float = 0.5) -> SliceParams:
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
        Kernel mapping ``(rng_key, state, log_density_fn, ...)`` to a new
        ``(SliceState, SliceInfo)`` pair.
    """

    def kernel(
        rng_key: RngKey,
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
    """Build a slice-sampling step function.

    Parameters
    ----------
    logdensity_fn : callable
        Function returning the (unnormalized) log density at a position.
    max_evals : int, optional
        Maximum number of density evaluations per slice. Default is 100.
    slice_fn : str or callable, optional
        Slice geometry. One of ``"linear"``, ``"poly"``, ``"axis"`` or
        ``"none"``, or a custom slice builder. Default is ``"linear"``.
    slice_fn_arg : callable, optional
        Function sampling the slice argument. Inferred from ``slice_fn`` when
        given as a string.
    **kwargs : Any
        Extra keyword arguments forwarded to the slice-argument sampler.

    Returns
    -------
    callable
        Step function mapping ``(rng_key, state, params)`` to a new state and
        info pair.
    """
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

    kernel_builder = lambda: build_kernel(slice_fn, slice_fn_arg)
    return make_step_from_kernel(
        logdensity_fn,
        kernel_builder,
        call_defaults={"max_evals": max_evals},
    )


def _slice_adaptation(
    *, target_num_evaluations=5, max_evaluations=100, **kwargs
):
    """Default adaptation; ``max_evaluations`` must match the kernel's ``max_evals``."""
    from probjax.inference.mcmc.adaptation import slice_step_size_adaptor

    return slice_step_size_adaptor(target_num_evaluations, max_evaluations, **kwargs)


slice = make_kernel_api(
    name="slice",
    init_fn=init,
    init_params_fn=init_params,
    build_step_fn=build_step,
    default_adaptation_fn=_slice_adaptation,
)


def lower_upper_bracket(
    log_density_fn: Callable,
    slice_fn: Callable,
    log_density_bound: ArrayLike,
    step_size: float,
    max_steps: int,
) -> Tuple[Array, Array, int]:
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
    key: RngKey,
    t_lower: ArrayLike,
    t_upper: ArrayLike,
    log_density_bound: ArrayLike,
    max_steps: int,
) -> Tuple[Array, Array, int]:
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
        ``(x_new, h, evals)`` giving the accepted position, its log density,
        and the number of evaluations performed.
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

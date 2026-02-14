from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import Array
from jaxtyping import Key, PyTree

from probjax.utils.sdeutil.core import _sdeint


def sdeint(
    rng: Key,
    drift: Callable[..., PyTree[Array]],
    diffusion: Callable[..., PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    *args,
    method: str = "euler_maruyama",
    dtype: Optional[jnp.dtype] = jnp.float32,
    sde_type: str = "ito",
    return_brownian: bool = False,
    return_state: bool = False,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    **kwargs,
) -> Union[
    Optional[PyTree[Array]],
    Tuple[Any, Optional[PyTree[Array]]],
    Tuple[
        Optional[PyTree[Array]],
        Optional[PyTree[Array]],
    ],
    Tuple[
        Any,
        Tuple[Optional[PyTree[Array]], Optional[PyTree[Array]]],
    ],
]:
    """Solve a stochastic differential equation.

    This is a high-level interface for solving SDEs using various numerical methods.
    It supports both Ito and Stratonovich SDEs with automatic noise-shape inference.

    Args:
        rng: Random number generator key
        drift: The drift function f(t, y, *args) that defines the deterministic part
            of the SDE dy = f(t, y, *args)dt + g(t, y, *args)dWt.
            The function should take time t as first argument, state y as second argument,
            and any additional arguments specified in *args.
        diffusion: The diffusion function g(t, y, *args) that defines the stochastic part
            of the SDE dy = f(t, y, *args)dt + g(t, y, *args)dWt.
            The function should take time t as first argument, state y as second argument,
            and any additional arguments specified in *args.
        y0: Initial state. Can be a single array or a PyTree of arrays.
        ts: Time points at which to evaluate the solution. Must be a 1D array of increasing values.
        *args: Additional positional arguments forwarded to drift and diffusion.
        method: Integration method to use. Available methods include:
            - "euler_maruyama": Euler-Maruyama method (order 0.5)
            - "exp_euler_maruyama": Exponential Euler-Maruyama (`split_drift` required)
            - "milstein": Milstein method (order 1.0)
            - "srk": Stochastic Runge-Kutta methods
        dtype: Data type for computation. Defaults to float32.
        sde_type: Type of SDE interpretation:
            - "ito": Ito interpretation (default)
            - "stratonovich": Stratonovich interpretation
            Noise layout is inferred automatically from `diffusion` output shape:
            scalar/vector outputs are treated as diagonal noise; matrix outputs
            are treated as full (including rectangular) noise.
        return_brownian: Whether to return Brownian motion paths along with the solution.
            Requires `collect_trace=True`.
        return_state: Whether to return solver state along with the solution.
        filter_state: Optional function to filter the state during integration.
            Useful for tracking specific components of the state. Returning ``None``
            disables tracing entirely.
        collect_trace: Whether to record the filtered quantity at every time step
            (`True`, default) or only return the filtered terminal state (`False`).
            Must be `True` when returning Brownian paths.
        check_points: Optional sequence of indices for grid integration.
        **kwargs: Additional keyword arguments forwarded to drift and diffusion.

    Returns:
        When `return_brownian=False`, returns the filtered trajectory (if `collect_trace`
        is True) or the filtered terminal state (if `collect_trace` is False). When
        `return_brownian=True`, returns a tuple of (state_trace, brownian_trace), both
        stacked over all time points. If `return_state=True`, the solver state is
        prepended to the output tuple.

    Notes:
        - The solution includes the initial condition y0 as the first point.
        - All computations are performed in the specified dtype.
        - The function is JIT-compiled for improved performance.
        - The state can be a PyTree of arrays, allowing for complex state structures.

    Example:
        >>> import jax.numpy as jnp
        >>> from probjax.utils.sdeint import sdeint
        >>> from jax import random
        >>>
        >>> # Geometric Brownian motion with PyTree state
        >>> def drift(t, state, mu, sigma):
        ...     return {"price": mu * state["price"]}
        >>> def diffusion(t, state, mu, sigma):
        ...     return {"price": sigma * state["price"]}
        >>>
        >>> # Initial conditions and parameters
        >>> rng = random.PRNGKey(0)
        >>> y0 = {"price": jnp.array([1.0])}
        >>> ts = jnp.linspace(0, 1, 100)
        >>> params = (0.1, 0.2)  # mu, sigma
        >>>
        >>> # Solve using Euler-Maruyama method
        >>> ys = sdeint(rng, drift, diffusion, y0, ts, *params)
    """
    return _sdeint(
        rng,
        drift,
        diffusion,
        y0,
        ts,
        args,
        kwargs,
        method=method,
        dtype=dtype,
        sde_type=sde_type,
        return_brownian=return_brownian,
        return_state=return_state,
        filter_state=filter_state,
        collect_trace=collect_trace,
        check_points=check_points,
    )

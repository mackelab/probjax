from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import Array
from jaxtyping import Key, PyTree
from jax import random

from probjax.utils.sdeutil.core import _sdeint


def sdeint(
    rng: Key,
    drift: Callable[[Array, PyTree[Array], ...], PyTree[Array]],
    diffusion: Callable[[Array, PyTree[Array], ...], PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    *args,
    method: str = "euler_maruyama",
    dtype: Optional[jnp.dtype] = jnp.float32,
    sde_type: str = "ito",
    return_brownian: bool = False,
    return_state: bool = False,
    noise_type: Optional[str] = None,
    filter_state: Optional[Callable[[PyTree[Array]], PyTree[Array]]] = None,
    check_points: Optional[Sequence[int]] = None,
) -> Union[
    PyTree[Array], Tuple[Any, PyTree[Array]], Tuple[Any, PyTree[Array], PyTree[Array]]
]:
    """Solve a stochastic differential equation.

    This is a high-level interface for solving SDEs using various numerical methods.
    It supports both Ito and Stratonovich SDEs, with options for different integration
    methods and noise types.

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
        *args: Additional arguments for the drift and diffusion functions.
        method: Integration method to use. Available methods include:
            - "euler_maruyama": Euler-Maruyama method (order 0.5)
            - "milstein": Milstein method (order 1.0)
            - "srk": Stochastic Runge-Kutta methods
        dtype: Data type for computation. Defaults to float32.
        sde_type: Type of SDE interpretation:
            - "ito": Ito interpretation (default)
            - "stratonovich": Stratonovich interpretation
        return_brownian: Whether to return Brownian motion paths along with the solution.
        return_state: Whether to return solver state along with the solution.
        noise_type: Type of noise in the diffusion term:
            - "diagonal": Diagonal noise (default for scalar diffusion)
            - "general": General noise matrix
        filter_state: Optional function to filter the state during integration.
            Useful for tracking specific components of the state.
        check_points: Optional sequence of indices for grid integration.

    Returns:
        Depending on return_brownian and return_state:
        - If return_brownian=False, return_state=False: solution trajectory
        - If return_brownian=False, return_state=True: (state, solution)
        - If return_brownian=True, return_state=False: (solution, brownian_paths)
        - If return_brownian=True, return_state=True: (state, solution, brownian_paths)

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
        *args,
        method=method,
        dtype=dtype,
        sde_type=sde_type,
        return_brownian=return_brownian,
        return_state=return_state,
        noise_type=noise_type,
        filter_output=filter_state,
        check_points=check_points,
    )

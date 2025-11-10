from typing import Callable, Optional, Sequence

import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from probjax.utils.odeutil import AdaptiveParams, _odeint


# @partial(custom_inverse, inv_argnum=1)
def odeint(
    drift: Callable[[Array, PyTree[Array], ...], PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    *args,
    method: str = "rk4",
    dtype: Optional[jnp.dtype] = jnp.float32,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[AdaptiveParams] = None,
) -> Optional[PyTree[Array]]:
    """Solve an ordinary differential equation.

    This is a high-level interface for solving ODEs using various numerical methods.
    It supports both fixed-step and adaptive-step integration, with a focus on
    performance through JAX transformations.

    Args:
        drift: The drift function f(t, y, *args) that defines the ODE dy/dt = f(t, y, *args).
            The function should take time t as first argument, state y as second argument,
            and any additional arguments specified in *args.
        y0: Initial state. Can be a single array or a PyTree of arrays.
        ts: Time points at which to evaluate the solution. Must be a 1D array of increasing values.
        *args: Additional arguments for the drift function.
        method: Integration method to use. Available methods include:
            Fixed-step methods:
                - "euler": Forward Euler method (order 1)
                - "rk4": 4th order Runge-Kutta method (order 4)

            Adaptive-step methods:
                - "dopri5": Dormand-Prince 5th order method (order 5)
                - "tsit5": Tsitouras 5th order method (order 5)
                - "dopri8": Dormand-Prince 8th order method (order 8)
                - "tsit8": Tsitouras 8th order method (order 8)
                - "bogacki_shampine": Bogacki-Shampine 3rd order method (order 3)
        dtype: Data type for computation. Defaults to float32.
        filter_state: Optional function to filter the state during integration.
            Useful for tracking specific components of the state. Returning ``None``
            disables tracing for the selected components.
        collect_trace: If ``True`` (default), return the filtered state at every
            requested time point (including the initial condition). If ``False``,
            return only the filtered terminal state, avoiding time-series storage.
        check_points: Optional sequence of indices for grid integration.
            Only used with fixed-step methods.
        adaptive_params: Parameters for adaptive integration methods.
            Controls error tolerances and step size adaptation.

    Returns:
        PyTree containing either:
            - The time-series trace with leading dimension ``len(ts)`` when
              ``collect_trace=True`` and the filter returns a PyTree.
            - The filtered terminal state when ``collect_trace=False``.
            - ``None`` when the provided filter returns ``None``.

    Notes:
        - The solution includes the initial condition y0 as the first point.
        - All computations are performed in the specified dtype.
        - The function is JIT-compiled for improved performance.

    Example:
        >>> import jax.numpy as jnp
        >>> from probjax.utils.odeint import odeint
        >>>
        >>> # Lotka-Volterra predator-prey model
        >>> def lotka_volterra(t, y, alpha, beta, delta, gamma):
        ...     prey, predator = y
        ...     dprey = alpha * prey - beta * prey * predator
        ...     dpredator = delta * prey * predator - gamma * predator
        ...     return jnp.array([dprey, dpredator])
        >>>
        >>> # Initial conditions and parameters
        >>> y0 = jnp.array([40.0, 9.0])  # prey, predator
        >>> ts = jnp.linspace(0, 10, 100)
        >>> params = (1.0, 0.1, 0.075, 0.5)  # alpha, beta, delta, gamma
        >>>
        >>> # Solve using adaptive integration
        >>> ys = odeint(lotka_volterra, y0, ts, *params, method="dopri5")
    """
    return _odeint(
        drift,
        y0,
        ts,
        *args,
        method=method,
        dtype=dtype,
        filter_state=filter_state,
        collect_trace=collect_trace,
        check_points=check_points,
        adaptive_params=adaptive_params,
    )


# Current custom inverse does not support Traced side effects...
# odeint.definv(_inv_odeint)
# odeint.definv_and_logdet(_inv_logdet_odeint)

from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from probjax.utils.functions import generic_drift
from probjax.utils.odeutil import AdaptiveParams, _odeint_custom
from probjax.utils.odeutil.inversion import SampleDist, TraceEstimator


def _wrap_if_plain_callable(
    drift: Callable[..., PyTree[Array]],
) -> Callable[..., PyTree[Array]]:
    """Wrap plain Python callables in :class:`generic_drift` so they flow as
    a pytree through ``jax.jit`` / ``custom_inverse``.

    If ``drift`` already is a registered pytree (marker subclasses,
    ``eqx.Module``, user-registered dataclasses, ...) it flows through
    unchanged — its array leaves participate in transformations, its
    callable/config leaves ride along as aux.
    """
    leaves, _ = jax.tree_util.tree_flatten(drift)
    if len(leaves) == 1 and leaves[0] is drift:
        return generic_drift(fn=drift)
    return drift


def odeint(
    drift: Callable[..., PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    *args: Any,
    method: str = "rk4",
    dtype: Optional[jnp.dtype] = jnp.float32,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[AdaptiveParams] = None,
    logdet_rng: Optional[Array] = None,
    trace_estimator: TraceEstimator = "exact",
    num_samples: int = 1,
    sample_dist: SampleDist = "rademacher",
) -> Optional[PyTree[Array]]:
    """Solve an ordinary differential equation ``dy/dt = drift(t, y, *args)``.

    ``drift`` may be any of:

    - a plain Python callable ``drift(t, y, *args)`` — it is automatically
      wrapped in :class:`probjax.utils.functions.generic_drift` so it rides
      as a pytree through ``jax.jit`` and the ``custom_inverse`` primitive;
    - a registered JAX pytree node (``eqx.Module``, ``flax.struct.PyTreeNode``,
      :class:`~probjax.utils.functions.Drift` subclass, ...) — its array
      leaves participate in transformations natively, non-array fields ride
      as aux.

    Drift keyword arguments are no longer supported. Pass parameters
    positionally via ``*args``, or bind them up-front with ``functools.partial``
    (or ``drift.bind_args(...)`` on a :class:`~probjax.utils.functions.Drift`).

    Args:
        drift: The drift function ``f(t, y, *args)``.
        y0: Initial state. Single array or pytree of arrays.
        ts: Time points at which to evaluate the solution.
        *args: Positional arguments forwarded to ``drift``.
        method: Integration method name.
        dtype: Computation dtype (default ``float32``).
        filter_state: Optional function to filter the state during integration.
        collect_trace: If ``True`` (default) return the filtered state at
            every time point; otherwise return only the filtered terminal state.
        check_points: Optional index sequence for checkpointed grid integration.
        adaptive_params: Parameters for adaptive integration methods.
        logdet_rng: RNG key for stochastic log-determinant estimators.
            Required when ``trace_estimator="hutchinson"``. Ignored on the
            forward path; only consumed by
            :func:`probjax.core.inverse_and_logabsdet`.
        trace_estimator: Log-det trace estimator used when the function is
            inverted via ``inverse_and_logabsdet``. One of:

            - ``"exact"`` (default): full Jacobian per step (O(d²) cost).
            - ``"hutchinson"``: FFJORD-style stochastic estimator
              ``tr(J) ≈ mean_k vᵀ_k J v_k`` via one JVP per probe vector;
              probe vectors are fixed across the trajectory so
              ``∫tr(J)dt`` stays unbiased. Requires ``logdet_rng``.
            - a callable ``trace_fn(drift_flat, t, x_flat, args) -> scalar``
              for custom structured-Jacobian strategies.
        num_samples: Hutchinson probe-vector count per trajectory. Higher
            values reduce variance linearly in cost.
        sample_dist: Hutchinson probe distribution — ``"rademacher"``
            (default, minimum-variance for general matrices) or
            ``"normal"``.

    Returns:
        Pytree containing either the time-series trace (when
        ``collect_trace=True``) or the filtered terminal state.

    Example:
        >>> import jax.numpy as jnp
        >>> from probjax.utils.odeint import odeint
        >>>
        >>> def lotka_volterra(t, y, alpha, beta, delta, gamma):
        ...     prey, predator = y
        ...     dprey = alpha * prey - beta * prey * predator
        ...     dpredator = delta * prey * predator - gamma * predator
        ...     return jnp.array([dprey, dpredator])
        >>>
        >>> y0 = jnp.array([40.0, 9.0])
        >>> ts = jnp.linspace(0, 10, 100)
        >>> ys = odeint(lotka_volterra, y0, ts, 1.0, 0.1, 0.075, 0.5,
        ...             method="dopri5")
    """
    drift = _wrap_if_plain_callable(drift)
    return _odeint_custom(
        y0,
        drift,
        ts,
        tuple(args),
        logdet_rng,
        method=method,
        dtype=dtype,
        filter_state=filter_state,
        collect_trace=collect_trace,
        check_points=check_points,
        adaptive_params=adaptive_params,
        trace_estimator=trace_estimator,
        num_samples=num_samples,
        sample_dist=sample_dist,
    )

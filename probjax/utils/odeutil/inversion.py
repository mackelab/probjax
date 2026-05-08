"""ODE inversion utilities and ``custom_inverse``-wrapped solver.

This module provides:

- :func:`_odeint_custom` — the ``custom_inverse``-wrapped ODE solver.
- :func:`_inv_odeint` — forward integration reversed in time, used as the
  inverse rule of ``_odeint_custom``.
- :func:`_inv_logdet_odeint` — same as above plus the log-determinant of
  the change-of-variables Jacobian accumulated along the reverse
  trajectory.
- :func:`make_augmented_drift` — builds the augmented drift that carries
  the log-det alongside the state during inverse integration. Supports
  exact trace (O(d²) Jacobian eval per step), Hutchinson's stochastic
  trace estimator (FFJORD-style, O(d) per step), and user-supplied trace
  functions for structured Jacobians.

Signature note: the ``custom_inverse`` primitive assumes the inverted
argument is a single pytree leaf at the same flat position as its logical
position. To keep that invariant, ``y0`` sits at position 0 and the drift
(a multi-leaf pytree) is passed at position 1.
"""

from functools import partial
from typing import Any, Callable, Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.utils.functions import generic_drift
from probjax.utils.jaxutils import ravel_pytree
from probjax.utils.odeutil.adaptive import AdaptiveParams
from probjax.utils.odeutil.core import _odeint

TraceEstimator = Union[Literal["exact", "hutchinson"], Callable]
SampleDist = Literal["rademacher", "normal"]


def _extract_final_state(
    trace_or_state: PyTree[Array],
    ts_len: int,
    collect_trace: bool,
) -> PyTree[Array]:
    """Pull the terminal state out of either a trajectory trace or a state."""
    if not collect_trace:
        return trace_or_state

    def select_last(x):
        if hasattr(x, "shape") and x.shape and x.shape[0] == ts_len:
            return x[-1]
        return x

    return jax.tree_util.tree_map(select_last, trace_or_state)


@partial(custom_inverse, inv_argnum=0)
def _odeint_custom(
    y0: PyTree[Array],
    drift: Callable[..., PyTree[Array]],
    ts: Array,
    args: Sequence[Any] = (),
    logdet_rng: Optional[Array] = None,
    *,
    method: str = "rk4",
    dtype: Optional[jnp.dtype] = jnp.float32,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[AdaptiveParams] = None,
    trace_estimator: TraceEstimator = "exact",
    num_samples: int = 1,
    sample_dist: SampleDist = "rademacher",
) -> Optional[PyTree[Array]]:
    """``custom_inverse``-wrapped ODE solver.

    ``drift`` flows as a dynamic pytree argument so its array leaves (e.g.
    weights in an ``eqx.Module``) participate in ``jax.jit`` / ``jax.grad``
    / ``jax.vmap``. See :func:`probjax.utils.odeint.odeint` for the public
    API.

    Log-det estimator kwargs (``trace_estimator``, ``num_samples``,
    ``sample_dist``) and ``logdet_rng`` only affect the inverse-and-logdet
    path; forward integration ignores them. They are exposed here because
    ``custom_inverse`` threads call-site kwargs into both forward and
    inverse rules.
    """
    del logdet_rng, trace_estimator, num_samples, sample_dist
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


def _inv_odeint(
    ys: Array,
    drift: Callable[..., PyTree[Array]],
    ts: Array,
    args: Sequence[Any] = (),
    logdet_rng: Optional[Array] = None,
    *,
    method: str = "rk4",
    dtype: Optional[jnp.dtype] = jnp.float32,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[Any] = None,
    trace_estimator: TraceEstimator = "exact",
    num_samples: int = 1,
    sample_dist: SampleDist = "rademacher",
) -> PyTree[Array]:
    """Inverse ODE integration (no log-det).

    Integrates the drift backwards from ``ys`` (a trajectory when
    ``collect_trace=True`` or the terminal state otherwise) to recover the
    initial condition. The log-det-estimator kwargs are ignored here; they
    ride along only because ``custom_inverse`` replays the call-site kwargs.

    ``filter_state`` is disallowed because the inverse problem is undefined
    when only filtered trajectories are available.
    """
    del logdet_rng, trace_estimator, num_samples, sample_dist
    if filter_state is not None:
        raise ValueError("odeint inversion is undefined when filter_state is provided.")

    final_state = _extract_final_state(ys, ts.shape[0], collect_trace)
    return _odeint(
        drift,
        final_state,
        ts[::-1],
        *tuple(args),
        method=method,
        dtype=dtype,
        filter_state=None,
        collect_trace=False,
        check_points=check_points,
        adaptive_params=adaptive_params,
    )


def _sample_trace_vectors(
    rng: Array,
    d: int,
    num_samples: int,
    sample_dist: SampleDist,
    dtype: Optional[jnp.dtype],
) -> Array:
    """Sample ``(num_samples, d)`` Hutchinson probe vectors."""
    target_dtype = dtype if dtype is not None else jnp.float32
    shape = (num_samples, d)
    if sample_dist == "rademacher":
        return jax.random.rademacher(rng, shape).astype(target_dtype)
    if sample_dist == "normal":
        return jax.random.normal(rng, shape, dtype=target_dtype)
    raise ValueError(
        f"Unknown sample_dist={sample_dist!r}; expected 'rademacher' or 'normal'."
    )


def _make_exact_aug_drift(
    drift: Callable,
    x_example: PyTree[Array],
    jac_fn: Callable = jax.jacrev,
):
    """Exact log-det: one full Jacobian per step."""
    _, unravel = ravel_pytree(x_example)

    def drift_flat(t, x_flat, *args):
        x = unravel(x_flat)
        dx = drift(t, x, *args)
        dx_flat, _ = ravel_pytree(dx)
        return dx_flat

    jac_flat = jac_fn(drift_flat, argnums=1)

    def aug_drift(t, state, *args):
        x, _ = state
        x_flat, _ = ravel_pytree(x)
        dx = drift(t, x, *args)
        J = jac_flat(t, x_flat, *args)
        dlogdet = jnp.trace(J)[None]
        return dx, dlogdet

    return generic_drift(fn=aug_drift)


def _make_hutchinson_aug_drift(
    drift: Callable,
    x_example: PyTree[Array],
    v_samples: Array,
):
    """Hutchinson log-det: tr(J) ≈ mean_k(vᵀ_k J v_k) via one JVP per sample.

    ``v_samples`` has shape ``(num_samples, d)`` where ``d`` is the raveled
    state dimension. The probe vectors are fixed for the whole trajectory
    (FFJORD's single-sample-per-path trick), so ``∫tr(J)dt`` is estimated
    without threading an RNG through each integration step.
    """
    _, unravel = ravel_pytree(x_example)

    def drift_flat(t, x_flat, *args):
        x = unravel(x_flat)
        dx = drift(t, x, *args)
        dx_flat, _ = ravel_pytree(dx)
        return dx_flat

    def aug_drift(t, state, *args):
        x, _ = state
        x_flat, _ = ravel_pytree(x)
        dx = drift(t, x, *args)

        def single_trace(v):
            _, Jv = jax.jvp(lambda xf: drift_flat(t, xf, *args), (x_flat,), (v,))
            return jnp.sum(v * Jv)

        traces = jax.vmap(single_trace)(v_samples)
        dlogdet = jnp.mean(traces, axis=0)[None]
        return dx, dlogdet

    return generic_drift(fn=aug_drift)


def _make_custom_aug_drift(
    drift: Callable,
    x_example: PyTree[Array],
    trace_fn: Callable,
):
    """User-supplied ``trace_fn(drift_flat, t, x_flat, args) -> scalar`` hook."""
    _, unravel = ravel_pytree(x_example)

    def drift_flat(t, x_flat, *args):
        x = unravel(x_flat)
        dx = drift(t, x, *args)
        dx_flat, _ = ravel_pytree(dx)
        return dx_flat

    def aug_drift(t, state, *args):
        x, _ = state
        x_flat, _ = ravel_pytree(x)
        dx = drift(t, x, *args)
        tr = trace_fn(drift_flat, t, x_flat, args)
        dlogdet = jnp.asarray(tr).reshape((1,))
        return dx, dlogdet

    return generic_drift(fn=aug_drift)


def make_augmented_drift(
    drift: Callable,
    x_example: PyTree[Array],
    *,
    trace_estimator: TraceEstimator = "exact",
    v_samples: Optional[Array] = None,
    jac_fn: Callable = jax.jacrev,
):
    """Augment a drift with a log-determinant state.

    Args:
        drift: ``(t, x, *args) -> pytree(x)`` — the original drift.
        x_example: pytree with the same structure as the runtime states,
            used to build flatten/unflatten.
        trace_estimator: ``"exact"`` (default, full Jacobian), ``"hutchinson"``
            (requires ``v_samples``), or a callable
            ``trace_fn(drift_flat, t, x_flat, args) -> scalar`` for custom
            structured-Jacobian strategies.
        v_samples: ``(num_samples, d)`` probe vectors for Hutchinson. Must
            be pre-sampled outside the integration so they stay fixed along
            the trajectory.
        jac_fn: Jacobian transform for the exact path (``jax.jacrev`` or
            ``jax.jacfwd``).

    Returns:
        Callable ``(t, (x, logdet), *args) -> (dx, dlogdet)`` wrapped as a
        :class:`~probjax.utils.functions.generic_drift` so it flows through
        ``_odeint`` as a registered pytree.
    """
    if callable(trace_estimator) and not isinstance(trace_estimator, str):
        return _make_custom_aug_drift(drift, x_example, trace_estimator)
    if trace_estimator == "exact":
        return _make_exact_aug_drift(drift, x_example, jac_fn=jac_fn)
    if trace_estimator == "hutchinson":
        if v_samples is None:
            raise ValueError(
                "trace_estimator='hutchinson' requires v_samples (shape "
                "(num_samples, d)); sample them before calling "
                "make_augmented_drift."
            )
        return _make_hutchinson_aug_drift(drift, x_example, v_samples)
    raise ValueError(
        f"Unknown trace_estimator={trace_estimator!r}; expected 'exact', "
        "'hutchinson', or a callable."
    )


def _inv_logdet_odeint(
    ys: Array,
    drift: Callable[..., PyTree[Array]],
    ts: Array,
    args: Sequence[Any] = (),
    logdet_rng: Optional[Array] = None,
    *,
    method: str = "rk4",
    dtype: Optional[jnp.dtype] = jnp.float32,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[Any] = None,
    trace_estimator: TraceEstimator = "exact",
    num_samples: int = 1,
    sample_dist: SampleDist = "rademacher",
):
    """Inverse ODE integration with log-determinant.

    Returns ``(y0, logdet)`` where ``logdet`` estimates
    ``-∫_0^T tr(∂f/∂x) dt`` accumulated along the reverse trajectory.

    The trace estimator is selected by ``trace_estimator``:

    - ``"exact"`` (default): evaluates the full ``d × d`` Jacobian per
      step via ``jax.jacrev``. Most accurate, O(d²) cost.
    - ``"hutchinson"``: FFJORD-style stochastic estimator
      ``tr(J) ≈ mean_k vᵀ_k J v_k`` via one JVP per probe vector.
      Probe vectors ``v`` are sampled once (``num_samples`` of them) from
      ``sample_dist`` before integration begins and held fixed across all
      steps — this keeps ``∫tr(J)dt`` unbiased without threading an RNG
      through each step. Requires ``logdet_rng``. O(d · num_samples) cost.
    - user callable ``trace_fn(drift_flat, t, x_flat, args) -> scalar``:
      bypass estimator machinery entirely, e.g. when the Jacobian has
      exploitable structure (banded, low-rank, analytic trace).
    """
    if filter_state is not None:
        raise ValueError("odeint inversion is undefined when filter_state is provided.")

    final_state = _extract_final_state(ys, ts.shape[0], collect_trace)

    v_samples: Optional[Array] = None
    if trace_estimator == "hutchinson":
        if logdet_rng is None:
            raise ValueError(
                "trace_estimator='hutchinson' requires logdet_rng to be a "
                "jax.random.PRNGKey."
            )
        flat_state, _ = ravel_pytree(final_state)
        v_samples = _sample_trace_vectors(
            logdet_rng,
            int(flat_state.shape[0]),
            int(num_samples),
            sample_dist,
            dtype,
        )

    drift_aug = make_augmented_drift(
        drift,
        final_state,
        trace_estimator=trace_estimator,
        v_samples=v_samples,
    )
    logdet0 = jnp.zeros((1,))
    yT, logdetsT = _odeint(
        drift_aug,
        (final_state, logdet0),
        ts[::-1],
        *tuple(args),
        method=method,
        dtype=dtype,
        filter_state=None,
        collect_trace=False,
        check_points=check_points,
        adaptive_params=adaptive_params,
    )
    return yT, logdetsT


# Register inverse functions for _odeint_custom
_odeint_custom.definv(_inv_odeint)
_odeint_custom.definv_and_logdet(_inv_logdet_odeint)

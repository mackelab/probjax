from typing import Any, Callable, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from probjax.utils.jaxutils import ravel_pytree
from probjax.utils.odeutil.core import _odeint


def _bind_drift_kwargs(
    drift: Callable[..., PyTree[Array]],
    kwargs: Optional[Mapping[str, Any]],
) -> Callable[..., PyTree[Array]]:
    """Bind a kwargs dict into drift by closure.

    ``custom_inverse`` captures the kwargs dict as a dynamic positional arg, so
    its leaves remain traced when needed. Wrapping here (inside the traced
    region) keeps those tracers live in the forward/inverse jaxpr rather than
    stashing them in a Python closure at call time.
    """
    if not kwargs:
        return drift

    kw = dict(kwargs)
    bind_args = getattr(drift, "bind_args", None)
    if callable(bind_args):
        return bind_args(**kw)

    def drift_with_kwargs(t: Array, y: PyTree[Array], *args: Any):
        return drift(t, y, *args, **kw)

    return drift_with_kwargs


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


def _inv_odeint(
    drift: Callable[..., PyTree[Array]],
    ys: Array,
    ts: Array,
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    method: str = "rk4",
    dtype: Optional[jnp.dtype] = jnp.float32,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[Any] = None,
) -> PyTree[Array]:
    """Inverse ODE integration.

    Integrates the drift backwards from ``ys`` (a trajectory when
    ``collect_trace=True`` or the terminal state otherwise) to recover the
    initial condition.

    ``filter_state`` is disallowed because the inverse problem is undefined
    when only filtered trajectories are available.
    """
    if filter_state is not None:
        raise ValueError(
            "odeint inversion is undefined when filter_state is provided."
        )

    drift_fn = _bind_drift_kwargs(drift, kwargs)
    final_state = _extract_final_state(ys, ts.shape[0], collect_trace)
    return _odeint(
        drift_fn,
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


def make_augmented_drift(drift, x_example, jac_fn=jax.jacrev):
    """Augment a drift with a log-determinant state.

    Args:
        drift: ``(t, x, *args) -> pytree(x)`` — the original drift.
        x_example: pytree with the same structure as the states used at call
            time, used to build flatten/unflatten.

    Returns:
        Callable ``(t, (x, logdet), *args) -> (dx, dlogdet)``.
    """
    x0_flat, unravel = ravel_pytree(x_example)

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

    return aug_drift


def _inv_logdet_odeint(
    drift: Callable[..., PyTree[Array]],
    ys: Array,
    ts: Array,
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    method: str = "rk4",
    dtype: Optional[jnp.dtype] = jnp.float32,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[Any] = None,
):
    """Inverse ODE integration with log-determinant.

    Returns ``(y0, logdet)`` where ``logdet`` is the log-determinant of the
    change-of-variables Jacobian accumulated along the reverse trajectory.
    """
    if filter_state is not None:
        raise ValueError(
            "odeint inversion is undefined when filter_state is provided."
        )

    drift_fn = _bind_drift_kwargs(drift, kwargs)
    final_state = _extract_final_state(ys, ts.shape[0], collect_trace)
    drift_aug = make_augmented_drift(drift_fn, final_state)
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

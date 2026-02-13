import jax
import jax.numpy as jnp
from jax import Array

from probjax.utils.jaxutils import ravel_pytree
from probjax.utils.odeutil.core import _odeint


def _bind_drift_kwargs(drift, drift_kwargs):
    if not drift_kwargs:
        return drift

    kwargs_dict = dict(drift_kwargs)

    def drift_with_kwargs(t, y, *args):
        return drift(t, y, *args, **kwargs_dict)

    return drift_with_kwargs


def _split_drift_args(args):
    """Support both legacy *args and new (drift_args, drift_kwargs) payload."""
    if len(args) == 2 and isinstance(args[0], tuple) and isinstance(args[1], dict):
        return tuple(args[0]), dict(args[1])
    return tuple(args), {}


def _ensure_invertible_kwargs(kwargs):
    if kwargs.get("filter_state") is not None:
        raise ValueError("odeint inversion is undefined when filter_state is provided.")
    return kwargs.pop("collect_trace", True)


def _extract_final_state(trace_or_state, ts_len: int, collect_trace: bool):
    if not collect_trace:
        return trace_or_state

    def select_last(x):
        if hasattr(x, "shape") and x.shape and x.shape[0] == ts_len:
            return x[-1]
        return x

    return jax.tree_util.tree_map(select_last, trace_or_state)


def _inv_odeint(drift, ys: Array, ts: Array, *args, **kwargs):
    """Inverse ODE integration.

    Args:
        drift: The drift function
        ys: The final state
        ts: Time points
        *args: Additional arguments for the drift function
        **kwargs: Additional keyword arguments for _odeint

    Returns:
        The initial state
    """
    drift_args, drift_kwargs = _split_drift_args(args)
    drift_fn = _bind_drift_kwargs(drift, drift_kwargs)
    collect_trace = _ensure_invertible_kwargs(kwargs)
    final_state = _extract_final_state(ys, ts.shape[0], collect_trace)
    kwargs = {**kwargs, "collect_trace": False, "filter_state": None}
    yT = _odeint(
        drift_fn,
        final_state,
        ts[::-1],
        *drift_args,
        **kwargs,
    )
    return yT


def make_augmented_drift(drift, x_example, jac_fn=jax.jacrev):
    """
    drift: (t, x, *args) -> pytree(x)
    x_example: pytree with same structure as the states you'll use
    """
    # Build flatten/unflatten using example structure
    x0_flat, unravel = ravel_pytree(x_example)

    def drift_flat(t, x_flat, *args):
        x = unravel(x_flat)
        dx = drift(t, x, *args)
        dx_flat, _ = ravel_pytree(dx)
        return dx_flat

    # Jacobian wrt flat state
    jac_flat = jac_fn(drift_flat, argnums=1)

    def aug_drift(t, state, *args):
        x, logdet = state

        # flatten current state using same convention
        x_flat, _ = ravel_pytree(x)

        dx = drift(t, x, *args)
        J = jac_flat(t, x_flat, *args)  # shape (n, n)
        dlogdet = jnp.trace(J)[None]  # (1,)

        return dx, dlogdet

    return aug_drift


def _inv_logdet_odeint(drift, ys, ts, *args, **kwargs):
    """Inverse ODE integration with log determinant computation.

    Args:
        drift: The drift function
        ys: The final state
        ts: Time points
        *args: Additional arguments for the drift function
        **kwargs: Additional keyword arguments for _odeint

    Returns:
        Tuple of (initial state, log determinant)
    """
    drift_args, drift_kwargs = _split_drift_args(args)
    drift_fn = _bind_drift_kwargs(drift, drift_kwargs)
    collect_trace = _ensure_invertible_kwargs(kwargs)
    final_state = _extract_final_state(ys, ts.shape[0], collect_trace)
    drift_aug = make_augmented_drift(drift_fn, final_state)
    kwargs = {**kwargs, "collect_trace": False, "filter_state": None}
    logdet0 = jnp.zeros((1,))
    result = _odeint(
        drift_aug,
        (final_state, logdet0),
        ts[::-1],
        *drift_args,
        **kwargs,
    )

    yT, logdetsT = result

    return yT, logdetsT

from dataclasses import fields, is_dataclass, replace
import types

import jax
import jax.numpy as jnp
from jax import Array

from probjax.utils.jaxutils import ravel_pytree
from probjax.utils.odeutil.core import _odeint


def _bind_drift_kwargs(drift, drift_kwargs):
    if not drift_kwargs:
        return drift

    kwargs_dict = dict(drift_kwargs)

    def drift_with_kwargs(t, y, *args, **dynamic_kwargs):
        return drift(t, y, *args, **kwargs_dict, **dynamic_kwargs)

    return drift_with_kwargs


def _has_tracer(tree):
    return any(
        isinstance(x, jax.core.Tracer)
        for x in jax.tree_util.tree_leaves(tree)
    )


def _make_cell(value):
    def inner():
        return value

    return inner.__closure__[0]


def _extract_function_dynamic_cells(fun):
    if not isinstance(fun, types.FunctionType):
        return None
    closure = fun.__closure__
    if not closure:
        return None

    dynamic_indices = []
    dynamic_values = []
    for i, cell in enumerate(closure):
        value = cell.cell_contents
        if _has_tracer(value):
            dynamic_indices.append(i)
            dynamic_values.append(value)

    if not dynamic_indices:
        return None
    return tuple(dynamic_indices), tuple(dynamic_values)


def _rebuild_function_with_dynamic_cells(fun, dynamic_indices, dynamic_values):
    closure = fun.__closure__
    if closure is None:
        return fun

    all_values = [cell.cell_contents for cell in closure]
    for i, value in zip(dynamic_indices, dynamic_values, strict=True):
        all_values[i] = value

    rebuilt = types.FunctionType(
        fun.__code__,
        fun.__globals__,
        name=fun.__name__,
        argdefs=fun.__defaults__,
        closure=tuple(_make_cell(v) for v in all_values),
    )
    if fun.__kwdefaults__ is not None:
        rebuilt.__kwdefaults__ = dict(fun.__kwdefaults__)
    return rebuilt


def _extract_dataclass_function_dynamic_cells(obj):
    if not is_dataclass(obj):
        return None

    dynamic_index_map = {}
    dynamic_value_map = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        dynamic = _extract_function_dynamic_cells(value)
        if dynamic is None:
            continue
        dyn_indices, dyn_values = dynamic
        dynamic_index_map[field.name] = dyn_indices
        dynamic_value_map[field.name] = dyn_values

    if not dynamic_index_map:
        return None
    return dynamic_index_map, dynamic_value_map


def _rebuild_dataclass_with_dynamic_cells(
    obj,
    dynamic_index_map,
    dynamic_value_map,
):
    updates = {}
    for name, dynamic_indices in dynamic_index_map.items():
        fun = getattr(obj, name)
        if not isinstance(fun, types.FunctionType):
            continue
        updates[name] = _rebuild_function_with_dynamic_cells(
            fun,
            dynamic_indices,
            dynamic_value_map[name],
        )
    if not updates:
        return obj
    return replace(obj, **updates)


def _prepare_drift_call(drift, drift_args, drift_kwargs):
    """Prepare drift invocation without closing over traced values."""
    drift_args_tuple = tuple(drift_args)

    dynamic_kwargs = None
    if drift_kwargs:
        kwargs_dict = dict(drift_kwargs)
        static_kwargs = {}
        traced_kwargs = {}
        for key, value in kwargs_dict.items():
            if _has_tracer(value):
                traced_kwargs[key] = value
            else:
                static_kwargs[key] = value

        if static_kwargs:
            drift = _bind_drift_kwargs(drift, static_kwargs)
        if traced_kwargs:
            dynamic_kwargs = traced_kwargs

    function_dynamic = _extract_function_dynamic_cells(drift)
    dataclass_dynamic = (
        None if function_dynamic is not None else _extract_dataclass_function_dynamic_cells(drift)
    )

    if (
        dynamic_kwargs is None
        and function_dynamic is None
        and dataclass_dynamic is None
    ):
        return drift, drift_args_tuple

    def drift_with_dynamic_payload(t, y, *packed):
        idx = 0
        positional_args = packed[: len(drift_args_tuple)]
        idx += len(drift_args_tuple)

        kw = {}
        if dynamic_kwargs is not None:
            kw = packed[idx]
            idx += 1

        drift_local = drift
        if function_dynamic is not None:
            dynamic_indices, _ = function_dynamic
            closure_values = packed[idx]
            idx += 1
            drift_local = _rebuild_function_with_dynamic_cells(
                drift,
                dynamic_indices,
                closure_values,
            )
        elif dataclass_dynamic is not None:
            dynamic_index_map, _ = dataclass_dynamic
            closure_payload = packed[idx]
            idx += 1
            drift_local = _rebuild_dataclass_with_dynamic_cells(
                drift,
                dynamic_index_map,
                closure_payload,
            )

        return drift_local(t, y, *positional_args, **kw)

    packed_args = drift_args_tuple
    if dynamic_kwargs is not None:
        packed_args = packed_args + (dynamic_kwargs,)
    if function_dynamic is not None:
        _, dynamic_values = function_dynamic
        packed_args = packed_args + (dynamic_values,)
    elif dataclass_dynamic is not None:
        _, dynamic_value_map = dataclass_dynamic
        packed_args = packed_args + (dynamic_value_map,)

    return drift_with_dynamic_payload, packed_args


def _split_drift_args(args):
    """Support both legacy *args and new (drift_args, drift_kwargs) payload."""
    if len(args) == 2 and isinstance(args[0], tuple) and (
        isinstance(args[1], dict) or args[1] is None
    ):
        kwargs = {} if args[1] is None else dict(args[1])
        return tuple(args[0]), kwargs
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
    drift_fn, packed_args = _prepare_drift_call(drift, drift_args, drift_kwargs)
    collect_trace = _ensure_invertible_kwargs(kwargs)
    final_state = _extract_final_state(ys, ts.shape[0], collect_trace)
    kwargs = {**kwargs, "collect_trace": False, "filter_state": None}
    yT = _odeint(
        drift_fn,
        final_state,
        ts[::-1],
        *packed_args,
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
    drift_fn, packed_args = _prepare_drift_call(drift, drift_args, drift_kwargs)
    collect_trace = _ensure_invertible_kwargs(kwargs)
    final_state = _extract_final_state(ys, ts.shape[0], collect_trace)
    drift_aug = make_augmented_drift(drift_fn, final_state)
    kwargs = {**kwargs, "collect_trace": False, "filter_state": None}
    logdet0 = jnp.zeros((1,))
    result = _odeint(
        drift_aug,
        (final_state, logdet0),
        ts[::-1],
        *packed_args,
        **kwargs,
    )

    yT, logdetsT = result

    return yT, logdetsT

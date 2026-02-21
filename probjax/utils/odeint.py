from functools import partial
from dataclasses import fields, is_dataclass, replace
import types
from typing import Any, Callable, Mapping, Optional, Sequence, cast

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.utils.odeutil import (
    AdaptiveParams,
    _inv_logdet_odeint,
    _inv_odeint,
    _odeint,
)


def _bind_drift_kwargs(
    drift: Callable[..., PyTree[Array]],
    drift_kwargs: Optional[Mapping[str, Any]],
) -> Callable[..., PyTree[Array]]:
    """Bind keyword arguments to drift while keeping positional ODE arguments."""
    if not drift_kwargs:
        return drift

    kwargs_dict = dict(drift_kwargs)
    bind_args = getattr(drift, "bind_args", None)
    if callable(bind_args):
        return cast(Callable[..., PyTree[Array]], bind_args(**kwargs_dict))

    def drift_with_kwargs(t: Array, y: PyTree[Array], *args: Any, **dynamic_kwargs: Any):
        return drift(t, y, *args, **kwargs_dict, **dynamic_kwargs)

    return drift_with_kwargs


def _has_tracer(tree: Any) -> bool:
    return any(
        isinstance(x, jax.core.Tracer)
        for x in jax.tree_util.tree_leaves(tree)
    )


def _make_cell(value: Any):
    def inner():
        return value

    return inner.__closure__[0]


def _extract_function_dynamic_cells(
    fun: Any,
) -> tuple[tuple[int, ...], tuple[Any, ...]] | None:
    if not isinstance(fun, types.FunctionType):
        return None
    closure = fun.__closure__
    if not closure:
        return None

    dynamic_indices: list[int] = []
    dynamic_values: list[Any] = []
    for i, cell in enumerate(closure):
        value = cell.cell_contents
        if _has_tracer(value):
            dynamic_indices.append(i)
            dynamic_values.append(value)

    if not dynamic_indices:
        return None
    return tuple(dynamic_indices), tuple(dynamic_values)


def _rebuild_function_with_dynamic_cells(
    fun: types.FunctionType,
    dynamic_indices: tuple[int, ...],
    dynamic_values: tuple[Any, ...],
) -> types.FunctionType:
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


def _extract_dataclass_function_dynamic_cells(
    obj: Any,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[Any, ...]]] | None:
    if not is_dataclass(obj):
        return None

    dynamic_index_map: dict[str, tuple[int, ...]] = {}
    dynamic_value_map: dict[str, tuple[Any, ...]] = {}
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
    obj: Any,
    dynamic_index_map: dict[str, tuple[int, ...]],
    dynamic_value_map: Mapping[str, tuple[Any, ...]],
) -> Any:
    updates: dict[str, Any] = {}
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


def _prepare_drift_call(
    drift: Callable[..., PyTree[Array]],
    drift_args: Sequence[Any],
    drift_kwargs: Optional[Mapping[str, Any]],
) -> tuple[Callable[..., PyTree[Array]], tuple[Any, ...]]:
    """Prepare drift invocation without closing over traced values."""
    drift_args_tuple = tuple(drift_args)

    dynamic_kwargs: Optional[dict[str, Any]] = None
    if drift_kwargs:
        kwargs_dict = dict(drift_kwargs)
        static_kwargs: dict[str, Any] = {}
        traced_kwargs: dict[str, Any] = {}
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

    def drift_with_dynamic_payload(t: Array, y: PyTree[Array], *packed: Any):
        idx = 0
        positional_args = packed[: len(drift_args_tuple)]
        idx += len(drift_args_tuple)

        kw: Mapping[str, Any] = {}
        if dynamic_kwargs is not None:
            kw = packed[idx]
            idx += 1

        drift_local: Any = drift
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


def _partition_drift_kwargs(
    drift_kwargs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    if not drift_kwargs:
        return {}, None

    static_kwargs: dict[str, Any] = {}
    traced_kwargs: dict[str, Any] = {}
    for key, value in dict(drift_kwargs).items():
        if _has_tracer(value):
            traced_kwargs[key] = value
        else:
            static_kwargs[key] = value

    return static_kwargs, traced_kwargs or None


def _lift_drift_traced_closure(
    drift: Callable[..., PyTree[Array]],
) -> tuple[Callable[..., PyTree[Array]], tuple[Any, ...]]:
    function_dynamic = _extract_function_dynamic_cells(drift)
    if function_dynamic is not None:
        dynamic_indices, dynamic_values = function_dynamic

        def drift_with_dynamic_closure(t: Array, y: PyTree[Array], *packed: Any):
            *drift_args, closure_values = packed
            rebuilt = _rebuild_function_with_dynamic_cells(
                cast(types.FunctionType, drift),
                dynamic_indices,
                closure_values,
            )
            return rebuilt(t, y, *drift_args)

        return drift_with_dynamic_closure, (dynamic_values,)

    dataclass_dynamic = _extract_dataclass_function_dynamic_cells(drift)
    if dataclass_dynamic is not None:
        dynamic_index_map, dynamic_value_map = dataclass_dynamic

        def drift_with_dynamic_closure(t: Array, y: PyTree[Array], *packed: Any):
            *drift_args, closure_payload = packed
            rebuilt = _rebuild_dataclass_with_dynamic_cells(
                drift,
                dynamic_index_map,
                closure_payload,
            )
            return rebuilt(t, y, *drift_args)

        return drift_with_dynamic_closure, (dynamic_value_map,)

    return drift, ()


@partial(custom_inverse, inv_argnum=1, static_argnums=(0,))
def _odeint_custom(
    drift: Callable[..., PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    drift_args: Sequence[Any] = (),
    drift_kwargs: Optional[Mapping[str, Any]] = None,
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
    drift_fn, packed_args = _prepare_drift_call(drift, drift_args, drift_kwargs)
    return _odeint(
        drift_fn,
        y0,
        ts,
        *packed_args,
        method=method,
        dtype=dtype,
        filter_state=filter_state,
        collect_trace=collect_trace,
        check_points=check_points,
        adaptive_params=adaptive_params,
    )


def odeint(
    drift: Callable[..., PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    *args,
    method: str = "rk4",
    dtype: Optional[jnp.dtype] = jnp.float32,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[AdaptiveParams] = None,
    **drift_kwargs,
) -> Optional[PyTree[Array]]:
    """Solve an ordinary differential equation.

    Additional keyword arguments are forwarded to `drift`.
    """
    static_drift_kwargs, dynamic_drift_kwargs = _partition_drift_kwargs(drift_kwargs)
    if static_drift_kwargs:
        drift = _bind_drift_kwargs(drift, static_drift_kwargs)

    drift, lifted_args = _lift_drift_traced_closure(drift)
    return _odeint_custom(
        drift,
        y0,
        ts,
        args + lifted_args,
        dynamic_drift_kwargs,
        method=method,
        dtype=dtype,
        filter_state=filter_state,
        collect_trace=collect_trace,
        check_points=check_points,
        adaptive_params=adaptive_params,
    )


_odeint_custom.definv(_inv_odeint)
_odeint_custom.definv_and_logdet(_inv_logdet_odeint)

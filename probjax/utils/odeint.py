from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence

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


def _leaf_is_array_like(leaf: Any) -> bool:
    """Heuristic for whether ``leaf`` is an array-like JAX value."""
    return hasattr(leaf, "shape") and hasattr(leaf, "dtype")


def _split_drift_pytree(
    drift: Callable[..., PyTree[Array]],
) -> tuple[Callable[..., PyTree[Array]], tuple[Any, ...]]:
    """Split a pytree-callable drift into (static wrapper, dynamic leaves).

    If ``drift`` is a registered JAX pytree node whose leaves are array-like
    values (e.g. ``equinox.Module``, ``flax.struct.PyTreeNode``), those leaves
    are lifted into positional arguments so they can ride through
    ``jax.jit``/``custom_inverse`` as traced values while the drift itself is
    represented by a small static reconstruction wrapper.

    Otherwise the drift is returned unchanged and should be passed as a static
    argument (it must therefore be hashable).
    """
    leaves, treedef = jax.tree_util.tree_flatten(drift)

    # A non-pytree callable (plain function, ``functools.partial``, marker
    # dataclass not registered as a pytree, ...) flattens to a single leaf
    # that IS the callable itself. In that case there is nothing to lift.
    if len(leaves) == 1 and leaves[0] is drift:
        return drift, ()

    # Pytree with no leaves (e.g. ``split_drift`` where both fields are in
    # aux_data) can also flow through unchanged — there is nothing dynamic.
    if not leaves:
        return drift, ()

    # Only lift when at least one leaf is array-like; otherwise assume the
    # caller intends the drift to be static (non-traced configuration data).
    if not any(_leaf_is_array_like(leaf) for leaf in leaves):
        return drift, ()

    n_leaves = len(leaves)

    def drift_from_leaves(t: Array, y: PyTree[Array], *dynamic: Any, **kwargs: Any):
        drift_leaves = list(dynamic[:n_leaves])
        rest = dynamic[n_leaves:]
        reconstructed = jax.tree_util.tree_unflatten(treedef, drift_leaves)
        return reconstructed(t, y, *rest, **kwargs)

    return drift_from_leaves, tuple(leaves)


@partial(custom_inverse, inv_argnum=1, static_argnums=(0,))
def _odeint_custom(
    drift: Callable[..., PyTree[Array]],
    y0: PyTree[Array],
    ts: Array,
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    method: str = "rk4",
    dtype: Optional[jnp.dtype] = jnp.float32,
    filter_state: Optional[Callable[[PyTree[Array]], Optional[PyTree[Array]]]] = None,
    collect_trace: bool = True,
    check_points: Optional[Sequence[int]] = None,
    adaptive_params: Optional[AdaptiveParams] = None,
) -> Optional[PyTree[Array]]:
    """Solve an ODE — internal ``custom_inverse``-wrapped implementation.

    ``drift`` is a static argument; traced drift state must reach this
    function through ``args`` (as positional values) or ``kwargs``. See
    :func:`odeint` for the public API.
    """
    if kwargs:
        kw = dict(kwargs)
        bind_args = getattr(drift, "bind_args", None)
        if callable(bind_args):
            # Marker dataclasses (``split_drift``, ``linear_drift``, ...) know
            # how to rebind kwargs without losing their isinstance identity,
            # which specialized solvers rely on.
            drift = bind_args(**kw)
        else:
            original_drift = drift

            def drift_with_kwargs(t: Array, y: PyTree[Array], *a: Any):
                return original_drift(t, y, *a, **kw)

            drift = drift_with_kwargs

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
    **kwargs,
) -> Optional[PyTree[Array]]:
    """Solve an ordinary differential equation.

    This is a high-level interface for solving ODEs using various numerical
    methods. It supports both fixed-step and adaptive-step integration.

    ``drift`` may be any of:

    - a plain Python callable ``drift(t, y, *args, **kwargs)``; any traced
      parameters must be forwarded explicitly via ``args`` / ``kwargs``;
    - a registered JAX pytree node (``equinox.Module``,
      ``flax.struct.PyTreeNode``, ...) that is callable with the same
      signature; its array leaves flow through as regular JAX inputs and
      participate in transformations such as ``jax.jit``, ``jax.grad``, and
      ``jax.vmap`` natively.

    Traced values captured implicitly in a plain drift's closure are no
    longer lifted automatically. Pass them through as explicit arguments
    or wrap them in a pytree-callable instead.

    Args:
        drift: The drift function ``f(t, y, *args, **kwargs)`` defining
            ``dy/dt = f(t, y, *args, **kwargs)``.
        y0: Initial state. Can be a single array or a PyTree of arrays.
        ts: Time points at which to evaluate the solution.
        *args: Additional positional arguments forwarded to ``drift``.
        method: Integration method to use.
        dtype: Data type for computation. Defaults to float32.
        filter_state: Optional function to filter the state during
            integration.
        collect_trace: If ``True`` (default), return the filtered state at
            every requested time point; otherwise only the filtered terminal
            state.
        check_points: Optional sequence of indices for grid integration.
        adaptive_params: Parameters for adaptive integration methods.
        **kwargs: Additional keyword arguments forwarded to ``drift``.

    Returns:
        PyTree containing either the time-series trace (when
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
    # Partition drift kwargs into traced (array-like) and static
    # (Python scalars, strings, bools, ...). Static kwargs are bound into
    # the drift at Python call time — keeping them out of the traced dict
    # means the drift body can still use them for Python control flow
    # (``if mode == "affine": ...``). Traced kwargs flow through a dynamic
    # dict so they participate in ``jax.jit`` / ``jax.grad`` / custom_inverse.
    static_kwargs = {}
    traced_kwargs = {}
    for name, value in kwargs.items():
        if _leaf_is_array_like(value):
            traced_kwargs[name] = value
        else:
            static_kwargs[name] = value

    if static_kwargs:
        bind = getattr(drift, "bind_args", None)
        if callable(bind):
            # Markers (``split_drift``, ``linear_drift``, ...) keep their
            # isinstance identity through ``bind_args``.
            drift = bind(**static_kwargs)
            static_kwargs = {}

    drift_static, drift_leaves = _split_drift_pytree(drift)

    if static_kwargs:
        # Plain or pytree-native drift: wrap the (possibly lifted) callable
        # to inject the static kwargs without sending them through tracing.
        _inner = drift_static
        _static = static_kwargs

        def drift_with_static(
            t: Array, y: PyTree[Array], *dynamic: Any, **extra: Any
        ):
            return _inner(t, y, *dynamic, **_static, **extra)

        drift_static = drift_with_static

    packed_args = drift_leaves + tuple(args)
    return _odeint_custom(
        drift_static,
        y0,
        ts,
        packed_args,
        traced_kwargs,
        method=method,
        dtype=dtype,
        filter_state=filter_state,
        collect_trace=collect_trace,
        check_points=check_points,
        adaptive_params=adaptive_params,
    )


_odeint_custom.definv(_inv_odeint)
_odeint_custom.definv_and_logdet(_inv_logdet_odeint)

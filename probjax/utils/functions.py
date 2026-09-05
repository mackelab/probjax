"""Unified drift API for ``odeint``/``sdeint``.

A drift is any callable ``f(t, y, *args) -> pytree(y)``. Two kinds flow
through our solvers:

* :class:`generic_drift` — wraps a plain Python callable. Registered as a
  pytree with zero array leaves; the callable rides as ``aux_data``.
* Marker drifts (:class:`linear_drift`, :class:`split_drift`,
  :class:`const_diffusion`, :class:`state_drift`, :class:`affine_drift`,
  :class:`additive_diffusion`) — dataclasses whose array-like fields are
  exposed as pytree leaves so they participate in ``jax.jit``, ``jax.grad``,
  ``jax.vmap`` natively, while non-array fields (functions, ``None``, Python
  scalars) ride as ``aux_data``.

The :func:`register_drift` decorator builds the pytree (un)flattener by
introspecting each dataclass field *at flatten time*: array-like values
become children, everything else becomes aux. A per-instance leaf mask in
aux records which slot came from where so reconstruction is faithful.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import jax

from probjax.utils.typing import Array, ArrayLike, PyTree


def _is_array_like(v: Any) -> bool:
    """Duck-type check: is ``v`` an array-like JAX value (including tracers)?"""
    return hasattr(v, "shape") and hasattr(v, "dtype")


def _replace_positional_arg(
    args: tuple[Any, ...], index: int, value: Any
) -> tuple[Any, ...]:
    if index < 0 or index >= len(args):
        raise IndexError(
            f"Argument index {index} is out of range for {len(args)} arguments."
        )
    return args[:index] + (value,) + args[index + 1 :]


def register_drift(cls):
    """Register a dataclass as a JAX pytree with per-instance field partitioning.

    At flatten time each field is classified:

    - array-like values (``hasattr('shape')`` and ``hasattr('dtype')``) become
      pytree children (leaves — participate in transformations);
    - everything else (functions, ``None``, Python scalars, strings) becomes
      ``aux_data`` (static, identified by hash/eq).

    The per-instance leaf mask lives in ``aux_data`` so heterogenous fields
    (e.g. ``linear_drift.A`` being an array *or* a callable depending on the
    instance) reconstruct faithfully.
    """
    field_names = tuple(f.name for f in dataclasses.fields(cls))

    def _flatten(obj):
        children: list[Any] = []
        aux_values: list[Any] = []
        mask: list[bool] = []
        for name in field_names:
            value = getattr(obj, name)
            if _is_array_like(value):
                children.append(value)
                mask.append(True)
            else:
                aux_values.append(value)
                mask.append(False)
        aux = (tuple(mask), tuple(aux_values))
        return tuple(children), aux

    def _unflatten(aux, children):
        mask, aux_values = aux
        children_iter = iter(children)
        aux_iter = iter(aux_values)
        kwargs = {
            name: next(children_iter) if is_leaf else next(aux_iter)
            for name, is_leaf in zip(field_names, mask)
        }
        return cls(**kwargs)

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls


class Drift:
    """Base class for drift/diffusion callables that flow as JAX pytrees.

    Subclasses are dataclasses decorated with :func:`register_drift`. Two
    hooks are exposed to ODE/SDE internals:

    - :meth:`__call__` — evaluates the drift at ``(t, y, *args)``.
    - :meth:`ravel_arg` — returns a variant that accepts a flat-array ``y``
      at position ``index`` (used by solvers that ravel pytree states).
    - :meth:`bind_args` — returns a new instance with positional args bound
      via closure. The default wraps in a :class:`generic_drift`, which
      loses the marker's type; marker subclasses override to preserve it.
    """

    def __call__(self, t: ArrayLike, y: PyTree, *args: Any) -> PyTree:
        raise NotImplementedError

    def ravel_arg(self, unravel: Callable[[Array], PyTree], index: int = 1):
        from probjax.utils.jaxutils import ravel_args

        def drift_raveled(*args):
            args = _replace_positional_arg(args, index, unravel(args[index]))
            value = self(*args)
            value_flat, _ = ravel_args(value)
            return value_flat

        return drift_raveled

    def bind_args(self, *args: Any) -> "Drift":
        """Bind positional ``*args`` into the drift via closure.

        Default wraps in :class:`generic_drift` (loses marker type). Markers
        override to preserve their dataclass identity so specialized solvers
        can still ``isinstance``-dispatch.
        """
        if not args:
            return self

        inner = self

        def bound(t, y):
            return inner(t, y, *args)

        return generic_drift(fn=bound)


@register_drift
@dataclass(frozen=True, eq=False)
class generic_drift(Drift):
    """Wrap a plain Python callable so it flows as a drift pytree.

    The callable is stashed as aux data; instances have zero array leaves.
    This is the automatic wrapping applied by :func:`odeint`/``sdeint`` for
    plain-function drifts so they travel through ``jax.jit`` / the
    ``custom_inverse`` primitive without being treated as static.
    """

    fn: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args: Any) -> PyTree:
        return self.fn(t, y, *args)

    def bind_args(self, *args: Any) -> "generic_drift":
        if not args:
            return self
        inner_fn = self.fn

        def bound(t, y):
            return inner_fn(t, y, *args)

        return generic_drift(fn=bound)


@register_drift
@dataclass(frozen=True, eq=False)
class split_drift(Drift):
    """Marker for drifts of the form ``c(t) * y + N(t, y)``.

    Used by exponential integrators that treat the linear scalar part
    (``c(t) * y``) analytically and the nonlinear part explicitly.
    """

    lin_coeff: Callable[[ArrayLike], Array]
    nonlin: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, x: PyTree, *args: Any) -> PyTree:
        coeff = self.lin_coeff(t)
        linear = jax.tree_util.tree_map(lambda xi: coeff * xi, x)
        nonlin = self.nonlin(t, x, *args)
        return jax.tree_util.tree_map(lambda li, ni: li + ni, linear, nonlin)

    def bind_args(self, *args: Any) -> "split_drift":
        if not args:
            return self
        inner_nonlin = self.nonlin

        def nonlin_bound(t, x):
            return inner_nonlin(t, x, *args)

        return split_drift(lin_coeff=self.lin_coeff, nonlin=nonlin_bound)

    def ravel_arg(
        self, unravel: Callable[[Array], PyTree], index: int = 1
    ) -> "split_drift":
        from probjax.utils.jaxutils import ravel_args

        inner_nonlin = self.nonlin

        def nonlin_raveled(*args):
            args = _replace_positional_arg(args, index, unravel(args[index]))
            value = inner_nonlin(*args)
            value_flat, _ = ravel_args(value)
            return value_flat

        return split_drift(lin_coeff=self.lin_coeff, nonlin=nonlin_raveled)


@register_drift
@dataclass(frozen=True, eq=False)
class state_drift(Drift):
    """Marker for autonomous drifts of the form ``f(y)`` (time-independent)."""

    drift: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args: Any) -> PyTree:
        del t
        return self.drift(y, *args)

    def bind_args(self, *args: Any) -> "state_drift":
        if not args:
            return self
        inner = self.drift

        def bound(y):
            return inner(y, *args)

        return state_drift(drift=bound)


@register_drift
@dataclass(frozen=True, eq=False)
class affine_drift(Drift):
    """Marker for affine drifts of the form ``A(t, y) + b(t)``."""

    linear: Callable[..., PyTree]
    bias: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args: Any) -> PyTree:
        linear_part = self.linear(t, y, *args)
        bias_part = self.bias(t, *args)
        return jax.tree_util.tree_map(lambda li, bi: li + bi, linear_part, bias_part)

    def bind_args(self, *args: Any) -> "affine_drift":
        if not args:
            return self
        inner_linear = self.linear
        inner_bias = self.bias

        def linear_bound(t, y):
            return inner_linear(t, y, *args)

        def bias_bound(t):
            return inner_bias(t, *args)

        return affine_drift(linear=linear_bound, bias=bias_bound)


@register_drift
@dataclass(frozen=True, eq=False)
class additive_diffusion(Drift):
    """Marker for state-independent diffusions of the form ``g(t)``.

    Used by SDE solvers to skip unnecessary re-evaluation of the diffusion
    when it doesn't depend on ``y``.
    """

    diffusion: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args: Any) -> PyTree:
        del y
        return self.diffusion(t, *args)

    def bind_args(self, *args: Any) -> "additive_diffusion":
        if not args:
            return self
        inner = self.diffusion

        def bound(t):
            return inner(t, *args)

        return additive_diffusion(diffusion=bound)


@register_drift
@dataclass(frozen=True, eq=False)
class linear_drift(Drift):
    """Marker for linear drift ``dy/dt = A(t) y + b(t)``.

    If ``A`` is an array the matrix exponential can be computed once; if
    ``A`` is a callable it's evaluated at each step. Flows as a pytree: an
    array ``A`` becomes a traced leaf; a callable ``A`` or ``b`` rides as
    aux data.
    """

    A: Union[Array, Callable[[ArrayLike], Array]]
    b: Optional[Callable[..., Array]] = None

    def __call__(self, t: ArrayLike, y: PyTree, *args: Any) -> PyTree:
        import jax.numpy as jnp

        if callable(self.A):
            A_t = self.A(t)
        else:
            A_t = self.A

        A_concrete = jnp.asarray(A_t)
        if A_concrete.ndim == 0:
            linear_part = jax.tree_util.tree_map(lambda yi: A_t * yi, y)
        else:
            linear_part = jax.tree_util.tree_map(lambda yi: A_t @ yi, y)

        if self.b is None:
            return linear_part
        bias_part = self.b(t, *args)
        return jax.tree_util.tree_map(lambda li, bi: li + bi, linear_part, bias_part)

    def bind_args(self, *args: Any) -> "linear_drift":
        if self.b is None or not args:
            return self
        b_fn = self.b

        def b_bound(t):
            return b_fn(t, *args)

        return linear_drift(A=self.A, b=b_bound)

    def ravel_arg(
        self, unravel: Callable[[Array], PyTree], index: int = 1
    ) -> "linear_drift":
        # ``linear_drift`` already operates on a single flat state — the
        # solver's ravel machinery is a no-op for it.
        del unravel, index
        return self

    @property
    def is_constant(self) -> bool:
        return not callable(self.A)


@register_drift
@dataclass(frozen=True, eq=False)
class const_diffusion(Drift):
    """Marker for constant diffusion ``g(y) = G`` (matrix or vector)."""

    G: Array

    def __call__(self, t: ArrayLike, y: PyTree, *args: Any) -> PyTree:
        del t, y, args
        return self.G

    def bind_args(self, *args: Any) -> "const_diffusion":
        del args
        return self

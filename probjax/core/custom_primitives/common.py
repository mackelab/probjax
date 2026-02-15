from __future__ import annotations

from typing import Any, Callable, Generic, Sequence, TypeVar

from jax._src import core as jax_core
from jax._src import linear_util as lu
from jax._src.api_util import debug_info, flatten_fun_nokwargs
from jax._src.interpreters import batching as batching_src
from jax._src.interpreters import partial_eval as pe
from jax.interpreters import batching
from jax.extend.core import ClosedJaxpr
from jax.tree_util import tree_leaves

T = TypeVar("T")


class Lazy(Generic[T]):
    """Lazy thunk that caches its result after first evaluation.

    This avoids repeated expensive tracing by computing the value only once
    and memoizing it for subsequent accesses.
    """

    __slots__ = ("_thunk", "_value", "_evaluated", "__weakref__")

    def __init__(self, thunk: Callable[[], T]) -> None:
        self._thunk = thunk
        self._value: T | None = None
        self._evaluated = False

    def get(self) -> T:
        """Return the lazily computed value, evaluating the thunk if needed."""
        if not self._evaluated:
            self._value = self._thunk()
            self._evaluated = True
        return self._value  # type: ignore[return-value]

    def __call__(self) -> T:
        """Allow using Lazy as a callable thunk for backwards compatibility."""
        return self.get()

    @property
    def is_evaluated(self) -> bool:
        """Check if the thunk has been evaluated."""
        return self._evaluated

    def map(self, fn: Callable[[T], T]) -> "Lazy[T]":
        """Return a new Lazy that applies fn to this Lazy's value."""
        return Lazy(lambda: fn(self.get()))


class LazyClosedJaxpr(Lazy[ClosedJaxpr]):
    """Lazy wrapper specifically for ClosedJaxpr to avoid eager tracing."""

    __slots__ = ()

    def __init__(self, thunk: Callable[[], ClosedJaxpr]) -> None:
        super().__init__(thunk)


def has_tracer(tree) -> bool:
    return any(isinstance(x, jax_core.Tracer) for x in tree_leaves(tree))


def ensure_hashable(x, where: str):
    try:
        hash(x)
    except TypeError as e:
        raise TypeError(f"{where} must be hashable; got {type(x)}") from e
    return x


def fail_on_tracer_constants(consts, where: str):
    if has_tracer(consts):
        raise TypeError(
            f"{where} closed over traced JAX values. "
            "Pass such data as dynamic arguments or mark them static before "
            "registering the custom primitive."
        )


def trace_to_closed_jaxpr(
    fun: Callable[..., Any],
    *,
    in_tree,
    in_avals,
    debug_name: str,
    const_context: str,
) -> tuple[ClosedJaxpr, tuple[Any, ...], Any]:
    info = debug_info(debug_name, fun, (), {})
    wrapped = lu.wrap_init(fun, debug_info=info)
    flat_fun, out_tree_thunk = flatten_fun_nokwargs(wrapped, in_tree)

    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    fail_on_tracer_constants(consts, const_context)
    return ClosedJaxpr(jaxpr, consts), tuple(out_avals), out_tree_thunk()


def normalize_axis_data(axis_data):
    if isinstance(axis_data, batching_src.AxisData):
        return axis_data
    return batching_src.AxisData(
        name="batch",
        size=axis_data,
        spmd_name=None,
        _ema=None,
    )


def batch_closed_jaxpr(
    closed_jaxpr: ClosedJaxpr,
    axis_data,
    in_axes: Sequence[int | None],
):
    axis_data = normalize_axis_data(axis_data)
    return batching_src.batch_jaxpr2(closed_jaxpr, axis_data, tuple(in_axes))


def move_mapped_axes_to_front(args, in_dims):
    import jax.numpy as jnp

    new_args = []
    in_axes = []
    for x, d in zip(args, in_dims, strict=False):
        if d is batching.not_mapped:
            new_args.append(x)
            in_axes.append(batching.not_mapped)
        else:
            new_args.append(jnp.moveaxis(x, d, 0) if d != 0 else x)
            in_axes.append(0)

    any_batched = any(d is not batching.not_mapped for d in in_axes)
    return new_args, in_axes, any_batched

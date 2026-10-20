# minjax_types.py
# Minimal typing helpers for JAX PyTrees and RNG keys (typed & legacy).
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    runtime_checkable,
)

import jax
import jax.numpy as jnp

__all__ = [
    "Array",
    "ArrayLike",
    "ArrayTree",
    "ArrayLikeTree",
    "DTypeLike",
    "PrecisionLike",
    "Scalar",
    "ScalarLike",
    "Shape",
    "RngKey",
    "RngKeyLike",
    "PyTree",
    "Leaf",
    "AnyTree",
    "SDEIntResult",
    "is_typed_key",
    "is_legacy_key",
    "is_key",
    "assert_key",
    # Modules
    "ModuleLike",
    "ModuleLikeType",
    # Standard typing
    "Callable",
    "Iterable",
    "Mapping",
    "Sequence",
    "Union",
    "Device",
]

# ---------------- Core aliases ----------------
Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = jax.typing.ArrayLike
DTypeLike: TypeAlias = jax.typing.DTypeLike
PrecisionLike: TypeAlias = Union[
    None,
    str,
    jax.lax.Precision,
    tuple[str, str],
    tuple[jax.lax.Precision, jax.lax.Precision],
]
if TYPE_CHECKING:
    # ``jax.Device`` is a nanobind class that type checkers resolve to a plain
    # variable through the public alias, so point them at the defining module.
    from jaxlib._jax import Device
else:
    Device = jax.Device

# Scalars: a Python number or a 0-d JAX array. Traced JAX scalars are
# ``Array``s, so anything that may be staged under ``jit`` should be typed
# as ``Scalar`` rather than ``float``.
Scalar: TypeAlias = float | int | Array
ScalarLike: TypeAlias = float | int | ArrayLike
Shape: TypeAlias = tuple[int, ...] | Sequence[int]

# Keys: support both typed-key dtype and legacy uint32[..., key_bits]
RngKey: TypeAlias = Array  # use runtime guards below to distinguish
RngKeyLike: TypeAlias = int | RngKey  # seed int or any key array

# Discover key bit-size (public API, robust to impl changes)
_KEY_BITS_SHAPE = jax.random.key_data(jax.random.key(0)).shape[-1]  # usually 2

# ---------------- PyTree typing ----------------
T = TypeVar("T")
#: A pytree whose leaves have type ``T``.
#:
#: The container branches are deliberately loose: a recursive generic alias
#: (the textbook ``tuple["PyTree[T]", ...] | ...``) cannot be resolved by
#: type checkers, which then rejects every pytree at the call site. Naming
#: the leaf type and keeping the containers permissive keeps the alias usable.
PyTree: TypeAlias = Union[T, None, tuple[Any, ...], list[Any], Mapping[Any, Any]]
Leaf: TypeAlias = Array | float | int | bool
AnyTree: TypeAlias = PyTree[Leaf]
ArrayTree: TypeAlias = PyTree[Array]
ArrayLikeTree: TypeAlias = PyTree[ArrayLike]


SDEIntResult: TypeAlias = Union[
    PyTree[Array],
    tuple[PyTree[Array], PyTree[Array]],
    tuple[Any, PyTree[Array], PyTree[Array]],
    tuple[Any, PyTree[Array], PyTree[Array], Array],
]


def is_typed_key(x: Any) -> bool:
    """True if x has a PRNG key dtype (typed keys made by jax.random.key)."""
    return isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key)


def is_legacy_key(x: Any) -> bool:
    """True if x is a legacy key: uint32 with trailing key-bits dimension."""
    return (
        isinstance(x, jax.Array)
        and x.dtype == jnp.uint32
        and x.ndim >= 1
        and x.shape[-1] == _KEY_BITS_SHAPE
    )


def is_key(x: Any) -> TypeGuard[RngKey]:
    """Type guard for either typed or legacy keys."""
    return is_typed_key(x) or is_legacy_key(x)


def assert_key(x: Array) -> RngKey:
    """Assert x is a valid key of either form; return it typed as Key."""
    if not is_key(x):
        raise ValueError(
            f"Expected PRNG key; got shape={tuple(x.shape)}, dtype={x.dtype}"
        )
    return x


# ---------------- Module typing ----------------
@runtime_checkable
class ModuleLike(Protocol):
    """Structural protocol for an nnx-like Module instance.

    Kept minimal on purpose for Pyright compatibility; describes only the
    callable interface we rely on at runtime.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


# Constructors for modules: return a callable ModuleLike instance.
# Using ModuleLike here helps pyright understand instances are callable.
ModuleLikeType: TypeAlias = Callable[..., ModuleLike]

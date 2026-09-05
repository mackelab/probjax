"""Shape-polymorphic JAX exports bound to mutable NNX models."""

from __future__ import annotations

import weakref
from typing import Callable, TypeVar

import jax
import jax.numpy as jnp
from flax import nnx
from jax import export as jax_export

from probjax.utils.typing import PyTree

_EXPORT_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_CachedExport = TypeVar("_CachedExport", bound="NNXExportedFunction")


def _is_shape(value) -> bool:
    return isinstance(value, tuple) and all(isinstance(dim, int) for dim in value)


def normalize_spec(value, dtype=jnp.float32) -> PyTree[jax.ShapeDtypeStruct]:
    """Normalize a shape, shape/dtype struct, or pytree of either."""

    def is_leaf(node):
        return isinstance(node, jax.ShapeDtypeStruct) or _is_shape(node)

    def to_struct(node):
        if isinstance(node, jax.ShapeDtypeStruct):
            return node
        return jax.ShapeDtypeStruct(tuple(int(dim) for dim in node), jnp.dtype(dtype))

    return jax.tree.map(to_struct, value, is_leaf=is_leaf)


def spec_from_value(value) -> PyTree[jax.ShapeDtypeStruct]:
    """Build a shape/dtype spec matching a concrete pytree value."""
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(
            jnp.asarray(leaf).shape,
            jnp.asarray(leaf).dtype,
        ),
        value,
    )


def spec_cache_key(spec) -> tuple:
    """Return a hashable cache-key component for a spec."""
    if spec is None:
        return (None,)
    leaves, treedef = jax.tree.flatten(spec)
    return (treedef, tuple((leaf.shape, jnp.dtype(leaf.dtype).str) for leaf in leaves))


def zeros_from_spec(spec):
    """Create a one-item example batch matching a spec."""
    return jax.tree.map(lambda leaf: jnp.zeros((1,) + leaf.shape, leaf.dtype), spec)


def flatten_spec_batch(value, spec, *, name: str):
    """Validate a pytree and merge all of its leading batch axes."""
    spec_leaves, treedef = jax.tree.flatten(spec)
    try:
        value_leaves = treedef.flatten_up_to(value)
    except (ValueError, TypeError) as err:
        raise ValueError(
            f"{name} does not match the expected pytree structure {treedef}."
        ) from err

    batch_shape = None
    flat_leaves = []
    for leaf, spec_leaf in zip(value_leaves, spec_leaves, strict=True):
        leaf = jnp.asarray(leaf, dtype=spec_leaf.dtype)
        event_ndim = len(spec_leaf.shape)
        if event_ndim and (
            leaf.ndim < event_ndim
            or leaf.shape[leaf.ndim - event_ndim :] != spec_leaf.shape
        ):
            raise ValueError(
                f"Expected trailing shape {spec_leaf.shape} for {name}, "
                f"got {leaf.shape}."
            )
        leaf_batch = leaf.shape[: leaf.ndim - event_ndim] if event_ndim else leaf.shape
        if batch_shape is None:
            batch_shape = leaf_batch
        elif leaf_batch != batch_shape:
            raise ValueError(
                f"Inconsistent batch shapes across {name} leaves: "
                f"{batch_shape} vs {leaf_batch}."
            )
        flat_leaves.append(leaf.reshape((-1,) + spec_leaf.shape))
    return batch_shape, treedef.unflatten(flat_leaves)


def flatten_broadcast_batch(value, batch_shape, spec, *, name: str):
    """Flatten a pytree, broadcasting an unbatched value to ``batch_shape``."""
    if spec is None:
        if value is not None:
            raise ValueError(f"This operation was built without {name}.")
        return None
    if value is None:
        raise ValueError(f"{name} is required for this operation.")

    spec_leaves, treedef = jax.tree.flatten(spec)
    try:
        value_leaves = treedef.flatten_up_to(value)
    except (ValueError, TypeError) as err:
        raise ValueError(
            f"{name} does not match the expected pytree structure {treedef}."
        ) from err

    flat_leaves = []
    for leaf, spec_leaf in zip(value_leaves, spec_leaves, strict=True):
        leaf = jnp.asarray(leaf, dtype=spec_leaf.dtype)
        expected_shape = batch_shape + spec_leaf.shape
        if leaf.shape == spec_leaf.shape:
            leaf = jnp.broadcast_to(leaf, expected_shape)
        elif leaf.shape != expected_shape:
            raise ValueError(
                f"Expected {name} shape {spec_leaf.shape} or {expected_shape}, "
                f"got {leaf.shape}."
            )
        flat_leaves.append(leaf.reshape((-1,) + spec_leaf.shape))
    return treedef.unflatten(flat_leaves)


def restore_batch(output, batch_shape):
    """Restore merged leading batch axes on every output leaf."""
    return jax.tree.map(
        lambda leaf: leaf.reshape(batch_shape + leaf.shape[1:]),
        output,
    )


def export_symbolic_batch(
    fn: Callable,
    args: tuple,
    axis_specs: tuple,
    *,
    constraints: tuple[str, ...] = ("b >= 1",),
) -> jax_export.Exported:
    """JIT and export a function with symbolic argument-axis specifications."""
    specs = jax_export.symbolic_args_specs(
        args,
        axis_specs,
        constraints=constraints,
    )
    return jax_export.export(jax.jit(fn))(*specs)


class NNXExportedFunction:
    """An exported function bound weakly to a mutable NNX model."""

    def __init__(self, model: nnx.Module, graphdef: nnx.GraphDef, exported) -> None:
        self._model_ref = weakref.ref(model)
        self._graphdef = graphdef
        self._invalidated = False
        self.exported = exported

    @property
    def invalidated(self) -> bool:
        return self._invalidated

    def invalidate(self) -> None:
        self._invalidated = True

    def model_and_state(self):
        if self._invalidated:
            raise RuntimeError("This compiled operation has been invalidated.")
        model = self._model_ref()
        if model is None:
            raise RuntimeError("The model used to build this export no longer exists.")
        graphdef, state = nnx.split(model)
        if graphdef != self._graphdef:
            raise RuntimeError(
                "The model structure changed after this export was built. "
                "Build it again."
            )
        return model, state


def get_cached_export(
    model: nnx.Module,
    key: tuple,
    build: Callable[[], _CachedExport],
) -> _CachedExport:
    """Return a per-model compiled operation, building it at most once."""
    model_cache = _EXPORT_CACHE.setdefault(model, {})
    operation = model_cache.get(key)
    if operation is not None:
        try:
            operation.model_and_state()
        except RuntimeError:
            operation = None
    if operation is None:
        operation = build()
        model_cache[key] = operation
    return operation


def clear_export_cache(model: nnx.Module) -> None:
    """Invalidate every compiled operation associated with a model."""
    operations = _EXPORT_CACHE.pop(model, {}).values()
    for operation in operations:
        operation.invalidate()

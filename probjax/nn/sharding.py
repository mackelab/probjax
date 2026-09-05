"""Flax-native sharding helpers (:mod:`probjax.nn.sharding`).

probjax modules annotate their parameters with *logical axis names* and rely
on flax's eager sharding (FLIP 4844): construct the model under an ambient
mesh and the parameters are sharded at creation. No configuration objects,
no constructor arguments.

Usage::

    from jax.sharding import AxisType

    # Auto axis types are recommended for model parallelism: GSPMD then
    # propagates shardings through contractions automatically.
    mesh = jax.make_mesh((2, 4), ("data", "model"),
                         axis_types=(AxisType.Auto, AxisType.Auto))

    with jax.set_mesh(mesh):
        model = Transformer(...)          # params sharded eagerly
        batch = jax.device_put(batch, NamedSharding(mesh, P("data")))
        model.fit(rng, batch)             # jit propagates shardings

    # Custom logical->mesh mapping (defaults to default_rules()):
    with jax.set_mesh(mesh), nnx.logical_axis_rules(default_rules()):
        ...

Logical names are resolved to physical mesh axes at *construction time* via
the active :func:`flax.nnx.logical_axis_rules` (or :func:`default_rules`),
so unmapped names simply degrade to replicated, and
``nnx.get_partition_spec(nnx.state(model, nnx.Param))`` yields physical
specs directly — e.g. for sharding optimizer state.

Without an active mesh everything here is a no-op: ``param_metadata``
returns ``{}`` (variables carry no sharding annotation, avoiding flax's
always-shard error on single devices) and ``constrain`` returns its input.

Note: combining head-sharded attention parameters with the pallas attention
kernels is unvalidated; use the default (non-pallas) attention path under
model parallelism.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
from flax.core import spmd as _flax_spmd
from jax.sharding import AxisType, PartitionSpec

__all__ = [
    "BATCH",
    "SEQ",
    "EMBED",
    "HIDDEN",
    "HEADS",
    "HEAD_DIM",
    "default_rules",
    "resolve_axes",
    "param_metadata",
    "constrain",
    "replicate",
]

# Canonical logical axis names.
BATCH = "batch"  # leading batch axis of activations
SEQ = "seq"  # sequence axis (never sharded by default)
EMBED = "embed"  # residual / feature axis (replicated by default)
HIDDEN = "hidden"  # MLP hidden feature axis        -> model axis
HEADS = "heads"  # attention heads axis            -> model axis
HEAD_DIM = "head_dim"  # per-head feature axis (replicated)


def default_rules(
    data_axis: str = "data", model_axis: str = "model"
) -> Tuple[Tuple[str, Optional[str]], ...]:
    """Logical-to-mesh-axis rules for a standard ``("data", "model")`` mesh."""
    return (
        (BATCH, data_axis),
        (HIDDEN, model_axis),
        (HEADS, model_axis),
        (EMBED, None),
        (HEAD_DIM, None),
        (SEQ, None),
    )


def _active_mesh():
    mesh = jax.sharding.get_abstract_mesh()
    return None if mesh.empty else mesh


def resolve_axes(*axes: Optional[str]) -> Optional[Tuple[Optional[str], ...]]:
    """Map logical axis names to physical mesh axes.

    Uses the active :func:`flax.nnx.logical_axis_rules` when set, else
    :func:`default_rules`. A name resolves to its mapped mesh axis only if
    that axis exists in the active mesh and has size > 1. Returns ``None``
    when no mesh is active or nothing resolves.
    """
    from probjax.core.custom_primitives.sharded_primitive import in_per_shard_body

    mesh = _active_mesh()
    if mesh is None or in_per_shard_body():
        return None

    rules = dict(_flax_spmd.get_logical_axis_rules() or default_rules())
    resolved = tuple(
        mesh_axis
        if (mesh_axis := rules.get(name)) is not None
        and mesh.shape.get(mesh_axis, 1) > 1
        else None
        for name in axes
    )
    if all(axis is None for axis in resolved):
        return None
    return resolved


def param_metadata(*axes: Optional[str]) -> dict:
    """Sharding metadata for nnx layers (``kernel_metadata=`` and friends).

    Returns ``{"sharding": <physical axes>}``, or ``{}`` when no mesh is
    active / nothing resolves — so single-device construction carries no
    annotation and costs nothing.
    """
    resolved = resolve_axes(*axes)
    if resolved is None:
        return {}
    return {"sharding": resolved}


def _is_vmap_batched(x) -> bool:
    """True when *x* carries an outer vmap batch dim.

    Constraints written for the unbatched call signature would land on the
    wrong dimensions after vmap moves axes, so constrain/replicate no-op.
    """
    from jax._src.interpreters import batching as _batching

    return isinstance(x, _batching.BatchTracer)


def constrain(x: jax.Array, *axes: Optional[str]) -> jax.Array:
    """Constrain activation ``x`` to the resolved ``PartitionSpec``.

    Trailing unmentioned dimensions are left unconstrained. No-op when no
    mesh is active, nothing resolves, or ``x`` is batched by an outer vmap
    (the axes would be misaligned); all other errors propagate.
    """
    if _is_vmap_batched(x):
        return x
    resolved = resolve_axes(*axes)
    if resolved is None:
        return x
    return _apply_spec(x, PartitionSpec(*resolved))


def replicate(x: jax.Array) -> jax.Array:
    """Constrain ``x`` to be fully replicated (gathered on every device).

    For modules whose implementation requires unsharded inputs (e.g. the
    LRU scan and spatial self-attention). No-op without an active mesh.
    """
    from probjax.core.custom_primitives.sharded_primitive import in_per_shard_body

    if _active_mesh() is None or in_per_shard_body() or _is_vmap_batched(x):
        return x
    return _apply_spec(x, PartitionSpec(*(None,) * x.ndim))


def _apply_spec(x: jax.Array, spec: PartitionSpec) -> jax.Array:
    """Apply a physical PartitionSpec under the active mesh.

    Explicit-axes meshes require ``jax.sharding.reshard``; Auto axes use
    ``jax.lax.with_sharding_constraint`` (mirrors flax.core.spmd handling).
    Under Manual axes (inside shard_map / custom_partitioning per-shard
    bodies) constraints are meaningless — the code already runs per-shard —
    so this is a no-op there.
    """
    mesh = jax.sharding.get_abstract_mesh()
    axis_types = set(mesh.axis_types)
    if AxisType.Manual in axis_types:
        return x
    if axis_types == {AxisType.Explicit}:
        return jax.sharding.reshard(x, spec)
    if AxisType.Explicit in axis_types:
        raise ValueError(
            "Meshes mixing Explicit and Auto axis types are not "
            f"supported by probjax.nn.sharding.constrain; got {mesh.axis_types}."
        )
    return jax.lax.with_sharding_constraint(x, spec)

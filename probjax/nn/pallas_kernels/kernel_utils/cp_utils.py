"""Utilities for custom_partitioning-wrapped Pallas attention kernels.

This module contains:
- Sentinel helpers: substitutes for None arrays required by custom_partitioning.
- Sharding rule builders for forward, JVP, and backward MHA passes.
- A generic CP wrapper factory (make_cp_function) that eliminates boilerplate.
- A batching rule for ``custom_partitioning_p`` that merges the ``vmap``
  dimension into the leading (batch) dimension so CP-wrapped kernels can
  be used directly under ``jax.vmap``.
- try_cp_or_raw: legacy "try custom_partitioning, fall back to raw" pattern.
  Kept as a safety net but should rarely trigger now that the batching rule
  is registered.
- _validate_mha_sharding: sharding validation for q/k/v operands.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Sentinel helpers
# ---------------------------------------------------------------------------
# custom_partitioning requires all positional args to be concrete arrays.
# Optional arrays (bias, mask ids, dropout masks, iterator offsets) may be
# None.  We substitute a scalar zero sentinel and unwrap on the other side.


def sentinel() -> jax.Array:
    """Dummy scalar placeholder for a ``None`` array arg."""
    return jnp.zeros((), jnp.float32)


def or_sentinel(x: Any) -> jax.Array:
    """Return *x* as-is if it is an array, otherwise a scalar sentinel."""
    return x if x is not None else sentinel()


def undo_sentinel(x: Any, present: bool) -> Any:
    """Return *x* if *present*, otherwise ``None``."""
    return x if present else None


# ---------------------------------------------------------------------------
# Sharding validation
# ---------------------------------------------------------------------------


def validate_mha_sharding(sharding: Any, name: str) -> None:
    """Validate that a NamedSharding on an MHA operand is supported.

    For q/k/v shaped ``(B, T, H, D)`` only sharding on the batch (dim 0) and
    heads (dim 2) dimensions is allowed.  Sharding on sequence (dim 1) or
    head_dim (dim 3) raises a ``ValueError``.

    Called inside the ``partition()`` callback of ``custom_partitioning``
    where the sharding object is always available.
    """
    spec = getattr(sharding, "spec", None)
    if spec is None or len(spec) == 0:
        return
    for dim_idx, axis in enumerate(spec):
        if axis is None:
            continue
        if dim_idx == 1:
            raise ValueError(
                f"Pallas flash-attention does not support sharding on the "
                f"sequence dimension (dim 1) of '{name}'. "
                f"Got PartitionSpec{tuple(spec)} which shards dim 1 over "
                f"mesh axis '{axis}'. "
                f"Only batch (dim 0) and heads (dim 2) sharding are supported."
            )
        if dim_idx == 3:
            raise ValueError(
                f"Pallas flash-attention does not support sharding on the "
                f"head_dim dimension (dim 3) of '{name}'. "
                f"Got PartitionSpec{tuple(spec)} which shards dim 3 over "
                f"mesh axis '{axis}'. "
                f"Only batch (dim 0) and heads (dim 2) sharding are supported."
            )


# ---------------------------------------------------------------------------
# Fallback error classification
# ---------------------------------------------------------------------------

_CP_BATCHING_MSG = "Batching rule for 'custom_partitioning' not implemented"


def _is_cp_batching_error(err: NotImplementedError) -> bool:
    """Return True if *err* is the known CP batching-rule gap."""
    return _CP_BATCHING_MSG in str(err)


# ---------------------------------------------------------------------------
# Sharding rule builders
# ---------------------------------------------------------------------------

# Unique replicated factor names per extracted-array slot.  Every dimension
# of every data array is replicated so that the GSPMD partitioner never
# shards them.
_FWD_DATA_RULES: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "b_data": (4, "eb0 eb1 eb2 eb3", ("eb0", "eb1", "eb2", "eb3")),
    "q_id": (4, "eq0 eq1 eq2 eq3", ("eq0", "eq1", "eq2", "eq3")),
    "k_id": (4, "ek0 ek1 ek2 ek3", ("ek0", "ek1", "ek2", "ek3")),
    "dropout_mask": (4, "ed0 ed1 ed2 ed3", ("ed0", "ed1", "ed2", "ed3")),
    "index_offset": (2, "eo0 eo1", ("eo0", "eo1")),
    "index_offset_size": (1, "es0", ("es0",)),
}

_BWD_DATA_RULES: dict[str, tuple[int, str, tuple[str, ...]]] = {
    "b_data": (4, "eb0 eb1 eb2 eb3", ("eb0", "eb1", "eb2", "eb3")),
    "q_data": (4, "eq0 eq1 eq2 eq3", ("eq0", "eq1", "eq2", "eq3")),
    "k_data": (4, "ek0 ek1 ek2 ek3", ("ek0", "ek1", "ek2", "ek3")),
    "dropout_mask": (4, "ed0 ed1 ed2 ed3", ("ed0", "ed1", "ed2", "ed3")),
    "q_index_offset": (2, "eo0 eo1", ("eo0", "eo1")),
    "q_index_offset_size": (1, "es0", ("es0",)),
    "kv_index_offset": (2, "ko0 ko1", ("ko0", "ko1")),
    "kv_index_offset_size": (1, "ks0", ("ks0",)),
}


def _data_rule_parts(
    data_rules: dict[str, tuple[int, str, tuple[str, ...]]],
    present_flags: dict[str, bool],
) -> tuple[list[str], list[str]]:
    """Return (rule_fragments, replication_factors) for optional data arrays.

    For each slot: if present, use its multi-dim rule and factors;
    if absent (sentinel scalar), emit ``""`` and no factors.
    """
    parts: list[str] = []
    factors: list[str] = []
    for name, (_ndim, rule, facs) in data_rules.items():
        if present_flags.get(name, False):
            parts.append(rule)
            factors.extend(facs)
        else:
            parts.append("")  # scalar sentinel → no dimensions
    return parts, factors


def build_mha_sharding_rule_fwd(
    present_flags: dict[str, bool],
    *,
    output_activations: bool,
) -> tuple[str, tuple[str, ...]]:
    """Sharding rule for the CP-wrapped forward pass.

    Positional args: q(B,T,H,D) k v rng_seed
        b_data q_id k_id dropout_mask index_offset index_offset_size
    """
    parts = [
        "batch seq heads head_dim",  # q
        "batch seq heads head_dim",  # k
        "batch seq heads head_dim",  # v
        "",  # rng_seed scalar
    ]
    data_parts, data_factors = _data_rule_parts(_FWD_DATA_RULES, present_flags)
    parts.extend(data_parts)
    out_rule = (
        "batch seq heads head_dim, batch heads seq"
        if output_activations
        else "batch seq heads head_dim"
    )
    rule = ", ".join(parts) + " -> " + out_rule
    return rule, tuple(["seq", "head_dim"] + data_factors)


def build_mha_sharding_rule_jvp(
    present_flags: dict[str, bool],
) -> tuple[str, tuple[str, ...]]:
    """Sharding rule for the CP-wrapped JVP-from-LSE pass.

    Positional args: q k v dq dk dv lse rng_seed
        b_data q_id k_id dropout_mask index_offset index_offset_size
    """
    parts = [
        "batch seq heads head_dim",  # q
        "batch seq heads head_dim",  # k
        "batch seq heads head_dim",  # v
        "batch seq heads head_dim",  # dq
        "batch seq heads head_dim",  # dk
        "batch seq heads head_dim",  # dv
        "batch heads seq",  # lse
        "",  # rng_seed scalar
    ]
    data_parts, data_factors = _data_rule_parts(_FWD_DATA_RULES, present_flags)
    parts.extend(data_parts)
    rule = ", ".join(parts) + " -> batch seq heads head_dim"
    return rule, tuple(["seq", "head_dim"] + data_factors)


def build_mha_sharding_rule_bwd(
    present_flags: dict[str, bool],
) -> tuple[str, tuple[str, ...]]:
    """Sharding rule for the CP-wrapped backward pass.

    Positional args: do q k v rng out lse
        b_data q_data k_data dropout_mask
        q_index_offset q_index_offset_size kv_index_offset kv_index_offset_size
    """
    parts = [
        "batch seq heads head_dim",  # do
        "batch seq heads head_dim",  # q
        "batch seq heads head_dim",  # k
        "batch seq heads head_dim",  # v
        "",  # rng scalar
        "batch seq heads head_dim",  # out
        "batch heads seq",  # lse
    ]
    data_parts, data_factors = _data_rule_parts(_BWD_DATA_RULES, present_flags)
    parts.extend(data_parts)
    rule = (
        ", ".join(parts) + " -> batch seq heads head_dim, "
        "batch seq heads head_dim, "
        "batch seq heads head_dim"
    )
    return rule, tuple(["seq", "head_dim"] + data_factors)


# ---------------------------------------------------------------------------
# Generic custom_partitioning wrapper factory
# ---------------------------------------------------------------------------


def make_cp_function(
    raw_fn: Callable,
    present_flags: dict[str, bool],
    sharding_rule: str,
    replication_factors: tuple[str, ...],
    *,
    # Optional hook called inside _partition to validate arg shardings.
    # Receives (flat_arg_shapes) before the lower_fn is returned.
    partition_validation_fn: Callable | None = None,
) -> Callable:
    """Wrap *raw_fn* with ``custom_partitioning``.

    *raw_fn* must accept all optional arrays as positional arguments, where
    absent arrays are represented by ``None``.  ``make_cp_function`` takes care
    of converting ``None`` → sentinel at the call site and sentinel → ``None``
    inside the wrapper before forwarding to *raw_fn*.

    The returned function has the same positional signature as *raw_fn* except
    that ``None`` args must be replaced by ``or_sentinel(arr)`` at every call
    site.

    Args:
        raw_fn: The underlying implementation (e.g. ``_mha_impl_raw``).
        present_flags: ``{arg_name: bool}`` mapping each optional positional
            arg to whether it is a real array.  The order must match the order
            of optional args in *raw_fn*'s signature.
        sharding_rule: XLA sharding rule string for ``def_partition``.
        replication_factors: Replication factor names for ``def_partition``.
        partition_validation_fn: Optional ``fn(flat_arg_shapes)`` called at
            partition time to validate sharding constraints.
    """
    from jax.experimental.custom_partitioning import custom_partitioning

    # Build the list of (flag_value,) in declaration order so we can undo
    # sentinels positionally.
    flag_values = list(present_flags.values())

    def _unwrap_and_call(*args):
        # Split into "always-present" prefix + "maybe-sentinel" suffix.
        # We rely on raw_fn being called with *all* positional args, where
        # the optional ones are at the end.  The caller must pass them in the
        # same order as present_flags.
        n_fixed = len(args) - len(flag_values)
        fixed = args[:n_fixed]
        optional = tuple(
            undo_sentinel(a, flag) for a, flag in zip(args[n_fixed:], flag_values)
        )
        return raw_fn(*fixed, *optional)

    @custom_partitioning
    def _cp_fn(*args):
        return _unwrap_and_call(*args)

    def _partition(mesh, arg_shapes, result_shape):
        if partition_validation_fn is not None:
            partition_validation_fn(jax.tree.leaves(arg_shapes))
        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(*args):
            return _unwrap_and_call(*args)

        return mesh, lower_fn, result_shardings, arg_shardings

    # Attach the raw callable so the batching rule can re-invoke it
    # with merged shapes (pallas_call jaxprs have baked-in shapes and
    # cannot be evaluated at different sizes via eval_jaxpr).
    _partition._cp_raw_fn = _unwrap_and_call  # type: ignore[attr-defined]

    _cp_fn.def_partition(
        partition=_partition,
        sharding_rule=sharding_rule,
        need_replication_factors=replication_factors,
    )
    return _cp_fn


# ---------------------------------------------------------------------------
# try_cp_or_raw
# ---------------------------------------------------------------------------


def try_cp_or_raw(
    cp_fn: Callable,
    cp_args: tuple,
    raw_fn: Callable,
    raw_args: tuple,
    raw_kwargs: dict,
) -> Any:
    """Call *cp_fn* with *cp_args*; fall back to *raw_fn* on CP errors.

    Falls back to *raw_fn* when ``custom_partitioning`` raises the known
    ``NotImplementedError`` for the missing batching rule (triggered under
    ``vmap``).  All other exceptions propagate normally.

    .. note::

       With the ``custom_partitioning_p`` batching rule registered below,
       this fallback should rarely trigger.  It is kept as a safety net.

    Args:
        cp_fn: custom_partitioning-wrapped function.
        cp_args: Positional args for *cp_fn* (optional arrays as sentinels).
        raw_fn: Bare implementation to call on fallback.
        raw_args: Positional args for *raw_fn* (optional arrays as ``None``).
        raw_kwargs: Keyword args for *raw_fn*.
    """
    try:
        return cp_fn(*cp_args)
    except NotImplementedError as err:
        if not _is_cp_batching_error(err):
            raise
    return raw_fn(*raw_args, **raw_kwargs)


# ---------------------------------------------------------------------------
# Batching rule for custom_partitioning_p
# ---------------------------------------------------------------------------
# JAX's ``custom_partitioning`` does not ship with a ``vmap`` batching rule.
# When a CP-wrapped function is called under ``jax.vmap``, the
# ``BatchTrace.process_primitive`` path raises ``NotImplementedError``.
#
# We register a global batching rule that handles this by *merging* the vmap
# dimension into the leading (batch) dimension of each input, evaluating the
# captured jaxpr at the merged shape, and then splitting the output back out.
#
# This is safe because all CP-wrapped kernels in this codebase treat dim 0 as
# a pure batch dimension that is processed independently per-element.


def _cp_batching_rule(
    axis_data,
    vals_in,
    dims_in,
    *,
    call,
    partition,
    in_tree,
    out_tree,
    static_args,
    **other_params,
):
    """Batching rule for ``custom_partitioning_p``.

    Merges the ``vmap`` mapped axis into the leading (batch) dimension,
    re-invokes the underlying computation with the merged shapes, then
    splits the output back.

    For batched args the mapped axis is moved to front and reshaped:
    ``(V, B, ...) -> (V*B, ...)``.  Unbatched args with at least one
    dimension are tiled along dim 0 so they broadcast correctly against
    the merged-batch inputs.

    The raw callable is obtained from the ``partition`` function's
    ``_cp_raw_fn`` attribute (set by :func:`make_cp_function`).  For CP
    wrappers created outside ``make_cp_function``, falls back to
    ``eval_jaxpr`` on the captured jaxpr.
    """
    from jax._src import core
    from jax._src.custom_partitioning import custom_partitioning_p
    from jax._src.interpreters.batching import not_mapped

    vmap_size = axis_data.size
    any_batched = any(d is not not_mapped for d in dims_in)

    if not any_batched:
        # Nothing to merge — just call through.
        out_flat = custom_partitioning_p.bind(
            *vals_in,
            call=call,
            partition=partition,
            in_tree=in_tree,
            out_tree=out_tree,
            static_args=static_args,
            **other_params,
        )
        if custom_partitioning_p.multiple_results:
            return out_flat, [not_mapped] * len(out_flat)
        return out_flat, not_mapped

    # Merge vmap dim into batch dim for batched args;
    # tile unbatched args so shapes are compatible.
    merged_vals = []
    for val, dim in zip(vals_in, dims_in):
        if dim is not not_mapped:
            val = jnp.moveaxis(val, dim, 0)
            if val.ndim > 1:
                # (V, B, ...) -> (V*B, ...)
                val = val.reshape(val.shape[0] * val.shape[1], *val.shape[2:])
        elif val.ndim > 0:
            # Unbatched array with a batch dim — tile V times along dim 0.
            val = jnp.tile(val, (vmap_size,) + (1,) * (val.ndim - 1))
        merged_vals.append(val)

    # Re-invoke the underlying computation with merged shapes.
    # Prefer the raw callable attached by make_cp_function (handles
    # pallas_call and other shape-sensitive primitives correctly).
    # Fall back to eval_jaxpr for CP wrappers created elsewhere.
    raw_fn = getattr(partition, "_cp_raw_fn", None)
    if raw_fn is not None:
        args = jax.tree_util.tree_unflatten(in_tree, merged_vals)
        result = raw_fn(*args)
        out_flat, _ = jax.tree_util.tree_flatten(result)
    else:
        out_flat = core.eval_jaxpr(call.jaxpr, call.consts, *merged_vals)

    # Split the merged batch dim back: (V*B, ...) -> (V, B, ...)
    if custom_partitioning_p.multiple_results:
        results = []
        out_dims = []
        for o in out_flat:
            if o.ndim > 0:
                o = o.reshape(vmap_size, -1, *o.shape[1:])
                results.append(o)
                out_dims.append(0)
            else:
                results.append(o)
                out_dims.append(not_mapped)
        return results, out_dims

    out = out_flat[0] if isinstance(out_flat, (list, tuple)) else out_flat
    if out.ndim > 0:
        out = out.reshape(vmap_size, -1, *out.shape[1:])
        return out, 0
    return out, not_mapped


def _register_cp_batching_rule() -> None:
    """Register the batching rule for ``custom_partitioning_p``.

    Called once at module import time.  Safe to call multiple times
    (idempotent).
    """
    from jax._src.custom_partitioning import custom_partitioning_p
    from jax._src.interpreters import batching

    batching.fancy_primitive_batchers[custom_partitioning_p] = _cp_batching_rule


_register_cp_batching_rule()

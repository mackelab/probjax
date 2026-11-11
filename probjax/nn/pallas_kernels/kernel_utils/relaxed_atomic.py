"""Relaxed-ordering atomic add for Pallas Triton kernels.

``jax.experimental.pallas.triton.atomic_add`` always lowers to an
acquire-release, GPU-scope ``atomic_rmw``: every element add carries memory
fences. Accumulation buffers that are only read after the kernel finishes
(e.g. the dQ buffer of a single-pass attention backward) need no ordering,
which is what FlashAttention-2 uses (``red.global.add.f32``). On an RTX 3090
the acquire-release form made such a backward ~4x slower than without the adds.

:func:`relaxed_atomic_add` is the same primitive with ``MemSemantic.RELAXED``.
It reuses the Pallas abstract evaluation and pointer lowering, so it accepts
the same ``(ref, idx, val, mask)`` arguments. If the private Pallas lowering
API it relies on is unavailable it falls back to the public ``atomic_add``.
"""

from __future__ import annotations

from typing import Any

from jax.experimental.pallas import triton as plgpu

__all__ = ["relaxed_atomic_add", "RELAXED_ATOMICS_AVAILABLE"]

try:  # Private Pallas Triton internals (checked at import time).
    from jax._src import core as _jax_core
    from jax._src import tree_util as _tree_util
    from jax._src.lib.triton import dialect as _tt_dialect
    from jax._src.pallas.triton import lowering as _lowering
    from jax._src.pallas.triton import primitives as _tprims
    from jax._src.state import indexing as _indexing
    from jax._src.state import primitives as _state_primitives
    from jax._src.state import types as _state_types

    _relaxed_atomic_add_p = _jax_core.Primitive("probjax_relaxed_atomic_add")

    @_relaxed_atomic_add_p.def_effectful_abstract_eval
    def _abstract_eval(*avals_flat, args_tree):
        return _tprims._atomic_abstract_eval(
            *avals_flat, args_tree=args_tree, atomic_type=_tprims.AtomicOpType.ADD
        )

    @_lowering.register_lowering(_relaxed_atomic_add_p)
    def _lowering_rule(ctx, *args_flat, args_tree):
        block_info, *_ = ctx.block_infos
        assert block_info is not None
        ptr, indexers, val, mask = args_tree.unflatten(args_flat)
        *_, value_aval, mask_aval = args_tree.unflatten(ctx.avals_in)
        indexers = list(indexers)
        if not indexers or not isinstance(indexers[-1], _indexing.NDIndexer):
            ref_aval = _state_types.transform_type(indexers, ctx.avals_in[0])
            indexers.append(_indexing.NDIndexer.make_trivial_indexer(ref_aval.shape))
        if len(indexers) != 1:
            raise NotImplementedError("Only single indexer is supported.")
        ptr = _lowering._compute_pointers_from_indices(ptr, block_info, indexers[0])
        val = _lowering._ensure_ir_value(val, value_aval)
        if mask is not None:
            mask = _lowering._ensure_ir_value(mask, mask_aval)
        return _lowering._atomic_rmw(
            _tt_dialect.RMWOp.FADD,
            ptr,
            val,
            mask=mask,
            semantic=_tt_dialect.MemSemantic.RELAXED,
        )

    def relaxed_atomic_add(x_ref_or_view, idx, val, *, mask: Any | None = None):
        """``x_ref[idx] += val`` with relaxed memory ordering (float add)."""
        x_ref, transforms = _state_primitives.get_ref_and_transforms(
            x_ref_or_view, idx, "relaxed_atomic_add"
        )
        args_flat, args_tree = _tree_util.tree_flatten((x_ref, transforms, val, mask))
        return _relaxed_atomic_add_p.bind(*args_flat, args_tree=args_tree)

    RELAXED_ATOMICS_AVAILABLE = True
except (ImportError, AttributeError):  # pragma: no cover - depends on JAX version

    def relaxed_atomic_add(x_ref_or_view, idx, val, *, mask: Any | None = None):
        """Fallback: the public (acquire-release) ``atomic_add``."""
        return plgpu.atomic_add(x_ref_or_view, idx, val, mask=mask)

    RELAXED_ATOMICS_AVAILABLE = False

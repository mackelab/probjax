"""Declarative, sharding-aware JAX primitives.

Each kernel (pallas or otherwise) declares a :class:`KernelSpec` — its
operands with named sharding factors, and its outputs. One spec drives five
things that used to be hand-written (and could drift) per kernel:

1. the ``custom_partitioning`` sharding rule string (Shardy-ready),
2. the backward spec, derived mechanically from the forward spec,
3. the sharding validator,
4. the vmap batching rule (registered on the kernel's own primitive —
   no global patching of ``custom_partitioning_p``),
5. the abstract evaluation (output shapes/dtypes).

:func:`make_kernel_primitive` builds a :class:`jax.extend.core.Primitive`
whose ``impl`` and lowering are the raw implementation. Sharding comes from
:func:`shardable_kernel`: it binds the primitive *inside* a memoized
``custom_partitioning`` wrapper applied at trace time (applying it at
lowering time is not possible: the nested lowering context loses the device
assignment, verified empirically). AD wrappers (``custom_vjp``/``custom_jvp``)
go outside, as ``custom_partitioning`` supports neither AD nor vmap itself.
Optional operands are genuinely optional (absent operands are not bound;
the ``present`` tuple in ``eqn.params`` records which are) — no scalar
sentinels.

For functions that are simply batch-parallel along a leading axis (e.g. the
vmapped flow inverse used by ``logpdf``), :func:`batch_shard` provides a
spec-free per-shard wrapper whose rule is derived from the actual avals.

``vmap`` support: this module registers a *general* batching rule for
``custom_partitioning_p`` that simply inlines the wrapped body under vmap
(the sharding annotation is dropped, the semantics are preserved, and any
kernel primitive inside applies its own batch-merging rule). This replaces
the previous kernel-specific merge/tile heuristic and is faithful for
arbitrary ``custom_partitioning`` users.

Factor conventions: dims are factor names like ``"batch"``, ``"seq"``,
``"heads"``; ``"_"`` is an anonymous always-replicated dim (given a fresh
factor name per operand). Factors in ``KernelSpec.shardable`` may be sharded;
all others are declared as replication factors to the partitioner and
enforced by the validator.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import mlir
from jax._src.interpreters import batching

__all__ = [
    "in_per_shard_body",
    "Operand",
    "Output",
    "KernelSpec",
    "ct",
    "res",
    "out_res",
    "grad",
    "derive_bwd_spec",
    "sharding_rule",
    "make_kernel_primitive",
    "bind_kernel",
    "shardable_kernel",
    "batch_shard",
    "def_partition_compat",
    "register_general_cp_batching",
]


# ---------------------------------------------------------------------------
# Per-shard body context
# ---------------------------------------------------------------------------
# Active while tracing a custom_partitioning per-shard body. Sharding
# constraints are meaningless there (the code already runs per-shard) and
# with_sharding_constraint is invalid under the Manual axes the partitioner
# lowers with — probjax.nn.sharding consults this flag to no-op.

import contextlib
import threading

_PER_SHARD = threading.local()


def in_per_shard_body() -> bool:
    return getattr(_PER_SHARD, "active", False)


@contextlib.contextmanager
def _per_shard_body():
    previous = in_per_shard_body()
    _PER_SHARD.active = True
    try:
        yield
    finally:
        _PER_SHARD.active = previous


# ---------------------------------------------------------------------------
# JAX version shims
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _def_partition_supports_need_replication_factors(def_partition) -> bool:
    """Whether ``custom_partitioning.def_partition`` accepts the kwarg."""
    try:
        return (
            "need_replication_factors" in inspect.signature(def_partition).parameters
        )
    except (TypeError, ValueError):
        return False


def def_partition_compat(def_partition, /, **kwargs):
    """Call ``def_partition`` across JAX versions.

    Older JAX builds do not accept ``need_replication_factors``.
    """
    if (
        "need_replication_factors" in kwargs
        and not _def_partition_supports_need_replication_factors(def_partition)
    ):
        kwargs.pop("need_replication_factors")
    return def_partition(**kwargs)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Operand:
    name: str
    dims: tuple[str, ...]  # factor names; "_" = anonymous replicated dim
    optional: bool = False


@dataclasses.dataclass(frozen=True)
class Output:
    dims: tuple[str, ...]
    # Operand name (str) to copy dtype from, a tuple of operand names for
    # jnp.result_type promotion, or a concrete dtype.
    dtype_like: Any


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    name: str
    operands: tuple[Operand, ...]
    outputs: tuple[Output, ...]
    shardable: frozenset[str] = frozenset({"batch", "heads", "groups"})

    def __post_init__(self):
        for out in self.outputs:
            if "_" in out.dims:
                raise ValueError(
                    f"{self.name}: outputs may not use anonymous dims; "
                    "name the factor so it can be resolved."
                )

    def operand(self, name: str) -> Operand:
        for op in self.operands:
            if op.name == name:
                return op
        raise KeyError(f"{self.name}: no operand named {name!r}")

    def present_operands(self, present: tuple[str, ...]) -> tuple[Operand, ...]:
        return tuple(
            op for op in self.operands if not op.optional or op.name in present
        )


def _named_dims(op: Operand) -> tuple[str, ...]:
    """Resolve anonymous '_' dims to fresh per-operand factor names."""
    return tuple(
        dim if dim != "_" else f"e_{op.name}_{i}" for i, dim in enumerate(op.dims)
    )


# ---------------------------------------------------------------------------
# Consumer 1: sharding rule
# ---------------------------------------------------------------------------


def sharding_rule(
    spec: KernelSpec, present: tuple[str, ...] = ()
) -> tuple[str, tuple[str, ...]]:
    """Einsum-like custom_partitioning rule + replication factors."""
    fragments = []
    factors: list[str] = []
    for op in spec.present_operands(present):
        dims = _named_dims(op)
        fragments.append(" ".join(dims))
        factors.extend(dims)
    out_fragments = [" ".join(out.dims) for out in spec.outputs]
    for out in spec.outputs:
        factors.extend(out.dims)

    rule = ", ".join(fragments) + " -> " + ", ".join(out_fragments)
    seen: dict[str, None] = {}
    for factor in factors:
        if factor not in spec.shardable:
            seen.setdefault(factor)
    return rule, tuple(seen)


# ---------------------------------------------------------------------------
# Consumer 2: derived backward spec
# ---------------------------------------------------------------------------
# Backward operands reference the forward spec so dims can never drift:
#   ct(i)        cotangent of forward output i
#   res(name)    forward operand carried as residual
#   out_res(i)   forward output carried as residual (e.g. out, lse)
#   grad(name)   backward OUTPUT: gradient w.r.t. forward operand `name`


@dataclasses.dataclass(frozen=True)
class ct:
    index: int = 0
    name: str | None = None  # override operand name (default: f"ct{i}")


@dataclasses.dataclass(frozen=True)
class res:
    name: str


@dataclasses.dataclass(frozen=True)
class out_res:
    index: int
    name: str


@dataclasses.dataclass(frozen=True)
class grad:
    name: str


def derive_bwd_spec(
    fwd: KernelSpec,
    *,
    name: str,
    operands: Sequence[ct | res | out_res | Operand],
    outputs: Sequence[grad],
    shardable: frozenset[str] | None = None,
) -> KernelSpec:
    """Build the backward KernelSpec mechanically from the forward one."""
    bwd_operands: list[Operand] = []
    for ref in operands:
        if isinstance(ref, ct):
            out = fwd.outputs[ref.index]
            bwd_operands.append(Operand(ref.name or f"ct{ref.index}", out.dims))
        elif isinstance(ref, res):
            fwd_op = fwd.operand(ref.name)
            bwd_operands.append(
                Operand(fwd_op.name, fwd_op.dims, optional=fwd_op.optional)
            )
        elif isinstance(ref, out_res):
            bwd_operands.append(Operand(ref.name, fwd.outputs[ref.index].dims))
        elif isinstance(ref, Operand):
            bwd_operands.append(ref)
        else:
            raise TypeError(f"Unsupported backward operand reference: {ref!r}")

    bwd_outputs = tuple(
        Output(fwd.operand(g.name).dims, dtype_like=g.name) for g in outputs
    )
    return KernelSpec(
        name=name,
        operands=tuple(bwd_operands),
        outputs=bwd_outputs,
        shardable=shardable if shardable is not None else fwd.shardable,
    )


# ---------------------------------------------------------------------------
# Consumer 3: validator
# ---------------------------------------------------------------------------


def _validate_shardings(
    spec: KernelSpec, present: tuple[str, ...], arg_shapes
) -> None:
    for op, shape in zip(spec.present_operands(present), arg_shapes, strict=True):
        sharding = getattr(shape, "sharding", None)
        pspec = getattr(sharding, "spec", None)
        if pspec is None:
            continue
        dims = _named_dims(op)
        for dim_idx, axis in enumerate(pspec):
            if axis is None or dim_idx >= len(dims):
                continue
            factor = dims[dim_idx]
            if factor not in spec.shardable:
                raise ValueError(
                    f"{spec.name}: sharding dimension {dim_idx} ('{factor}') of "
                    f"operand '{op.name}' over mesh axis '{axis}' is not "
                    f"supported; got PartitionSpec{tuple(pspec)}. Shardable "
                    f"factors: {sorted(spec.shardable)}."
                )


# ---------------------------------------------------------------------------
# Consumer 5: abstract eval
# ---------------------------------------------------------------------------


def _resolve_factors(
    spec: KernelSpec, present: tuple[str, ...], shapes: Sequence[tuple[int, ...]]
) -> dict[str, int]:
    """Resolve factor sizes from the *required* operands.

    Optional aux operands (mask/bias data, iota ids, ...) are skipped: their
    layouts vary (broadcast shapes, block-level arrays) and their dims are
    replicated regardless, so they neither define nor constrain factor sizes.
    """
    sizes: dict[str, int] = {}
    for op, shape in zip(spec.present_operands(present), shapes, strict=True):
        if op.optional:
            continue
        dims = _named_dims(op)
        if len(dims) != len(shape):
            raise ValueError(
                f"{spec.name}: operand '{op.name}' expected rank {len(dims)} "
                f"({dims}), got shape {tuple(shape)}."
            )
        for factor, size in zip(dims, shape):
            previous = sizes.setdefault(factor, int(size))
            if previous != int(size):
                raise ValueError(
                    f"{spec.name}: inconsistent size for factor '{factor}': "
                    f"{previous} vs {size} (operand '{op.name}')."
                )
    return sizes


# ---------------------------------------------------------------------------
# Static-param wrapping
# ---------------------------------------------------------------------------


class _IdKey:
    """Identity-hash wrapper for unhashable static params (e.g. BlockSpec).

    JAX requires eqn params to be hashable (enforced under remat/partial
    eval); values like ``pl.BlockSpec`` are not. They are wrapped at bind
    time and unwrapped only when calling the raw implementation.
    """

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        return isinstance(other, _IdKey) and self.obj is other.obj


def _wrap_statics(static: Mapping[str, Any]) -> dict[str, Any]:
    wrapped = {}
    for key, value in static.items():
        try:
            hash(value)
        except TypeError:
            value = _IdKey(value)
        wrapped[key] = value
    return wrapped


def _unwrap_statics(static: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (value.obj if isinstance(value, _IdKey) else value)
        for key, value in static.items()
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SPEC_BY_PRIM: dict[Primitive, KernelSpec] = {}


def _reconstruct_args(
    spec: KernelSpec, present: tuple[str, ...], args: Sequence[Any]
) -> list[Any]:
    """Map bound (present-only) args back to the full operand list with Nones."""
    full: list[Any] = []
    it = iter(args)
    for op in spec.operands:
        if op.optional and op.name not in present:
            full.append(None)
        else:
            full.append(next(it))
    return full


def make_kernel_primitive(
    spec: KernelSpec,
    *,
    impl: Callable,
) -> Primitive:
    """Build a Primitive for a kernel from its spec.

    ``impl(*operands, **static_params)`` is the raw implementation; absent
    optional operands are passed as ``None``. All non-operand parameters go
    into ``eqn.params`` at bind time; unhashable values are identity-wrapped
    (fresh closures only reduce the partitioning cache hit rate, never
    correctness).
    """
    prim = Primitive(spec.name)
    prim.multiple_results = len(spec.outputs) > 1
    _SPEC_BY_PRIM[prim] = spec

    def _call_impl(args, present, static):
        full = _reconstruct_args(spec, present, args)
        return impl(*full, **_unwrap_statics(static))

    # -- impl: eager / single-device path (no custom_partitioning) --------
    def _impl(*args, present, **static):
        return _call_impl(args, present, static)

    prim.def_impl(_impl)

    # -- abstract eval -----------------------------------------------------
    def _abstract_eval(*avals, present, **static):
        del static
        sizes = _resolve_factors(spec, present, [a.shape for a in avals])
        dtypes = {
            op.name: aval.dtype
            for op, aval in zip(spec.present_operands(present), avals, strict=True)
        }
        results = []
        for out in spec.outputs:
            shape = tuple(sizes[f] for f in out.dims)
            if isinstance(out.dtype_like, str):
                dtype = dtypes[out.dtype_like]
            elif isinstance(out.dtype_like, tuple):
                dtype = jnp.result_type(*(dtypes[n] for n in out.dtype_like))
            else:
                dtype = jnp.dtype(out.dtype_like)
            results.append(jax.core.ShapedArray(shape, dtype))
        return results if prim.multiple_results else results[0]

    prim.def_abstract_eval(_abstract_eval)

    # -- lowering: the raw implementation ----------------------------------
    # (Sharding is handled by shardable_kernel's trace-time
    # custom_partitioning wrapper; a lowering-time wrapper is impossible —
    # the nested lowering context loses the device assignment.)
    def _lowered(*args, present, **static):
        return _call_impl(args, present, static)

    mlir.register_lowering(
        prim, mlir.lower_fun(_lowered, multiple_results=prim.multiple_results)
    )

    # -- batching (scoped to this primitive) --------------------------------
    def _batching(axis_data, vals_in, dims_in, *, present, **static):
        from jax._src.interpreters.batching import not_mapped

        vmap_size = axis_data.size
        operands = spec.present_operands(present)

        merged = []
        for op, val, dim in zip(operands, vals_in, dims_in, strict=True):
            has_batch = len(op.dims) > 0 and op.dims[0] == "batch"
            if dim is not not_mapped:
                if not has_batch:
                    raise NotImplementedError(
                        f"{spec.name}: cannot vmap over operand '{op.name}' "
                        "which has no batch dimension."
                    )
                val = jnp.moveaxis(val, dim, 0)
                val = val.reshape(val.shape[0] * val.shape[1], *val.shape[2:])
            elif has_batch:
                # Unbatched operand with a batch dim: tile to the merged size.
                val = jnp.tile(val, (vmap_size,) + (1,) * (val.ndim - 1))
            merged.append(val)

        out = prim.bind(*merged, present=present, **static)
        outs = out if prim.multiple_results else (out,)
        split = tuple(o.reshape(vmap_size, -1, *o.shape[1:]) for o in outs)
        dims_out = (0,) * len(split)
        if prim.multiple_results:
            return list(split), list(dims_out)
        return split[0], 0

    batching.fancy_primitive_batchers[prim] = _batching

    return prim


def _split_present(
    spec: KernelSpec, arrays: Mapping[str, Any]
) -> tuple[tuple[str, ...], list[Any]]:
    present = []
    args = []
    for op in spec.operands:
        value = arrays.get(op.name)
        if value is None:
            if not op.optional:
                raise ValueError(
                    f"{spec.name}: required operand '{op.name}' is missing."
                )
            continue
        if op.optional:
            present.append(op.name)
        args.append(value)
    return tuple(present), args


def bind_kernel(prim: Primitive, arrays: Mapping[str, Any], **static) -> Any:
    """Bind a kernel primitive from a name->array mapping (no sharding wrap).

    Optional operands with value ``None`` (or missing) are omitted and
    recorded in the ``present`` param. Required operands must be arrays.
    """
    spec = _SPEC_BY_PRIM[prim]
    present, args = _split_present(spec, arrays)
    return prim.bind(*args, present=present, **_wrap_statics(static))


@functools.lru_cache(maxsize=None)
def _cp_for(prim: Primitive, present: tuple[str, ...], static_items: tuple):
    """Memoized custom_partitioning wrapper binding *prim* per-shard."""
    from jax.experimental.custom_partitioning import custom_partitioning

    spec = _SPEC_BY_PRIM[prim]
    static = dict(static_items)

    def _fn(*args):
        return prim.bind(*args, present=present, **static)

    cp = custom_partitioning(_fn)
    rule, replication_factors = sharding_rule(spec, present)

    def _partition(mesh, arg_shapes, result_shape):
        _validate_shardings(spec, present, jax.tree.leaves(arg_shapes))
        result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

        def lower_fn(*args):
            with _per_shard_body():
                return _fn(*args)

        return mesh, lower_fn, result_shardings, arg_shardings

    def_partition_compat(
        cp.def_partition,
        partition=_partition,
        sharding_rule=rule,
        need_replication_factors=replication_factors,
    )
    return cp


def shardable_kernel(prim: Primitive, arrays: Mapping[str, Any], **static) -> Any:
    """Bind *prim* inside its memoized custom_partitioning wrapper.

    Under jit with sharded inputs, GSPMD consults the spec-derived sharding
    rule and runs the kernel per-shard. Eagerly (no tracers), the wrapper is
    skipped and the primitive's impl runs directly.
    """
    spec = _SPEC_BY_PRIM[prim]
    present, args = _split_present(spec, arrays)
    wrapped = _wrap_statics(static)
    if not any(isinstance(arg, jax.core.Tracer) for arg in args):
        return prim.bind(*args, present=present, **wrapped)
    cp = _cp_for(prim, present, tuple(sorted(wrapped.items())))
    return cp(*args)


# ---------------------------------------------------------------------------
# batch_shard: spec-free per-shard wrapper for batch-parallel functions
# ---------------------------------------------------------------------------


def _batch_rule_from_shapes(
    in_shapes: tuple[tuple[int, ...], ...],
    out_shapes: tuple[tuple[int, ...], ...],
    num_batch_inputs: int,
) -> tuple[str, tuple[str, ...]]:
    """Rule: leading axis of the first *num_batch_inputs* rank>=1 arrays is
    the shared batch; remaining inputs (hoisted consts) are replicated."""
    fragments = []
    factors: list[str] = []
    for i, shape in enumerate(in_shapes):
        if len(shape) == 0:
            fragments.append("")
            continue
        if i < num_batch_inputs:
            dims = ("b",) + tuple(f"i{i}_{d}" for d in range(1, len(shape)))
            factors.extend(dims[1:])
        else:
            dims = tuple(f"c{i}_{d}" for d in range(len(shape)))
            factors.extend(dims)
        fragments.append(" ".join(dims))
    out_fragments = []
    for i, shape in enumerate(out_shapes):
        if len(shape) == 0:
            out_fragments.append("")
            continue
        dims = ("b",) + tuple(f"o{i}_{d}" for d in range(1, len(shape)))
        out_fragments.append(" ".join(dims))
        factors.extend(dims[1:])
    rule = ", ".join(fragments) + " -> " + ", ".join(out_fragments)
    return rule, tuple(dict.fromkeys(factors))


def batch_shard(fn: Callable) -> Callable:
    """Run *fn* per-shard along the leading (batch) axis under an active mesh.

    For functions that are independent along the leading axis of all their
    rank>=1 array inputs and outputs (e.g. a vmapped flow inverse). Under
    jit with batch-sharded inputs, the body runs per-shard — local, unsharded
    shapes — which avoids GSPMD rematerialization in Auto mode and sidesteps
    unimplemented sharded-op cases in Explicit mode. Without an active mesh
    (or called eagerly), *fn* runs unchanged.

    Closure constants (e.g. module weights) are hoisted into explicit,
    fully-replicated operands via ``jax.closure_convert`` —
    ``custom_partitioning`` rejects bodies with captured consts. Scalar
    inputs/outputs are replicated; all non-leading dims are replication
    factors.
    """
    from jax.experimental.custom_partitioning import custom_partitioning

    @functools.wraps(fn)
    def wrapped(*args):
        from jax.sharding import AxisType

        mesh = jax.sharding.get_abstract_mesh()
        has_tracer = any(isinstance(a, jax.core.Tracer) for a in args)
        if mesh.empty or not has_tracer:
            return fn(*args)
        if AxisType.Explicit in set(mesh.axis_types):
            # Explicit sharding-in-types requires per-op sharding rules that
            # arbitrary bodies (scans, gathers) don't have yet; run the
            # unwrapped fn and let JAX report what it can't shard.
            return fn(*args)

        # Hoist ALL closure consts (jax.closure_convert only hoists
        # differentiable ones; integer mask/permutation arrays would remain
        # captured and custom_partitioning rejects captured consts).
        closed = jax.make_jaxpr(fn)(*args)
        consts = tuple(closed.consts)
        all_args = tuple(args) + consts
        num_batch = len(args)
        out_struct = jax.eval_shape(fn, *args)
        out_tree = jax.tree.structure(out_struct)

        def _positional(*xs):
            flat = jax.core.eval_jaxpr(
                closed.jaxpr, list(xs[num_batch:]), *xs[:num_batch]
            )
            return jax.tree.unflatten(out_tree, flat)

        shapes = tuple(jnp.shape(a) for a in all_args)
        out_shapes = tuple(o.shape for o in jax.tree.leaves(out_struct))
        rule, replication = _batch_rule_from_shapes(shapes, out_shapes, num_batch)

        cp = custom_partitioning(_positional)

        def _partition(mesh, arg_shapes, result_shape):
            result_shardings = jax.tree.map(lambda s: s.sharding, result_shape)
            arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)

            def lower_fn(*xs):
                # Re-trace the original fn at the LOCAL (per-shard) shapes:
                # the hoisted-const jaxpr is shape-specialized to the global
                # batch and cannot be evaluated on shards. Closure consts are
                # permitted here (the no-consts restriction applies only to
                # the top-level custom_partitioning trace).
                with _per_shard_body():
                    return fn(*xs[:num_batch])

            return mesh, lower_fn, result_shardings, arg_shardings

        def_partition_compat(
            cp.def_partition,
            partition=_partition,
            sharding_rule=rule,
            need_replication_factors=replication,
        )
        return cp(*all_args)

    return wrapped


# ---------------------------------------------------------------------------
# General batching rule for custom_partitioning_p
# ---------------------------------------------------------------------------
# JAX ships no vmap rule for custom_partitioning. Under vmap we inline the
# wrapped body (dropping the sharding annotation, preserving semantics); any
# kernel primitive inside then applies its own batch-merging rule. This is
# faithful for arbitrary custom_partitioning users, unlike the previous
# kernel-specific merge/tile heuristic it replaces.


def _cp_general_batching(axis_data, vals_in, dims_in, *, call, **params):
    from jax._src.interpreters.batching import not_mapped

    def fn(*args):
        out = jax.core.eval_jaxpr(call.jaxpr, call.consts, *args)
        return out

    in_axes = tuple(None if d is not_mapped else int(d) for d in dims_in)
    out = jax.vmap(fn, in_axes=in_axes, out_axes=0, axis_size=axis_data.size)(
        *vals_in
    )
    return out, [0] * len(out)


def _cp_general_jvp(primals, tangents, *, call, **params):
    """Inline the custom_partitioning body under differentiation.

    custom_partitioning has no AD rule; since it is semantically the
    identity wrapper around its body, differentiating the inlined body is
    faithful (the sharding annotation is dropped for the tangent
    computation; the AD layers of probjax kernels sit OUTSIDE their CP
    wrappers and never hit this path).
    """
    from jax.interpreters import ad

    def fn(*args):
        return jax.core.eval_jaxpr(call.jaxpr, call.consts, *args)

    tangents = [
        ad.instantiate_zeros(t) if isinstance(t, ad.Zero) else t for t in tangents
    ]
    return jax.jvp(fn, tuple(primals), tuple(tangents))


def _cp_general_transpose(cts, *args, call, **params):
    """Transpose the inlined custom_partitioning body.

    Linear operands arrive as UndefinedPrimal; the body's own primitives
    (e.g. a kernel jvp primitive) supply their transpose rules — which is
    how a CP-wrapped tangent computation transposes into the fused backward
    kernel.
    """
    from jax.interpreters import ad

    def fn(*xs):
        return jax.core.eval_jaxpr(call.jaxpr, call.consts, *xs)

    is_linear = [ad.is_undefined_primal(a) for a in args]
    fixed = [a for a, lin in zip(args, is_linear) if not lin]
    examples = [
        jax.ShapeDtypeStruct(a.aval.shape, a.aval.dtype)
        for a, lin in zip(args, is_linear)
        if lin
    ]

    def linear_fn(linear_args):
        lin_iter = iter(linear_args)
        fixed_iter = iter(fixed)
        full = [
            next(lin_iter) if lin else next(fixed_iter) for lin in is_linear
        ]
        return fn(*full)

    transpose_fn = jax.linear_transpose(linear_fn, examples)
    (linear_cts,) = transpose_fn(list(cts))
    lin_iter = iter(linear_cts)
    return tuple(next(lin_iter) if lin else None for lin in is_linear)


def register_general_cp_batching() -> None:
    """Register (or re-register) the general custom_partitioning vmap rule.

    Idempotent. Called at import; ``old_pallas`` overwrites this table entry
    with its legacy heuristic on import, so A/B comparisons should call this
    again afterwards.
    """
    from jax._src.custom_partitioning import custom_partitioning_p
    from jax.interpreters import ad

    batching.fancy_primitive_batchers[custom_partitioning_p] = _cp_general_batching
    ad.primitive_jvps[custom_partitioning_p] = _cp_general_jvp
    ad.primitive_transposes[custom_partitioning_p] = _cp_general_transpose


register_general_cp_batching()

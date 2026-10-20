from __future__ import annotations

from typing import Any, Sequence

from jax._src import ad_util
from jax._src import core as jax_core
from jax._src.dispatch import prim_requires_devices_during_lowering
from jax._src.interpreters import ad as ad_src
from jax.extend.core import ClosedJaxpr
from jax.interpreters import mlir


def call_impl(
    *args: Any, forward_jaxpr: ClosedJaxpr, single_result: bool = False, **params: Any
) -> Any:
    del params
    outs = jax_core.eval_jaxpr(forward_jaxpr.jaxpr, forward_jaxpr.consts, *args)
    if single_result:
        return outs[0]
    return outs


def call_abstract_eval(
    *avals: Any, forward_jaxpr: ClosedJaxpr, single_result: bool = False, **params: Any
) -> Any:
    del avals, params
    if single_result:
        return forward_jaxpr.out_avals[0]
    return tuple(forward_jaxpr.out_avals)


def call_lowering(
    ctx: Any,
    *mlir_args: Any,
    forward_jaxpr: ClosedJaxpr,
    lowering_name: str,
    **params: Any,
) -> Any:
    del params
    return mlir.core_call_lowering(
        ctx,
        *mlir_args,
        name=lowering_name,
        call_jaxpr=forward_jaxpr,
    )


def mark_primitive_requires_devices(primitive: jax_core.Primitive) -> None:
    """Register *primitive* so JAX populates ``device_assignment`` during lowering.

    Call primitives that capture a ``forward_jaxpr`` may contain
    ``custom_partitioning`` ops inside that jaxpr.  CP's lowering rule
    requires a ``ShardingContext`` with a valid ``device_assignment``.

    JAX decides whether to populate ``device_assignment`` by calling
    ``jaxpr_has_prim_requiring_devices``, which traverses
    ``core.subjaxprs``.  If the jaxpr is stored behind a ``Lazy`` wrapper
    or is otherwise opaque to that traversal, CP inside the jaxpr will hit
    ``device_assignment is None``.

    Registering the enclosing call primitive here guarantees the device
    assignment is always present when the primitive appears.
    """
    prim_requires_devices_during_lowering.add(primitive)


def jvp_from_forward_jaxpr(
    forward_jaxpr: ClosedJaxpr, primals: Sequence[Any], tangents: Sequence[Any]
) -> tuple[Any, ...]:
    nonzeros = [not isinstance(t, ad_util.Zero) for t in tangents]
    jvp_cj, out_nonzeros = ad_src.jvp_jaxpr(forward_jaxpr, nonzeros, instantiate=False)
    nonzero_tangents = [t for t in tangents if not isinstance(t, ad_util.Zero)]

    outs = jax_core.eval_jaxpr(
        jvp_cj.jaxpr,
        jvp_cj.consts,
        *primals,
        *nonzero_tangents,
    )
    n_primals_out = len(forward_jaxpr.out_avals)
    primal_outs = list(outs[:n_primals_out])
    tangent_outs_nz = list(outs[n_primals_out:])

    tangent_outs = []
    nz_iter = iter(tangent_outs_nz)
    for nz, aval in zip(out_nonzeros, forward_jaxpr.out_avals, strict=False):
        if nz:
            tangent_outs.append(next(nz_iter))
        else:
            tangent_outs.append(ad_util.Zero(aval))
    return primal_outs, tangent_outs

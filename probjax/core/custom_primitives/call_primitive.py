from jax._src import ad_util
from jax._src import core as jax_core
from jax._src.interpreters import ad as ad_src
from jax.interpreters import mlir


def call_impl(*args, forward_jaxpr, single_result: bool = False, **params):
    del params
    outs = jax_core.eval_jaxpr(forward_jaxpr.jaxpr, forward_jaxpr.consts, *args)
    if single_result:
        return outs[0]
    return outs


def call_abstract_eval(*avals, forward_jaxpr, single_result: bool = False, **params):
    del avals, params
    if single_result:
        return forward_jaxpr.out_avals[0]
    return tuple(forward_jaxpr.out_avals)


def call_lowering(ctx, *mlir_args, forward_jaxpr, lowering_name: str, **params):
    del params
    return mlir.core_call_lowering(
        ctx,
        *mlir_args,
        name=lowering_name,
        call_jaxpr=forward_jaxpr,
    )


def jvp_from_forward_jaxpr(forward_jaxpr, primals, tangents):
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

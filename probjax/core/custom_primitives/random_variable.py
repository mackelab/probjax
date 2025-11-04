from functools import lru_cache, partial
from threading import local

import jax
from jax._src.ad_util import Zero
from jax.core import eval_jaxpr
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe


class NameStack(local):
    def __init__(self):
        self.counts = {}

    def get_name(self, prefix="rv"):
        if prefix not in self.counts:
            self.counts[prefix] = 0
        count = self.counts[prefix]
        name = f"{prefix}_{count}"
        self.counts[prefix] += 1
        return name


name_stack = NameStack()


def _build_jaxpr_thunk_rvs(rvs_fn, key, args, kwds=None, shape=(), **params):
    del params

    if kwds is None:
        kwds = {}

    @lru_cache
    def _jaxpr_thunk():
        return jax.make_jaxpr(rvs_fn)(key, *args, **kwds, shape=shape)

    return _jaxpr_thunk


def _build_jaxpr_thunk_logpdf(
    rvs_fn, logpdf_fn, key, args, kwds=None, shape=(), **params
):
    del params

    if kwds is None:
        kwds = {}

    @lru_cache
    def _jaxpr_thunk():
        out_shape = jax.eval_shape(rvs_fn, key, *args, **kwds, shape=shape)
        out_aval = jax.core.ShapedArray(out_shape.shape, out_shape.dtype)
        return jax.make_jaxpr(logpdf_fn)(out_aval, *args, **kwds)

    return _jaxpr_thunk


def _rv_impl(key, *args, kwds=None, dist=None, shape=(), name=None, **params):
    del dist, name
    if kwds is None:
        kwds = {}
    rvs_fn = params.pop("rvs_fn")

    return rvs_fn(key, *args, **kwds, shape=shape)


def _rv_abstract_eval(key, *args, kwds=None, dist=None, shape=(), name=None, **params):
    del dist, name
    if kwds is None:
        kwds = {}

    rvs_fn = params.pop("rvs_fn")

    out = jax.eval_shape(rvs_fn, key, *args, **kwds, shape=shape)
    out = jax.core.ShapedArray(out.shape, out.dtype)
    return out


def _rv_lowering(ctx, key, *args, kwds=None, dist=None, name=None, **params):
    if kwds is None:
        kwds = {}

    rvs_jaxpr_thunk = params.pop("rvs_jaxpr_thunk")
    return mlir.core_call_lowering(
        ctx,
        key,
        *args,
        name=name,
        call_jaxpr=rvs_jaxpr_thunk(),
    )


def _rv_batching_rule(batched_args, batch_dims, **params):
    logpdf_fn = params.pop("logpdf_fn")
    kwds = params.pop("kwds")

    # Create batched versions of the functions
    rvs_fn_batched = jax.vmap(
        partial(_rv_impl, kwds=kwds, **params), in_axes=batch_dims
    )
    # No key for logpdf
    if any(batch_dims[1:]):
        logpdf_fn_batched = jax.vmap(logpdf_fn, in_axes=batch_dims[1:])
    else:
        logpdf_fn_batched = logpdf_fn

    # Create new jaxpr thunk!
    del params["rvs_fn"]
    del params["rvs_jaxpr_thunk"]
    del params["logpdf_jaxpr_thunk"]

    # Properly unpack batched_args: first element is key, rest are args
    key, *args = batched_args

    out = rv_p.bind(
        key,
        *args,
        rvs_fn=rvs_fn_batched,
        logpdf_fn=logpdf_fn_batched,
        **params,
    )

    out_dims = 0

    return out, out_dims


def custom_rv_jvp(primals, tangents, **params):
    # NOTE Differentiating a random variable will lead to a different distribution!
    # Example: x ~ N(loc, scale) then x = loc + scale * z, where z ~ N(0, 1)
    # Hence then d/dloc x = Dirac(1.) and d/dscale x = N(0, 1)
    # In other words the correct VJP would return another random variable with adjusted
    # distribution.
    rvs_fn_jaxpr = params["rvs_jaxpr_thunk"]()

    nonzeros = [type(t) is not Zero for t in tangents]
    forward_jvp_jaxpr, forward_out_nz = ad.jvp_jaxpr(
        rvs_fn_jaxpr, nonzeros, instantiate=False
    )
    nonzero_tangents = [t for t in tangents if type(t) is not Zero]
    forward_jvp_jaxpr_ = pe.convert_constvars_jaxpr(forward_jvp_jaxpr.jaxpr)

    # TODO: This should be bound to a new primitive with adjusted dist, pdf,...
    # For now we just evaluate the jaxpr and return the result i.e. we will lose its
    # interpretation as a random variable.
    new_primals, new_tangent = eval_jaxpr(
        forward_jvp_jaxpr_, forward_jvp_jaxpr.consts, *primals, *nonzero_tangents
    )

    return new_primals, new_tangent


def _rv_transpose_rule(*args, **kwargs):
    return ad.call_transpose(rv_p, *args, **kwargs)


class RandomVariable(Primitive):
    def __init__(self):
        super().__init__("random_variable")

    def bind(
        self,
        key,
        *args,
        shape=(),
        dist=None,
        name=None,
        rvs_fn=None,
        logpdf_fn=None,
        rvs_jaxpr_thunk=None,
        logpdf_jaxpr_thunk=None,
        kwds=None,
    ):
        if rvs_fn is None:
            rvs_fn = dist.rvs
        if logpdf_fn is None:
            logpdf_fn = dist.logpdf

        if kwds is None:
            kwds = {}

        if name is None:
            prefix = getattr(dist, "name", "rv")
            name = name_stack.get_name(prefix)

        args, kwds = dist._parse_args(*args, **kwds)

        if rvs_jaxpr_thunk is None:
            rvs_jaxpr_thunk = _build_jaxpr_thunk_rvs(
                rvs_fn, key, args, kwds, shape=shape
            )
        if logpdf_jaxpr_thunk is None:
            logpdf_jaxpr_thunk = _build_jaxpr_thunk_logpdf(
                rvs_fn, logpdf_fn, key, args, kwds, shape=shape
            )

        return super().bind(
            key,
            *args,
            kwds=None,
            shape=shape,
            dist=dist,
            name=name,
            rvs_fn=rvs_fn,
            rvs_jaxpr_thunk=rvs_jaxpr_thunk,
            logpdf_fn=logpdf_fn,
            logpdf_jaxpr_thunk=logpdf_jaxpr_thunk,
        )


rv_p = RandomVariable()
rv_p.def_impl(_rv_impl)
rv_p.def_abstract_eval(_rv_abstract_eval)
batching.primitive_batchers[rv_p] = _rv_batching_rule
mlir.register_lowering(rv_p, _rv_lowering)
ad.primitive_jvps[rv_p] = custom_rv_jvp
ad.primitive_transposes[rv_p] = _rv_transpose_rule

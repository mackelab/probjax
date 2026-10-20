from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ContextDecorator, contextmanager
from functools import lru_cache
from threading import local
from typing import Any, Callable, Literal, cast

from jax._src.core import shaped_abstractify
from jax._src.interpreters import ad as ad_src
from jax.extend.core import ClosedJaxpr, Primitive
from jax.interpreters import ad, batching, mlir
from jax.tree_util import tree_flatten, tree_unflatten

from probjax.core.custom_primitives.call_primitive import (
    call_abstract_eval,
    call_impl,
    call_lowering,
    jvp_from_forward_jaxpr,
)
from probjax.core.custom_primitives.common import (
    ThreadLocalDepth,
    batch_closed_jaxpr,
    ensure_hashable,
    move_mapped_axes_to_front,
    must_emit_primitive,
    parse_env_flag,
    trace_to_closed_jaxpr,
)
from probjax.core.custom_primitives.contracts import (
    parse_random_variable_call_params,
)


class NameStack(local):
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.scopes: list[str] = []

    def _scoped_prefix(self, prefix: str = "rv") -> str:
        if not self.scopes:
            return prefix
        return "__".join(tuple(self.scopes) + (prefix,))

    def get_name(self, prefix: str = "rv") -> str:
        scoped_prefix = self._scoped_prefix(prefix)
        if scoped_prefix not in self.counts:
            self.counts[scoped_prefix] = 0
        count = self.counts[scoped_prefix]
        name = f"{scoped_prefix}_{count}"
        self.counts[scoped_prefix] += 1
        return name

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        self.scopes.append(str(name))
        try:
            yield
        finally:
            self.scopes.pop()


name_stack = NameStack()


_RV_TRACING_ENV_VAR = "PROBJAX_RV_TRACING"

# Read once at import: opting the whole process back into the legacy
# always-emit behaviour. Tests can monkeypatch this module attribute.
_ALWAYS_TRACE_RV = parse_env_flag(_RV_TRACING_ENV_VAR)

_rv_tracing_state = ThreadLocalDepth()


def rv_tracing_enabled() -> bool:
    """Whether distribution sampling emits traceable ``rv_p`` sites.

    Read at trace time: outside :func:`enable_rv_tracing` (and without
    ``PROBJAX_RV_TRACING=1``) sampling is an ordinary JAX computation with no
    primitive, no ``forward_jaxpr`` tracing, and no ``name_stack`` side effects.
    """
    return _ALWAYS_TRACE_RV or _rv_tracing_state.depth > 0


class enable_rv_tracing(ContextDecorator):
    """Emit traceable ``rv_p`` sites for distribution sampling.

    Only needed when tracing a model whose jaxpr is inspected for random
    variables outside the PPL transformations -- ``trace``,
    ``joint_sample``, ``log_joint_fn``/``log_potential_fn``/``log_prob_fn``,
    and ``intervene``/``condition``/``substitute`` enable it automatically
    around their internal tracing. Everywhere else (``jit``, ``vmap``,
    ``grad``, ``scan``, plain sampling) sampling stays primitive-free.

    Reentrant and thread-local; safe to nest.
    """

    def __enter__(self) -> "enable_rv_tracing":
        _rv_tracing_state.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        # ``Literal[False]`` (rather than ``bool``) tells type checkers that
        # the block never swallows an exception, so names bound inside it are
        # definitely bound afterwards.
        _rv_tracing_state.__exit__(*exc)
        return False


@lru_cache(maxsize=4096)
def _trace_random_variable_jaxpr(
    rvs_fn: Callable,
    in_tree: Any,
    in_avals: Sequence[Any],
    shape: Any,
    kwds_items: tuple[tuple[str, Any], ...],
) -> ClosedJaxpr:
    kwds = dict(kwds_items)

    def f_dyn(*flat_inputs: Any) -> Any:
        inputs = tree_unflatten(in_tree, flat_inputs)
        key, *args = inputs
        return rvs_fn(key, *args, shape=shape, **kwds)

    fn_name = getattr(rvs_fn, "__name__", str(rvs_fn))
    forward_jaxpr, _, _ = trace_to_closed_jaxpr(
        f_dyn,
        in_tree=in_tree,
        in_avals=in_avals,
        debug_name="random_variable forward",
        const_context=f"random_variable forward ({fn_name})",
    )
    return forward_jaxpr


def _validated_rv_params(
    forward_jaxpr: ClosedJaxpr,
    in_tree: Any,
    shape: Any,
    dist: Any,
    name: str | None,
    rvs_fn: Callable,
    logpdf_fn: Callable,
    kwds_items: tuple[tuple[str, Any], ...],
) -> None:
    """Validate random-variable call params (shared impl/abstract preamble)."""
    _ = parse_random_variable_call_params({
        "forward_jaxpr": forward_jaxpr,
        "in_tree": in_tree,
        "shape": shape,
        "dist": dist,
        "name": name,
        "rvs_fn": rvs_fn,
        "logpdf_fn": logpdf_fn,
        "kwds_items": kwds_items,
    })


def _rv_impl(
    *flat_inputs: Any,
    forward_jaxpr: ClosedJaxpr,
    in_tree: Any,
    shape: Any,
    dist: Any,
    name: str | None,
    rvs_fn: Callable,
    logpdf_fn: Callable,
    kwds_items: tuple[tuple[str, Any], ...],
) -> Any:
    _validated_rv_params(
        forward_jaxpr, in_tree, shape, dist, name, rvs_fn, logpdf_fn, kwds_items
    )
    return call_impl(
        *flat_inputs,
        forward_jaxpr=forward_jaxpr,
        single_result=True,
        in_tree=in_tree,
        shape=shape,
        dist=dist,
        name=name,
        rvs_fn=rvs_fn,
        logpdf_fn=logpdf_fn,
        kwds_items=kwds_items,
    )


def _rv_abstract_eval(
    *flat_avals: Any,
    forward_jaxpr: ClosedJaxpr,
    in_tree: Any,
    shape: Any,
    dist: Any,
    name: str | None,
    rvs_fn: Callable,
    logpdf_fn: Callable,
    kwds_items: tuple[tuple[str, Any], ...],
) -> Any:
    _validated_rv_params(
        forward_jaxpr, in_tree, shape, dist, name, rvs_fn, logpdf_fn, kwds_items
    )
    return call_abstract_eval(
        *flat_avals,
        forward_jaxpr=forward_jaxpr,
        single_result=True,
        in_tree=in_tree,
        shape=shape,
        dist=dist,
        name=name,
        rvs_fn=rvs_fn,
        logpdf_fn=logpdf_fn,
        kwds_items=kwds_items,
    )


def _rv_lowering(ctx: Any, *mlir_args: Any, **params: Any) -> Any:
    params = dict(params)
    forward_jaxpr = params.pop("forward_jaxpr")
    lowering_name = params.pop("name", "random_variable")
    return call_lowering(
        ctx,
        *mlir_args,
        forward_jaxpr=forward_jaxpr,
        lowering_name=lowering_name,
        **params,
    )


def _rv_batching_rule(
    axis_data: Any,
    batched_args: Sequence[Any],
    batch_dims: Sequence[int | None],
    **params: Any,
) -> tuple[Any, int | None]:
    params = dict(params)
    forward_jaxpr = params["forward_jaxpr"]

    new_args, in_axes, any_batched = move_mapped_axes_to_front(batched_args, batch_dims)
    if not any_batched:
        out = Primitive.bind(rv_p, *new_args, **params)
        return out, None

    batched_forward_jaxpr, out_axes = batch_closed_jaxpr(
        forward_jaxpr, axis_data, in_axes
    )

    rebound_params = dict(params, forward_jaxpr=batched_forward_jaxpr)
    out = Primitive.bind(rv_p, *new_args, **rebound_params)
    out_axis = out_axes[0] if out_axes else None
    return out, out_axis


def _rv_jvp(
    primals: Sequence[Any], tangents: Sequence[Any], **params: Any
) -> tuple[Any, ...]:
    primal_outs, tangent_outs = jvp_from_forward_jaxpr(
        params["forward_jaxpr"],
        primals,
        tangents,
    )
    return primal_outs[0], tangent_outs[0]


def _rv_transpose_rule(*args: Any, **kwargs: Any) -> Any:
    call_transpose = ad_src.call_transpose
    return call_transpose(rv_p, *args, **kwargs)


class RandomVariableCallPrimitive(Primitive):
    def __init__(self) -> None:
        super().__init__("random_variable")

    def bind(
        self,
        key: Any,
        *args: Any,
        shape: Any = (),
        dist: Any = None,
        name: str | None = None,
        rvs_fn: Callable | None = None,
        logpdf_fn: Callable | None = None,
        kwds: Mapping[str, Any] | None = None,
        forward_jaxpr: ClosedJaxpr | None = None,
        in_tree: Any = None,
        kwds_items: tuple[tuple[str, Any], ...] | None = None,
        **params: Any,
    ) -> Any:
        if dist is None and (rvs_fn is None or logpdf_fn is None):
            raise ValueError("dist must be provided when rvs_fn/logpdf_fn are not set")

        dist_obj = cast(Any, dist)
        if rvs_fn is None:
            rvs_fn = getattr(dist_obj, "_rvs_impl", None)
            if rvs_fn is None:
                rvs_fn = dist_obj.rvs
        if rvs_fn is None:
            raise ValueError("The distribution must provide an `rvs` implementation.")
        if logpdf_fn is None:
            logpdf_fn = dist_obj.logpdf

        if kwds is None:
            kwds = {} if kwds_items is None else dict(kwds_items)

        args, parsed_kwds = dist_obj._parse_args(*args, **kwds)

        # Opt-in PPL tracing: outside `enable_rv_tracing` (and without
        # PROBJAX_RV_TRACING=1), sampling is an ordinary JAX computation.
        # This skips the primitive, the forward-jaxpr trace, the site-name
        # counter, and the static-param hashability checks -- so array-valued
        # kwargs also pass straight through to `rvs_fn`.
        if not rv_tracing_enabled():
            return rvs_fn(key, *args, shape=shape, **parsed_kwds)

        if name is None:
            prefix = getattr(dist_obj, "name", "rv")
            name = name_stack.get_name(prefix)

        if kwds_items is None:
            kwds_items = tuple(
                sorted(
                    (k, ensure_hashable(v, f"kwds['{k}']"))
                    for k, v in parsed_kwds.items()
                )
            )

        # Inside PPL tracing: outside any trace, with no tracers, sample
        # directly. Inside a trace the primitive must be emitted even for
        # concrete arguments, or the random variable disappears from the jaxpr
        # and every interpreter that looks for it -- trace, log_potential,
        # intervene -- stops seeing it.
        if not must_emit_primitive((key, args, parsed_kwds)):
            return rvs_fn(key, *args, shape=shape, **parsed_kwds)

        dyn_inputs = (key, *args)
        flat_inputs, dyn_tree = tree_flatten(dyn_inputs)
        if in_tree is None:
            in_tree = dyn_tree

        if forward_jaxpr is None:
            in_avals = tuple(map(shaped_abstractify, flat_inputs))
            forward_jaxpr = _trace_random_variable_jaxpr(
                rvs_fn,
                in_tree,
                in_avals,
                shape,
                kwds_items,
            )

        return super().bind(
            *flat_inputs,
            forward_jaxpr=forward_jaxpr,
            in_tree=in_tree,
            shape=shape,
            dist=dist_obj,
            name=name,
            rvs_fn=rvs_fn,
            logpdf_fn=logpdf_fn,
            kwds_items=kwds_items,
            **params,
        )


rv_p = RandomVariableCallPrimitive()
call_rv_p = rv_p

rv_p.def_impl(_rv_impl)
rv_p.def_abstract_eval(_rv_abstract_eval)
batching.fancy_primitive_batchers[rv_p] = _rv_batching_rule
mlir.register_lowering(rv_p, _rv_lowering)
ad.primitive_jvps[rv_p] = _rv_jvp
ad.primitive_transposes[rv_p] = _rv_transpose_rule

# See ``mark_primitive_requires_devices`` in ``call_primitive`` for details.
from probjax.core.custom_primitives.call_primitive import (  # noqa: E402
    mark_primitive_requires_devices,
)

mark_primitive_requires_devices(rv_p)


__all__ = [
    "NameStack",
    "call_rv_p",
    "enable_rv_tracing",
    "name_stack",
    "rv_p",
    "rv_tracing_enabled",
]

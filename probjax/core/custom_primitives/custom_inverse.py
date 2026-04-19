from functools import lru_cache, update_wrapper
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax._src.core import shaped_abstractify
from jax._src.util import safe_map, safe_zip
from jax.extend.core import ClosedJaxpr, Primitive
from jax.interpreters import ad, batching, mlir
from jax.tree_util import tree_flatten, tree_unflatten

from probjax.core.custom_primitives.call_primitive import (
    call_abstract_eval,
    call_impl,
    call_lowering,
    jvp_from_forward_jaxpr,
)
from probjax.core.custom_primitives.contracts import parse_custom_inverse_call_params
from probjax.core.custom_primitives.common import (
    Lazy,
    LazyClosedJaxpr,
    batch_closed_jaxpr,
    ensure_hashable,
    has_tracer,
    move_mapped_axes_to_front,
    trace_to_closed_jaxpr,
)

map = safe_map
zip = safe_zip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_inverse_thunk(msg: str):
    def thunk():
        raise ValueError(msg)

    return thunk


# ---------------------------------------------------------------------------
# custom_inverse wrapper
# ---------------------------------------------------------------------------


class custom_inverse:
    """
    Attach a custom inverse (and optional log-det) to a function via a primitive.

    - Plain (non-traced) calls: direct call to `fun`.
    - Under JAX transforms:
        emits `custom_inverse_call_p` with
          * `forward_jaxpr`: closed JAXPR of forward
          * `inverse_jaxpr_thunk`: lazy constructor for inverse JAXPR
        The thunk is only ever called by your inverse interpreter.
    """

    def __init__(self, fun: Callable, inv_argnum=0, static_argnums=None) -> None:
        update_wrapper(self, fun)
        self.fun = fun
        self.inv_argnum = inv_argnum
        self.static_argnums = (
            None if static_argnums is None else tuple(sorted(static_argnums))
        )

        self.inv_fun = None
        self.inv_fun_and_log_det = None
        self.value_and_logdet_fun = None

        # Per-instance cached builder.
        @lru_cache(maxsize=2048)
        def _trace(
            dyn_idxs: Tuple[int, ...],
            in_tree,
            in_avals: Tuple[Any, ...],
            static_args_key: Tuple[Any, ...],
            params_key: Tuple[Tuple[str, Any], ...],
        ):
            static_args = static_args_key
            params = dict(params_key)
            return self._build_jaxprs_for_signature(
                dyn_idxs, in_tree, in_avals, static_args, params
            )

        self._trace = _trace

    # ----- registration API -----

    def _clear_cache(self):
        self._trace.cache_clear()

    def definv(self, inv_fun: Callable) -> Callable:
        """Define inverse; log-det defaults to NaN."""

        def inv_and_ld(*a, **k):
            return inv_fun(*a, **k), jnp.nan

        self.inv_fun = inv_fun
        self.inv_fun_and_log_det = inv_and_ld
        self._clear_cache()
        return inv_and_ld

    def definv_and_logdet(self, inv_fun_and_log_det: Callable) -> Callable:
        """Define inverse that returns (x, logdet)."""
        self.inv_fun_and_log_det = inv_fun_and_log_det
        if self.inv_fun is None:
            self.inv_fun = lambda *a, **k: inv_fun_and_log_det(*a, **k)[0]
        self._clear_cache()
        return inv_fun_and_log_det

    def defvalue_and_logdet(self, value_and_logdet_fun: Callable) -> Callable:
        """Optionally expose forward value_and_logdet."""
        self.value_and_logdet_fun = value_and_logdet_fun
        return value_and_logdet_fun

    # ----- Python-level helpers -----

    def inv(self, *a, **k):
        if self.inv_fun is None:
            raise AttributeError("Inverse not defined. Use definv/definv_and_logdet.")
        return self.inv_fun(*a, **k)

    def inv_and_logdet(self, *a, **k):
        if self.inv_fun_and_log_det is None:
            raise AttributeError("Inverse+logdet not defined.")
        return self.inv_fun_and_log_det(*a, **k)

    def value_and_logdet(self, *a, **k):
        if self.value_and_logdet_fun is None:
            raise AttributeError("value_and_logdet not defined.")
        return self.value_and_logdet_fun(*a, **k)

    # ----- core tracing helper -----

    def _build_jaxprs_for_signature(
        self,
        dyn_idxs: Tuple[int, ...],
        in_tree,
        in_avals: Tuple[Any, ...],
        static_args: Tuple[Any, ...],
        params: dict,
    ):
        """Given a call signature (no concrete values), build lazy forward jaxpr + inverse thunk.

        Both forward and inverse jaxprs are now lazy - they are only traced when
        actually needed (during impl/abstract_eval for forward, or during inverse
        interpretation for inverse).
        """
        dyn_idxs = tuple(dyn_idxs)
        static_idxs = self.static_argnums or ()
        static_args = tuple(static_args)

        n_args = len(dyn_idxs) + len(static_idxs)
        if len(static_args) != len(static_idxs):
            raise ValueError("Mismatch between static_argnums and static_args.")
        if set(dyn_idxs) | set(static_idxs) != set(range(n_args)):
            raise ValueError("dyn_idxs/static_argnums must partition positional args.")

        static_pos_to_val = {idx: static_args[i] for i, idx in enumerate(static_idxs)}

        def assemble_args(dyn_args_tuple):
            # dyn_args_tuple has len == len(dyn_idxs), in that order.
            full = [None] * n_args
            # place statics
            for idx, val in static_pos_to_val.items():
                full[idx] = val
            # place dynamics
            for j, v in enumerate(dyn_args_tuple):
                full[dyn_idxs[j]] = v
            return tuple(full)

        # Inverted arg must be dynamic
        if self.inv_argnum not in dyn_idxs:
            raise ValueError(
                "inv_argnum must refer to a non-static positional argument."
            )
        inv_argnum_dyn_index = dyn_idxs.index(self.inv_argnum)

        # ---------- lazy forward jaxpr ----------
        def forward_jaxpr_thunk():
            def f_dyn(*dyn_args_tuple):
                return self.fun(*assemble_args(dyn_args_tuple), **params)

            fun_name = getattr(self.fun, "__name__", str(self.fun))
            forward_jaxpr, out_avals, out_tree = trace_to_closed_jaxpr(
                f_dyn,
                in_tree=in_tree,
                in_avals=in_avals,
                debug_name="custom_inverse forward",
                const_context=f"custom_inverse forward ({fun_name})",
            )

            if not out_avals:
                raise ValueError("custom_inverse expects at least one output.")

            return forward_jaxpr, out_avals, out_tree

        lazy_forward = Lazy(forward_jaxpr_thunk)

        # ---------- lazy inverse jaxpr thunk ----------
        def inverse_jaxpr_thunk():
            if self.inv_fun_and_log_det is None:
                raise ValueError(
                    "Inverse JAXPR requested, but no inverse was registered via "
                    "definv/definv_and_logdet."
                )

            # Force forward to get out_avals for inverse input signature
            _, out_avals, _ = lazy_forward.get()

            def inv_dyn(*dyn_args_tuple):
                full = assemble_args(dyn_args_tuple)
                return self.inv_fun_and_log_det(*full, **params)

            inv_in_avals = list(in_avals)
            inv_in_avals[inv_argnum_dyn_index] = out_avals[0]
            inv_name = getattr(
                self.inv_fun_and_log_det, "__name__", "custom_inverse inverse"
            )
            inverse_jaxpr, _, _ = trace_to_closed_jaxpr(
                inv_dyn,
                in_tree=in_tree,
                in_avals=tuple(inv_in_avals),
                debug_name="custom_inverse inverse",
                const_context=f"custom_inverse inverse ({inv_name})",
            )
            return inverse_jaxpr

        lazy_inverse = LazyClosedJaxpr(inverse_jaxpr_thunk)

        return lazy_forward, inv_argnum_dyn_index, lazy_inverse

    # ----- transformed call -----

    def __call__(self, *args, **kwargs) -> Any:
        name = getattr(self.fun, "__name__", str(self.fun))
        if self.inv_fun_and_log_det is None:
            raise AttributeError(
                f"No inverse defined for custom_inverse function {name}; "
                f"use definv or definv_and_logdet first."
            )

        # Fast path: no tracers -> plain Python
        if not has_tracer((args, kwargs)):
            return self.fun(*args, **kwargs)

        # Enforce hashable kwargs (by assumption)
        params_items = tuple(
            sorted((k, ensure_hashable(v, f"kwargs['{k}']")) for k, v in kwargs.items())
        )

        n_args = len(args)
        static_idxs = self.static_argnums or ()
        dyn_idxs = tuple(i for i in range(n_args) if i not in static_idxs)

        static_args = tuple(
            ensure_hashable(args[i], f"static arg {i}") for i in static_idxs
        )
        dyn_args = tuple(args[i] for i in dyn_idxs)

        # Flatten dynamic args & abstract
        args_flat, in_tree = tree_flatten(dyn_args)
        in_avals = tuple(map(shaped_abstractify, args_flat))

        # Lookup / build lazy jaxprs
        lazy_forward, inv_argnum_dyn_index, lazy_inverse = self._trace(
            dyn_idxs,
            in_tree,
            in_avals,
            static_args,
            params_items,
        )

        # Emit the primitive with lazy forward and inverse jaxprs
        out_flat = custom_inverse_call_p.bind(
            *args_flat,
            lazy_forward=lazy_forward,
            inverse_jaxpr_thunk=lazy_inverse,
            in_tree=in_tree,
            inv_argnum=inv_argnum_dyn_index,
        )

        # Get out_tree from lazy_forward (forces evaluation if needed for output structure)
        _, _, out_tree = lazy_forward.get()
        return tree_unflatten(out_tree, out_flat)


# ---------------------------------------------------------------------------
# Primitive definition
# ---------------------------------------------------------------------------

custom_inverse_call_p = Primitive("custom_inverse_call_p")
custom_inverse_call_p.multiple_results = True

# The captured forward jaxpr is stored behind a ``Lazy`` wrapper that
# ``jaxpr_has_prim_requiring_devices`` cannot traverse.  See
# ``mark_primitive_requires_devices`` in ``call_primitive`` for details.
from probjax.core.custom_primitives.call_primitive import (  # noqa: E402
    mark_primitive_requires_devices,
)

mark_primitive_requires_devices(custom_inverse_call_p)


def _resolve_forward_jaxpr(lazy_forward) -> ClosedJaxpr:
    """Resolve lazy forward jaxpr to ClosedJaxpr, evaluating if needed."""
    if isinstance(lazy_forward, Lazy):
        forward_jaxpr, _, _ = lazy_forward.get()
        return forward_jaxpr
    # Backwards compatibility: if already a ClosedJaxpr
    return lazy_forward


def custom_inverse_call_impl(
    *args,
    lazy_forward,
    inverse_jaxpr_thunk,
    in_tree,
    inv_argnum: int,
):
    forward_jaxpr = _resolve_forward_jaxpr(lazy_forward)
    _ = parse_custom_inverse_call_params({
        "forward_jaxpr": forward_jaxpr,
        "inverse_jaxpr_thunk": inverse_jaxpr_thunk,
        "in_tree": in_tree,
        "inv_argnum": inv_argnum,
    })
    return call_impl(
        *args,
        forward_jaxpr=forward_jaxpr,
        single_result=False,
        inverse_jaxpr_thunk=inverse_jaxpr_thunk,
        in_tree=in_tree,
        inv_argnum=inv_argnum,
    )


def custom_inverse_call_abstract_eval(
    *avals,
    lazy_forward,
    inverse_jaxpr_thunk,
    in_tree,
    inv_argnum: int,
):
    forward_jaxpr = _resolve_forward_jaxpr(lazy_forward)
    _ = parse_custom_inverse_call_params({
        "forward_jaxpr": forward_jaxpr,
        "inverse_jaxpr_thunk": inverse_jaxpr_thunk,
        "in_tree": in_tree,
        "inv_argnum": inv_argnum,
    })
    return call_abstract_eval(
        *avals,
        forward_jaxpr=forward_jaxpr,
        single_result=False,
        inverse_jaxpr_thunk=inverse_jaxpr_thunk,
        in_tree=in_tree,
        inv_argnum=inv_argnum,
    )


custom_inverse_call_p.def_impl(custom_inverse_call_impl)
custom_inverse_call_p.def_abstract_eval(custom_inverse_call_abstract_eval)


def custom_inverse_call_lowering(
    ctx, *mlir_args, lazy_forward, inverse_jaxpr_thunk, in_tree, inv_argnum
):
    forward_jaxpr = _resolve_forward_jaxpr(lazy_forward)
    _ = parse_custom_inverse_call_params({
        "forward_jaxpr": forward_jaxpr,
        "inverse_jaxpr_thunk": inverse_jaxpr_thunk,
        "in_tree": in_tree,
        "inv_argnum": inv_argnum,
    })
    return call_lowering(
        ctx,
        *mlir_args,
        forward_jaxpr=forward_jaxpr,
        lowering_name="custom_inverse_forward",
        inverse_jaxpr_thunk=inverse_jaxpr_thunk,
        in_tree=in_tree,
        inv_argnum=inv_argnum,
    )


mlir.register_lowering(custom_inverse_call_p, custom_inverse_call_lowering)


# ---------------------------------------------------------------------------
# JVP: forward-only; inverse thunk errors if asked to invert this
# ---------------------------------------------------------------------------


def custom_inverse_jvp(
    primals, tangents, lazy_forward, inverse_jaxpr_thunk, in_tree, inv_argnum
):
    """JVP rule: evaluate the JVP'ed forward jaxpr inline.

    Re-emitting the primitive with a JVP'ed jaxpr would require a matching
    ``partial_eval`` rule for reverse-mode ``jax.grad`` to split the call
    into known primal outputs and unknown tangent outputs. Evaluating the
    JVP jaxpr directly as ordinary JAX ops lets standard partial evaluation
    see through to the underlying ops, so ``jax.grad`` and
    ``jax.linearize`` work without a custom partial_eval registration.

    The trade-off is that the JVP'ed call is no longer a single primitive
    — ``inverse(grad(f))`` cannot be taken. The inverse of the *original*
    (un-transformed) forward is unchanged, so ``grad(inverse(f))`` and
    ``inverse(f)`` continue to work as before.
    """
    del inverse_jaxpr_thunk, in_tree, inv_argnum

    forward_jaxpr = _resolve_forward_jaxpr(lazy_forward)
    return jvp_from_forward_jaxpr(forward_jaxpr, primals, tangents)


ad.primitive_jvps[custom_inverse_call_p] = custom_inverse_jvp


# ---------------------------------------------------------------------------
# vmap: re-emit primitive, keep inverse lazy (fixes inverse(vmap(f)))
# ---------------------------------------------------------------------------


def batch_custom_inverse_call(
    axis_data, args, in_dims, lazy_forward, inverse_jaxpr_thunk, in_tree, inv_argnum
):
    forward_jaxpr = _resolve_forward_jaxpr(lazy_forward)

    # Move mapped axes to front; track which args are batched.
    new_args, in_axes, any_batched = move_mapped_axes_to_front(args, in_dims)
    if not any_batched:
        # Nothing to do; keep primitive as-is.
        outs = custom_inverse_call_p.bind(
            *new_args,
            lazy_forward=forward_jaxpr,
            inverse_jaxpr_thunk=inverse_jaxpr_thunk,
            in_tree=in_tree,
            inv_argnum=inv_argnum,
        )
        out_dims = [batching.not_mapped] * len(outs)
        return outs, out_dims

    # Batch the forward jaxpr.
    batched_forward_jaxpr, out_axes = batch_closed_jaxpr(
        forward_jaxpr, axis_data, in_axes
    )

    # Lazily batch the inverse jaxpr only if someone actually asks for it.
    def batched_inverse_thunk():
        inv_cj = (
            inverse_jaxpr_thunk()
            if callable(inverse_jaxpr_thunk)
            else inverse_jaxpr_thunk
        )
        if inv_cj is None:
            raise ValueError("No inverse defined for batched custom_inverse call.")
        batched_inv_cj, _ = batch_closed_jaxpr(inv_cj, axis_data, in_axes)
        return batched_inv_cj

    # Re-emit the primitive so inverse() still sees it.
    outs = custom_inverse_call_p.bind(
        *new_args,
        lazy_forward=batched_forward_jaxpr,
        inverse_jaxpr_thunk=batched_inverse_thunk,
        in_tree=in_tree,
        inv_argnum=inv_argnum,
    )

    return outs, list(out_axes)


batching.fancy_primitive_batchers[custom_inverse_call_p] = batch_custom_inverse_call


# ---------------------------------------------------------------------------
# Transpose: reuse forward, inverse thunk -> error
# ---------------------------------------------------------------------------


def custom_inverse_transpose(
    cts, *args, lazy_forward, inverse_jaxpr_thunk, in_tree, inv_argnum
):
    forward_jaxpr = _resolve_forward_jaxpr(lazy_forward)

    # Transposed primitive has no sensible inverse; install failing thunk.
    del inverse_jaxpr_thunk
    err_thunk = _error_inverse_thunk(
        "Inverse of a transposed custom_inverse call is not supported."
    )
    return ad.call_transpose(
        custom_inverse_call_p,
        cts,
        *args,
        lazy_forward=forward_jaxpr,
        inverse_jaxpr_thunk=err_thunk,
        in_tree=in_tree,
        inv_argnum=inv_argnum,
    )


ad.primitive_transposes[custom_inverse_call_p] = custom_inverse_transpose

# Log-determinant inverse rules - explicit logdet formulas for common primitives.
# All rules are registered in the unified REGISTRY with Context.INVERSE_LOGDET.

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Literal

from probjax.core.interpreters.inverse.rules import (
    _BIVARIATE_INVERSES,
    _UNIVARIATE_GUARDS,
    _UNIVARIATE_INVERSES,
    _get_inverse_cost_fn,
    _max_valid,
    _min_valid,
    _pass_through,
    _values_equal,
    dot_general_left_inverse_and_logdet,
    dot_general_right_inverse_and_logdet,
    fft_inverse_type,
    invert_bitcast_convert_type,
    invert_broadcast_in_dim,
    invert_concat,
    invert_convert_element_type,
    invert_gather,
    invert_integer_pow,
    invert_reshape,
    invert_scatter,
    invert_scatter_add,
    invert_select_n,
    invert_slice,
    invert_squeeze,
    invert_transpose,
    pack_cond_values,
    parse_scan_problem,
    parse_while_problem,
    prepare_cond_branch_problem,
    prepare_cond_branches,
    read_state_values,
    scan_reverse_indices,
    solve_nested_values_and_state,
    state_from_vars,
    unpack_cond_values,
    verify_while_candidate,
)
from probjax.core.jaxpr_propagation.utils import (
    Knowness,
    ProcessingRuleFactory,
    primitive_bind_params,
)
from probjax.core.registry import (
    REGISTRY,
    Context,
    ProcessedResult,
    apply_inverse_guard,
    chain_logdet_into,
    invalid_inverse_value,
    inverse_roundtrip_valid,
    is_static_zero,
    register_bivariate_inverse_logdet,
    register_univariate_inverse_logdet,
)

INVERSE_AND_LOGABSDET_STATE_NAMESPACE = "inverse_and_logabsdet.log_dets"
_LOGABSDET_PROCESSING_RULE_FACTORY: ProcessingRuleFactory | None = None


def set_logabsdet_processing_rule_factory(factory):
    global _LOGABSDET_PROCESSING_RULE_FACTORY
    _LOGABSDET_PROCESSING_RULE_FACTORY = factory


def _make_logabsdet_processing_rule(state_namespace):
    if _LOGABSDET_PROCESSING_RULE_FACTORY is None:
        raise RuntimeError("Logabsdet processing rule factory has not been configured.")
    return _LOGABSDET_PROCESSING_RULE_FACTORY(state_namespace)


def _sum_previous_log_dets(context, outvars):
    """Sum accumulated log-dets, skipping static zeros without staging.

    Returns ``(total, nontrivial)`` like the interpreter's ``_sum_log_dets``;
    hand-written rules chain through :func:`chain_logdet_into` so a
    volume-preserving stretch stages no ``previous + 0.0`` additions.
    """
    if context is None:
        return jnp.asarray(0.0), False
    state = context.read_run_state(namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE)
    if state is None:
        return jnp.asarray(0.0), False

    total = jnp.asarray(0.0)
    nontrivial = False
    for var in outvars:
        if isinstance(var, Literal):
            continue
        term = state.get(var, 0.0)
        if is_static_zero(term):
            continue
        nontrivial = True
        total = total + jnp.asarray(term)
    return total, nontrivial


#: Primitives whose Jacobian is diagonal, so differentiating the inverse
#: elementwise is exact. The autodiff fallback is valid only for these.
#:
#: Every primitive the fallback used to serve now has an explicit rule, so this
#: is a safety net for primitives added later -- not a live code path. Anything
#: absent raises rather than returning a diagonal guess, which is what silently
#: produced wrong log-determinants for reshape, concatenate, slice and scatter.
_ELEMENTWISE_PRIMITIVES = frozenset({
    jax.lax.abs_p,
    jax.lax.acos_p,
    jax.lax.acosh_p,
    jax.lax.asin_p,
    jax.lax.asinh_p,
    jax.lax.atan_p,
    jax.lax.atanh_p,
    jax.lax.cbrt_p,
    jax.lax.cos_p,
    jax.lax.cosh_p,
    jax.lax.erf_p,
    jax.lax.erf_inv_p,
    jax.lax.exp_p,
    jax.lax.exp2_p,
    jax.lax.expm1_p,
    jax.lax.integer_pow_p,
    jax.lax.log_p,
    jax.lax.log1p_p,
    jax.lax.logistic_p,
    jax.lax.neg_p,
    jax.lax.pow_p,
    jax.lax.rsqrt_p,
    jax.lax.sin_p,
    jax.lax.sinh_p,
    jax.lax.sqrt_p,
    jax.lax.square_p,
    jax.lax.tan_p,
    jax.lax.tanh_p,
})


def is_elementwise_primitive(primitive) -> bool:
    """Whether the diagonal autodiff fallback is valid for ``primitive``."""
    return primitive in _ELEMENTWISE_PRIMITIVES


def value_and_log_det_diagonal(f):
    """Autodiff fallback: value and log-det assuming a **diagonal** Jacobian.

    Only correct for elementwise maps -- it differentiates under nested vmap, so
    for anything that moves elements around it computes a quantity that is not
    the log-determinant. Callers must gate on
    :func:`is_elementwise_primitive` first.
    """
    grad_fn = jax.value_and_grad(f)

    def log_det_fn(*args, **kwargs):
        args_arrays = [jnp.array(arg) if jnp.ndim(arg) == 0 else arg for arg in args]
        args_arrays = jnp.broadcast_arrays(*args_arrays)
        n_dim = args_arrays[0].ndim
        vmaped_grad_fn = grad_fn
        for _ in range(n_dim):
            vmaped_grad_fn = jax.vmap(vmaped_grad_fn)
        value, det = vmaped_grad_fn(*args_arrays, **kwargs)

        # No epsilon: a singular Jacobian is -inf, not log(1e-10) = -23.03,
        # which is a plausible-looking finite log-density for a point where the
        # map is not invertible at all.
        log_det = jnp.log(jnp.abs(det))
        while log_det.ndim > 0:
            log_det = jnp.sum(log_det, axis=-1)
        return value, log_det

    return log_det_fn


def inverse_and_logabsdet_state_reducer(env, eqn, state, eqn_state, context=None):
    """State reducer for accumulating log-determinants."""
    del env, eqn, context
    base_state = {} if state is None else dict(state)
    if not eqn_state:
        return base_state
    merged_state = dict(base_state)
    merged_state.update(eqn_state)
    return merged_state


# =============================================================================
# Register univariate inverse+logdet rules with explicit formulas
# =============================================================================

# exp: d/dy[log(y)] = 1/y => log|Jacobian| = -log(|y|)
register_univariate_inverse_logdet(
    jax.lax.exp_p,
    jax.lax.log_p,
    lambda out_val, in_val, params: -jnp.sum(jnp.log(jnp.abs(out_val))),
)

# log: d/dy[exp(y)] = exp(y) = y (since x = exp(y)), so the
# log|Jacobian| is log(|out|) = y_sum?
# Actually: x = exp(y), dx/dy = exp(y) = x = out, so log|det| = sum(log(|out|))
# But we want d(input)/d(output) for inverse. If forward is log, inverse is exp.
# d/dy[exp(y)] = exp(y). So log|det| = sum(y) where y = output (of forward log)
register_univariate_inverse_logdet(
    jax.lax.log_p,
    jax.lax.exp_p,
    lambda out_val, in_val, params: jnp.sum(out_val),  # sum(y) where y is log output
)

# neg: d/dy[-y] = -1 => log|Jacobian| = 0
# conj: |det| = 1 => log|Jacobian| = 0
# copy: identity => log|Jacobian| = 0
# Plain `0.0` (not `jnp.asarray(0.0)`): the chaining helper skips staging for
# Python-level zeros, so these contribute no equations at all.
register_univariate_inverse_logdet(
    jax.lax.neg_p,
    jax.lax.neg_p,
    lambda out_val, in_val, params: 0.0,
)
register_univariate_inverse_logdet(
    jax.lax.copy_p,
    jax.lax.copy_p,
    lambda out_val, in_val, params: 0.0,
)


# rev (array reversal / jnp.flip): permutation, volume-preserving => log|Jacobian| = 0
# rev is its own inverse: rev(rev(x, dims), dims) = x
@REGISTRY.rule(jax.lax.rev_p, Context.INVERSE_LOGDET)
def invert_rev_and_logdet(eqn, known_invars, known_outvars, context=None):
    del known_invars
    if known_outvars[0] is None:
        return None
    in_val = eqn.primitive.bind(*known_outvars, **eqn.params)
    updates = {}
    for var in eqn.invars:
        if not isinstance(var, Literal):
            updates[var] = 0.0
    return ProcessedResult(eqn.invars, [in_val], updates)


# Rearranging primitives (squeeze, transpose, gather-as-permutation) move every
# element exactly once, so log|Jacobian| = 0. They need an explicit rule rather
# than the generic autodiff fallback: that fallback differentiates the inverse
# elementwise under vmap, but these inverse rules need the whole array (they
# reshape or scatter into it) and fail on a scalar tracer.
def register_rearrangement_inverse_logdet(
    primitive, inverse_rule, *, strict=True, selects=False, multi_input=False
):
    """Register a zero-log-det INVERSE_LOGDET rule delegating to ``inverse_rule``.

    With ``strict``, a primitive that does not preserve the element count is not
    a bijection and its log-determinant is rejected rather than silently taken
    as zero. ``broadcast_in_dim`` opts out: it legitimately duplicates elements,
    and its inverse rule recovers the single distinct value, which contributes
    nothing to the determinant.

    ``selects`` is for primitives that move a *subset* of elements -- ``slice``,
    ``scatter``, ``select_n``. Their element counts differ by design, but they
    still scale nothing, so the factor is 1 and the contribution 0. Whether the
    dropped elements can be recovered is a question about completeness, which
    the propagation engine already tracks and reports as NaN in
    ``_materialize_inverse_targets``; answering it again here would NaN valid
    log-dets, as it did for ``concatenate`` of a sliced passthrough.
    """

    @REGISTRY.rule(primitive, Context.INVERSE_LOGDET)
    def rule(eqn, known_invars, known_outvars, context=None):
        result = inverse_rule(eqn, known_invars, known_outvars)
        if result is None:
            return None

        if multi_input:
            # concatenate builds its output from every input, so comparing only
            # the first would call a valid rearrangement lossy. Not the default:
            # gather and scatter carry index operands in invars that are not
            # data and must not be counted.
            in_size = sum(
                math.prod(var.aval.shape)
                for var in eqn.invars
                if not isinstance(var, Literal)
            )
        else:
            in_size = math.prod(eqn.invars[0].aval.shape)
        out_size = math.prod(eqn.outvars[0].aval.shape)
        if in_size != out_size and not selects:
            if strict:
                raise NotImplementedError(
                    f"log-determinant of {primitive.name} is undefined here: it "
                    f"maps {in_size} inputs to {out_size} outputs, so it is not "
                    "a rearrangement of every element."
                )
            local_logdet = jnp.asarray(jnp.nan)
        else:
            local_logdet = jnp.asarray(0.0)

        for value in result.resolved_vals:
            # A rule may hand back a Knowness wrapper rather than a bare array
            # -- concatenate does, when only some inputs are resolved -- so
            # unwrap before asking JAX about the dtype.
            if isinstance(value, Knowness):
                value = value.value
            if value is None:
                continue
            if jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact):
                local_logdet = jnp.where(
                    jnp.any(jnp.isnan(value)), jnp.asarray(jnp.nan), local_logdet
                )

        updates = {
            var: local_logdet for var in eqn.invars if not isinstance(var, Literal)
        }
        return ProcessedResult(result.resolved_vars, result.resolved_vals, updates)

    return rule


register_rearrangement_inverse_logdet(jax.lax.squeeze_p, invert_squeeze)
register_rearrangement_inverse_logdet(jax.lax.transpose_p, invert_transpose)
register_rearrangement_inverse_logdet(jax.lax.gather_p, invert_gather)
register_rearrangement_inverse_logdet(
    jax.lax.broadcast_in_dim_p, invert_broadcast_in_dim, strict=False
)

# The remaining structural primitives. Without these they fell through to the
# autodiff fallback, which differentiates elementwise under nested vmap: for
# reshape that produced a (4,4) tangent for a 4-element array and failed MLIR
# verification, and for concatenate it leaked the engine's internal partial
# knowness into a TypeError. Both are rearrangements, so log|J| = 0.
register_rearrangement_inverse_logdet(jax.lax.reshape_p, invert_reshape)
register_rearrangement_inverse_logdet(
    jax.lax.concatenate_p, invert_concat, multi_input=True
)
register_rearrangement_inverse_logdet(jax.lax.slice_p, invert_slice, selects=True)
# dynamic_slice is deliberately absent: its inverse recovers only the window,
# leaving the operand partially known, and a partially recovered input has no
# square Jacobian to take a determinant of. Registering it anyway made the
# engine re-schedule the equation forever. The fence in the log-det interpreter
# reports it by name instead.
register_rearrangement_inverse_logdet(jax.lax.scatter_p, invert_scatter, selects=True)


@REGISTRY.rule(jax.lax.scatter_add_p, Context.INVERSE_LOGDET)
def invert_scatter_add_and_logdet(eqn, known_invars, known_outvars, context=None):
    """Scatter-add is a translation of the operand: log-det zero, always."""
    del context
    result = invert_scatter_add(eqn, known_invars, known_outvars)
    if result is None:
        return None
    updates = {}
    for var in result.resolved_vars:
        if not isinstance(var, Literal):
            updates[var] = 0.0
    return ProcessedResult(result.resolved_vars, result.resolved_vals, updates)


register_rearrangement_inverse_logdet(jax.lax.select_n_p, invert_select_n, selects=True)


# FFT/IFFT: the inverse map is the swapped transform, a complex-linear map
# M = F/n (forward FFT) whose real Jacobian has |det| = n^-n, verified against
# slogdet of the explicit matrix. Forward IFFT inverts by FFT: +n log n.
# RFFT/IRFFT change the element count and decline in the inverse rule.
def _fft_inverse_logdet(out_val, in_val, params):
    n = math.prod(params["fft_lengths"])
    if n <= 1:
        return 0.0
    sign = 1.0 if jax.lax.FftType(params["fft_type"]) == jax.lax.FftType.IFFT else -1.0
    return sign * n * math.log(n)


@REGISTRY.rule(jax.lax.fft_p, Context.INVERSE_LOGDET)
def invert_fft_and_logdet(eqn, known_invars, known_outvars, context=None):
    del known_invars
    out = known_outvars[0]
    if out is None:
        return None
    inverse_type = fft_inverse_type(eqn.params["fft_type"])
    if inverse_type is None:
        return None
    _, bind_params = primitive_bind_params(
        jax.lax.fft_p,
        eqn.params,  # type: ignore[attr-defined]
    )
    in_val = jax.lax.fft_p.bind(out, **dict(bind_params, fft_type=inverse_type))
    local_logdet = _fft_inverse_logdet(out, in_val, eqn.params)
    updates = {}
    for var in eqn.invars:
        if not isinstance(var, Literal):
            updates[var] = local_logdet
    return ProcessedResult([eqn.invars[0]], [in_val], updates)


def sqrt_inverse_fn(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.pow_p.bind(x, 2.0, **params)


register_univariate_inverse_logdet(
    jax.lax.sqrt_p,
    sqrt_inverse_fn,
    lambda out_val, in_val, params: jnp.sum(jnp.log(2.0) + jnp.log(jnp.abs(out_val))),
    # Same image guard as the INVERSE rule: a negative output has no preimage,
    # and squaring it would otherwise return a finite, wrong answer.
    guard=_UNIVARIATE_GUARDS[jax.lax.sqrt_p],
)


# cbrt: x = y^3, d/dy[y^3] = 3y^2 => log|det| = sum(log(3) + 2*log(|y|))
def cbrt_inverse_fn(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.pow_p.bind(x, 3.0, **params)


register_univariate_inverse_logdet(
    jax.lax.cbrt_p,
    cbrt_inverse_fn,
    lambda out_val, in_val, params: jnp.sum(
        jnp.log(3.0) + 2.0 * jnp.log(jnp.abs(out_val))
    ),
)

# ---------------------------------------------------------------------------
# Elementwise transcendentals
# ---------------------------------------------------------------------------
#
# These previously fell through to ``value_and_log_det_diagonal``, which
# differentiates the inverse numerically. That is valid for an elementwise map
# but strictly worse than the closed form: near a boundary it loses accuracy
# (0.18 nats on tanh at 1e-7 from the edge) and it floors a singular Jacobian at
# log(1e-10) instead of -inf.
#
# Each entry is log|d(inverse)/d(out)|, summed. ``out_val`` is the forward map's
# output -- the point the inverse is evaluated at.

_SQRT_PI = float(np.sqrt(np.pi))


def _register_elementwise(forward_prim, inverse_prim_or_fn, logdet, **kwargs):
    register_univariate_inverse_logdet(
        forward_prim, inverse_prim_or_fn, logdet, **kwargs
    )


# sin/cos: inverse asin/acos, both with |d/dy| = 1/sqrt(1 - y^2)
_register_elementwise(
    jax.lax.sin_p,
    _UNIVARIATE_INVERSES[jax.lax.sin_p],
    lambda out_val, in_val, params: -0.5 * jnp.sum(jnp.log(1.0 - out_val**2)),
)
_register_elementwise(
    jax.lax.cos_p,
    _UNIVARIATE_INVERSES[jax.lax.cos_p],
    lambda out_val, in_val, params: -0.5 * jnp.sum(jnp.log(1.0 - out_val**2)),
)
# tan: inverse atan, d/dy atan = 1/(1 + y^2)
_register_elementwise(
    jax.lax.tan_p,
    _UNIVARIATE_INVERSES[jax.lax.tan_p],
    lambda out_val, in_val, params: -jnp.sum(jnp.log1p(out_val**2)),
)
# asin: inverse sin, d/dy sin = cos(y)
_register_elementwise(
    jax.lax.asin_p,
    _UNIVARIATE_INVERSES[jax.lax.asin_p],
    lambda out_val, in_val, params: jnp.sum(jnp.log(jnp.abs(jnp.cos(out_val)))),
    guard=_UNIVARIATE_GUARDS[jax.lax.asin_p],
)
# acos: inverse cos, d/dy cos = -sin(y)
_register_elementwise(
    jax.lax.acos_p,
    _UNIVARIATE_INVERSES[jax.lax.acos_p],
    lambda out_val, in_val, params: jnp.sum(jnp.log(jnp.abs(jnp.sin(out_val)))),
    guard=_UNIVARIATE_GUARDS[jax.lax.acos_p],
)
# atan: inverse tan, d/dy tan = 1 + tan(y)^2
_register_elementwise(
    jax.lax.atan_p,
    _UNIVARIATE_INVERSES[jax.lax.atan_p],
    lambda out_val, in_val, params: jnp.sum(jnp.log1p(jnp.tan(out_val) ** 2)),
    guard=_UNIVARIATE_GUARDS[jax.lax.atan_p],
)
# sinh: inverse asinh, d/dy asinh = 1/sqrt(1 + y^2)
_register_elementwise(
    jax.lax.sinh_p,
    _UNIVARIATE_INVERSES[jax.lax.sinh_p],
    lambda out_val, in_val, params: -0.5 * jnp.sum(jnp.log1p(out_val**2)),
)
# cosh: inverse acosh, d/dy acosh = 1/sqrt(y^2 - 1)
_register_elementwise(
    jax.lax.cosh_p,
    _UNIVARIATE_INVERSES[jax.lax.cosh_p],
    lambda out_val, in_val, params: -0.5 * jnp.sum(jnp.log(out_val**2 - 1.0)),
)
# asinh: inverse sinh, d/dy sinh = cosh(y)
_register_elementwise(
    jax.lax.asinh_p,
    _UNIVARIATE_INVERSES[jax.lax.asinh_p],
    lambda out_val, in_val, params: jnp.sum(jnp.log(jnp.cosh(out_val))),
)
# acosh: inverse cosh, d/dy cosh = sinh(y)
_register_elementwise(
    jax.lax.acosh_p,
    _UNIVARIATE_INVERSES[jax.lax.acosh_p],
    lambda out_val, in_val, params: jnp.sum(jnp.log(jnp.abs(jnp.sinh(out_val)))),
    guard=_UNIVARIATE_GUARDS[jax.lax.acosh_p],
)
# atanh: inverse tanh, d/dy tanh = 1 - tanh(y)^2
_register_elementwise(
    jax.lax.atanh_p,
    _UNIVARIATE_INVERSES[jax.lax.atanh_p],
    lambda out_val, in_val, params: jnp.sum(jnp.log1p(-(jnp.tanh(out_val) ** 2))),
)
# erf: inverse erf_inv, d/dy erf_inv = (sqrt(pi)/2) * exp(erf_inv(y)^2), and
# in_val is exactly erf_inv(out_val).
_register_elementwise(
    jax.lax.erf_p,
    _UNIVARIATE_INVERSES[jax.lax.erf_p],
    lambda out_val, in_val, params: jnp.sum(jnp.log(_SQRT_PI / 2.0) + in_val**2),
)
# erf_inv: inverse erf, d/dy erf = (2/sqrt(pi)) * exp(-y^2)
_register_elementwise(
    jax.lax.erf_inv_p,
    _UNIVARIATE_INVERSES[jax.lax.erf_inv_p],
    lambda out_val, in_val, params: jnp.sum(jnp.log(2.0 / _SQRT_PI) - out_val**2),
)
# rsqrt: x = y^-2, d/dy = -2 y^-3
_register_elementwise(
    jax.lax.rsqrt_p,
    _UNIVARIATE_INVERSES[jax.lax.rsqrt_p],
    lambda out_val, in_val, params: jnp.sum(
        jnp.log(2.0) - 3.0 * jnp.log(jnp.abs(out_val))
    ),
    guard=_UNIVARIATE_GUARDS[jax.lax.rsqrt_p],
)


# integer_pow: x = y^(1/n), d/dy = (1/n) y^(1/n - 1)
@REGISTRY.rule(jax.lax.integer_pow_p, Context.INVERSE_LOGDET)
def invert_integer_pow_and_logdet(eqn, known_invars, known_outvars, context=None):
    result = invert_integer_pow(eqn, known_invars, known_outvars)
    if result is None:
        return None
    exponent = eqn.params["y"]
    out_val = jnp.asarray(known_outvars[0])
    if exponent == 0:
        local = jnp.asarray(jnp.nan)
    else:
        inv_n = 1.0 / exponent
        local = jnp.sum(
            jnp.log(jnp.abs(inv_n)) + (inv_n - 1.0) * jnp.log(jnp.abs(out_val))
        )
    previous, prev_nontrivial = _sum_previous_log_dets(context, eqn.outvars)
    updates = {}
    for var in result.resolved_vars:
        if not isinstance(var, Literal):
            chain_logdet_into(updates, var, previous, prev_nontrivial, local)
    return ProcessedResult(result.resolved_vars, result.resolved_vals, updates)


# exp2: x = log2(y), d/dy = 1/(y ln2)
_register_elementwise(
    jax.lax.exp2_p,
    _UNIVARIATE_INVERSES[jax.lax.exp2_p],
    lambda out_val, in_val, params: (
        -jnp.sum(jnp.log(jnp.abs(out_val)) + jnp.log(jnp.log(2.0)))
    ),
)


# pow (x ** y): solving for the base is a power, solving for the exponent is a
# ratio of logs.
def _pow_left_logdet(out_val, result, other, params):
    # base = out ** (1/e); d/d(out) = (1/e) out ** (1/e - 1)
    inv_e = 1.0 / jnp.asarray(other)
    return jnp.sum(jnp.log(jnp.abs(inv_e)) + (inv_e - 1.0) * jnp.log(jnp.abs(out_val)))


def _pow_right_logdet(out_val, result, other, params):
    # exponent = log(out)/log(base); d/d(out) = 1/(out * log(base))
    return -jnp.sum(jnp.log(jnp.abs(out_val)) + jnp.log(jnp.abs(jnp.log(other))))


register_bivariate_inverse_logdet(
    jax.lax.pow_p,
    _BIVARIATE_INVERSES[jax.lax.pow_p][0],
    _BIVARIATE_INVERSES[jax.lax.pow_p][1],
    _pow_left_logdet,
    _pow_right_logdet,
)


# real / imag discard a component outright: the inverse rules already return
# NaN, and no determinant is defined for a projection.
def _register_projection_logdet(primitive, inverse_rule):
    @REGISTRY.rule(primitive, Context.INVERSE_LOGDET)
    def rule(eqn, known_invars, known_outvars, context=None):
        result = inverse_rule(eqn, known_invars, known_outvars)
        if result is None:
            return None
        updates = {
            var: jnp.asarray(jnp.nan)
            for var in result.resolved_vars
            if not isinstance(var, Literal)
        }
        return ProcessedResult(result.resolved_vars, result.resolved_vals, updates)

    return rule


for _projection in (jax.lax.real_p, jax.lax.imag_p):
    _register_projection_logdet(_projection, REGISTRY.get(_projection, Context.INVERSE))


# ---------------------------------------------------------------------------
# Dtype and complex reinterpretations: volume preserving, log|J| = 0
# ---------------------------------------------------------------------------
register_rearrangement_inverse_logdet(
    jax.lax.convert_element_type_p, invert_convert_element_type
)
# conj is its own inverse and |det| = 1.
_register_elementwise(
    jax.lax.conj_p,
    _UNIVARIATE_INVERSES[jax.lax.conj_p],
    lambda out_val, in_val, params: 0.0,
)
register_rearrangement_inverse_logdet(
    jax.lax.bitcast_convert_type_p, invert_bitcast_convert_type, selects=True
)


# tanh/atanh: d/dy[tanh(y)] = sech^2(y) = 1 - tanh^2(y)
# Since x = tanh(y), dx/dy = 1 - x^2, so log|det| = sum(log(1 - x^2))
# For inverse (forward=tanh), out = x = tanh(y), in = y = atanh(x)
register_univariate_inverse_logdet(
    jax.lax.tanh_p,
    jax.lax.atanh_p,
    lambda out_val, in_val, params: -jnp.sum(jnp.log(1.0 - out_val**2)),
)

# logistic (sigmoid): d/dy[logit(y)] = 1/(y*(1-y))
# For inverse of logistic, input is y = sigmoid(x), output is x
# d[logit(y)]/dy = 1/(y*(1-y)), so log|det| = -sum(log(y) + log(1-y))
register_univariate_inverse_logdet(
    jax.lax.logistic_p,
    lambda x, **params: jax.lax.log_p.bind(x) - jax.lax.log1p_p.bind(-x),
    lambda out_val, in_val, params: -jnp.sum(jnp.log(out_val) + jnp.log1p(-out_val)),
)

# log1p/expm1: log1p inverse is expm1
# d/dy[expm1(y)] = exp(y) = expm1(y) + 1 = 1 + y (approximately for small y)
# Actually exp(y) exactly. So log|det| = sum(y)
register_univariate_inverse_logdet(
    jax.lax.log1p_p,
    jax.lax.expm1_p,
    lambda out_val, in_val, params: jnp.sum(out_val),
)

# expm1: inverse is log1p
# d/dy[log1p(y)] = 1/(1+y)
# For expm1 forward, out = expm1(x) = e^x - 1, inverse gives x = log1p(out)
# d[log1p(y)]/dy = 1/(1+y), so log|det| = -sum(log(1+out))
register_univariate_inverse_logdet(
    jax.lax.expm1_p,
    jax.lax.log1p_p,
    lambda out_val, in_val, params: -jnp.sum(jnp.log(1.0 + out_val)),
)


# =============================================================================
# Register bivariate inverse+logdet rules with explicit formulas
# =============================================================================


# mul: z = x * y. Solving for x: x = z/y, dx/dz = 1/y => log|det| = -sum(log|y|)
# The known operand may have been broadcast (e.g. scalar * vector); its logdet
# contribution counts once per output element.
def _mul_left_logdet(out_val, result, other, params):
    # Solving for left: left = out/right, d(left)/d(out) = 1/right
    other = jnp.broadcast_to(other, jnp.shape(out_val))
    return -jnp.sum(jnp.log(jnp.abs(other) + 1e-10))


def _mul_right_logdet(out_val, result, other, params):
    # Solving for right: right = out/left, d(right)/d(out) = 1/left
    other = jnp.broadcast_to(other, jnp.shape(out_val))
    return -jnp.sum(jnp.log(jnp.abs(other) + 1e-10))


register_bivariate_inverse_logdet(
    jax.lax.mul_p,
    jax.lax.div_p,
    jax.lax.div_p,
    _mul_left_logdet,
    _mul_right_logdet,
)


# div: z = x / y. Solving for x: x = z * y, dx/dz = y => log|det| = sum(log|y|)
# Solving for y: y = x / z, dy/dz = -x/z^2 => log|det| = sum(log|x|) - 2*sum(log|z|)
def _div_left_logdet(out_val, result, other, params):
    # Solving for left (numerator): left = out * right, d(left)/d(out) = right
    other = jnp.broadcast_to(other, jnp.shape(out_val))
    return jnp.sum(jnp.log(jnp.abs(other) + 1e-10))


def _div_right_logdet(out_val, result, other, params):
    # Solving for right (denominator): right = left / out
    # d(right)/d(out) = -left / out^2
    other = jnp.broadcast_to(other, jnp.shape(out_val))
    return jnp.sum(jnp.log(jnp.abs(other) + 1e-10)) - 2.0 * jnp.sum(
        jnp.log(jnp.abs(out_val) + 1e-10)
    )


register_bivariate_inverse_logdet(
    jax.lax.div_p,
    jax.lax.mul_p,
    lambda x, y, **params: jax.lax.div_p.bind(y, x, **params),
    _div_left_logdet,
    _div_right_logdet,
)

# add: z = x + y. Solving for either: dx/dz = 1 => log|det| = 0
# sub: z = x - y. Solving for x: x = z + y, dx/dz = 1
# Solving for y: y = x - z, dy/dz = -1 => log|det| = 0
# Plain `0.0`: the chaining helper skips staging for Python-level zeros.
register_bivariate_inverse_logdet(
    jax.lax.add_p,
    jax.lax.sub_p,
    jax.lax.sub_p,
    lambda out_val, result, other, params: 0.0,
    lambda out_val, result, other, params: 0.0,
)
register_bivariate_inverse_logdet(
    jax.lax.sub_p,
    jax.lax.add_p.bind,
    lambda x, y, **params: jax.lax.sub_p.bind(y, x, **params),
    lambda out_val, result, other, params: 0.0,
    lambda out_val, result, other, params: 0.0,
)

# max/min: on the active side the output is the input, so dx/dz = 1 there and
# the log-det is 0 -- with the same guard poisoning the clipped side to NaN.
register_bivariate_inverse_logdet(
    jax.lax.max_p,
    _pass_through,
    _pass_through,
    lambda out_val, result, other, params: 0.0,
    lambda out_val, result, other, params: 0.0,
    guard=_max_valid,
)
register_bivariate_inverse_logdet(
    jax.lax.min_p,
    _pass_through,
    _pass_through,
    lambda out_val, result, other, params: 0.0,
    lambda out_val, result, other, params: 0.0,
    guard=_min_valid,
)


# =============================================================================
# Custom inverse+logdet rules for control flow primitives
# =============================================================================


@REGISTRY.rule(jax.lax.dot_general_p, Context.INVERSE_LOGDET)
def invert_dot_general_and_logdet(eqn, known_invars, known_outvars, context=None):
    out = known_outvars[0]
    lhs, rhs = known_invars

    if out is None:
        return None

    if lhs is None and rhs is None:
        raise NotImplementedError(
            "dot_general inverse requires at least one known input"
        )

    target_index = 0 if lhs is None else 1
    missing_var = eqn.invars[target_index]
    try:
        if lhs is None:
            missing_value, log_abs_det = dot_general_left_inverse_and_logdet(
                out, rhs, **eqn.params
            )
            replayed = eqn.primitive.bind(missing_value, rhs, **eqn.params)
        else:
            missing_value, log_abs_det = dot_general_right_inverse_and_logdet(
                out, lhs, **eqn.params
            )
            replayed = eqn.primitive.bind(lhs, missing_value, **eqn.params)
        valid = jnp.all(inverse_roundtrip_valid(replayed, out))
        missing_value = apply_inverse_guard(
            missing_value,
            missing_var.aval,
            valid,
            message="dot_general output has no unique inverse",
        )
        log_abs_det = jnp.where(valid, log_abs_det, jnp.asarray(jnp.nan))
    except NotImplementedError:
        missing_value = invalid_inverse_value(
            missing_var.aval,
            message="dot_general shape does not define a unique inverse",
        )
        log_abs_det = jnp.asarray(jnp.nan)

    updates = {}
    if not isinstance(missing_var, Literal):
        previous, prev_nontrivial = _sum_previous_log_dets(context, eqn.outvars)
        chain_logdet_into(updates, missing_var, previous, prev_nontrivial, log_abs_det)

    return ProcessedResult([missing_var], [missing_value], updates)


@REGISTRY.rule(jax.lax.cond_p, Context.INVERSE_LOGDET)
def invert_cond_and_logdet(eqn, known_invars, known_outvars, context=None):
    del context
    if any(out is None for out in known_outvars):
        return None
    branch_index, branches = prepare_cond_branches(eqn, known_invars)
    target_sub_vars, target_outer_vars, _, _ = prepare_cond_branch_problem(
        eqn, branches[0], known_invars, known_outvars
    )

    if not target_sub_vars:
        return ProcessedResult([], [], {})

    def make_branch_solver(branch):
        def solve(packed):
            branch_inputs, branch_outputs = unpack_cond_values(
                known_invars, known_outvars, packed
            )
            branch_targets, _, known_vars, known_vals = prepare_cond_branch_problem(
                eqn, branch, branch_inputs, branch_outputs
            )
            nested_values, nested_state = solve_nested_values_and_state(
                jaxpr=branch.jaxpr,
                consts=branch.consts,
                known_vars=known_vars,
                known_vals=known_vals,
                target_vars=branch_targets,
                process_eqn=_make_logabsdet_processing_rule(
                    INVERSE_AND_LOGABSDET_STATE_NAMESPACE
                ),
                cost_fn=_get_inverse_cost_fn(),
                reducer=inverse_and_logabsdet_state_reducer,
                initial_state={},
                state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
            )
            if any(value is None for value in nested_values):
                raise NotImplementedError(
                    "cond inverse+logdet could not recover branch inputs"
                )
            nested_state = {} if nested_state is None else nested_state
            logdets = tuple(
                jnp.asarray(nested_state.get(var, 0.0)) for var in branch_targets
            )
            return tuple(nested_values), logdets

        return solve

    nested_values, nested_logdets = jax.lax.switch(
        branch_index,
        tuple(make_branch_solver(branch) for branch in branches),
        pack_cond_values(known_invars, known_outvars),
    )
    updates = {
        outer_var: logdet
        for outer_var, logdet in zip(target_outer_vars, nested_logdets, strict=False)
        if not isinstance(outer_var, Literal)
    }

    return ProcessedResult(target_outer_vars, list(nested_values), updates)


@REGISTRY.rule(jax.lax.scan_p, Context.INVERSE_LOGDET)
def invert_scan_and_logdet(eqn, known_invars, known_outvars, context=None):
    if any(out is None for out in known_outvars):
        return None
    problem = parse_scan_problem(eqn, known_invars, known_outvars)
    if problem["ys_outvars"]:
        raise NotImplementedError(
            "scan inverse+logdet currently supports carry-only scans"
        )

    missing_indices = [
        i for i, value in enumerate(problem["known_carry_in_vals"]) if value is None
    ]
    if not missing_indices:
        return ProcessedResult([], [], {})

    outer_state = {}
    if context is not None:
        state = context.read_run_state(namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE)
        if state is not None:
            outer_state = state

    current_carry = list(problem["known_carry_out_vals"])
    current_carry_logdets = [
        jnp.asarray(outer_state.get(var, 0.0))
        if not isinstance(var, Literal)
        else jnp.asarray(0.0)
        for var in problem["carry_outvars"]
    ]

    for index in scan_reverse_indices(problem["length"], problem["reverse"]):
        x_step_vals = [value[index] for value in problem["known_xs_vals"]]

        known_vars = (
            list(problem["body_const_invars"])
            + list(problem["body_x_invars"])
            + list(problem["body_carry_outvars"])
        )
        known_vals = list(problem["known_const_vals"]) + x_step_vals + current_carry

        initial_state = state_from_vars(
            problem["body_carry_outvars"],
            current_carry_logdets,
        )

        recovered_carry, nested_state = solve_nested_values_and_state(
            jaxpr=problem["body"].jaxpr,
            consts=problem["body"].consts,
            known_vars=known_vars,
            known_vals=known_vals,
            target_vars=problem["body_carry_invars"],
            process_eqn=_make_logabsdet_processing_rule(
                INVERSE_AND_LOGABSDET_STATE_NAMESPACE
            ),
            cost_fn=_get_inverse_cost_fn(),
            reducer=inverse_and_logabsdet_state_reducer,
            initial_state=initial_state,
            state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
        )

        if any(v is None for v in recovered_carry):
            raise NotImplementedError("scan inverse+logdet could not recover carry")

        next_logdets = read_state_values(
            nested_state,
            problem["body_carry_invars"],
            default_factory=lambda: jnp.asarray(0.0),
        )
        current_carry = list(recovered_carry)
        current_carry_logdets = next_logdets

    output_vars = [problem["carry_invars"][i] for i in missing_indices]
    output_vals = [current_carry[i] for i in missing_indices]

    updates = {}
    for i in missing_indices:
        var = problem["carry_invars"][i]
        if not isinstance(var, Literal):
            updates[var] = current_carry_logdets[i]

    return ProcessedResult(output_vars, output_vals, updates)


@REGISTRY.rule(jax.lax.while_p, Context.INVERSE_LOGDET)
def invert_while_and_logdet(eqn, known_invars, known_outvars, context=None):
    if any(out is None for out in known_outvars):
        return None
    problem = parse_while_problem(eqn, known_invars, known_outvars)

    missing_indices = [
        i for i, value in enumerate(problem["known_state_in_vals"]) if value is None
    ]
    if not missing_indices:
        return ProcessedResult([], [], {})

    anchor_indices = [
        i for i, value in enumerate(problem["known_state_in_vals"]) if value is not None
    ]
    if not anchor_indices:
        raise NotImplementedError(
            "while inverse+logdet requires at least one known input state"
        )

    outer_state = {}
    if context is not None:
        state = context.read_run_state(namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE)
        if state is not None:
            outer_state = state

    current_state = list(problem["known_state_out_vals"])
    current_state_logdets = [
        jnp.asarray(outer_state.get(var, 0.0))
        if not isinstance(var, Literal)
        else jnp.asarray(0.0)
        for var in problem["state_outvars"]
    ]

    max_reverse_steps = 10000
    recovered_state = None
    recovered_state_logdets = None

    for reverse_steps in range(max_reverse_steps + 1):
        anchors_match = all(
            _values_equal(current_state[i], problem["known_state_in_vals"][i])
            for i in anchor_indices
        )
        if anchors_match and verify_while_candidate(
            problem, current_state, reverse_steps
        ):
            recovered_state = list(current_state)
            recovered_state_logdets = list(current_state_logdets)
            break

        if reverse_steps == max_reverse_steps:
            break

        known_vars = list(problem["body_const_invars"]) + list(
            problem["body"].jaxpr.outvars
        )
        known_vals = list(problem["known_body_const_vals"]) + current_state

        initial_state = state_from_vars(
            problem["body"].jaxpr.outvars,
            current_state_logdets,
        )

        previous_state, nested_state = solve_nested_values_and_state(
            jaxpr=problem["body"].jaxpr,
            consts=problem["body"].consts,
            known_vars=known_vars,
            known_vals=known_vals,
            target_vars=problem["body_state_invars"],
            process_eqn=_make_logabsdet_processing_rule(
                INVERSE_AND_LOGABSDET_STATE_NAMESPACE
            ),
            cost_fn=_get_inverse_cost_fn(),
            reducer=inverse_and_logabsdet_state_reducer,
            initial_state=initial_state,
            state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
        )

        if any(v is None for v in previous_state):
            raise NotImplementedError("while inverse+logdet could not recover state")

        next_logdets = read_state_values(
            nested_state,
            problem["body_state_invars"],
            default_factory=lambda: jnp.asarray(0.0),
        )

        current_state = list(previous_state)
        current_state_logdets = next_logdets

    if recovered_state is None or recovered_state_logdets is None:
        raise NotImplementedError(
            "while inverse+logdet could not determine loop iteration count"
        )

    output_vars = [problem["state_invars"][i] for i in missing_indices]
    output_vals = [recovered_state[i] for i in missing_indices]

    updates = {}
    for i in missing_indices:
        var = problem["state_invars"][i]
        if not isinstance(var, Literal):
            updates[var] = recovered_state_logdets[i]

    return ProcessedResult(output_vars, output_vals, updates)

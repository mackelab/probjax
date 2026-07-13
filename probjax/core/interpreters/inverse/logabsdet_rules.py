# Log-determinant inverse rules - explicit logdet formulas for common primitives.
# All rules are registered in the unified REGISTRY with Context.INVERSE_LOGDET.

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.extend.core import Literal

from probjax.core.interpreters.inverse.rules import (
    _get_inverse_cost_fn,
    dot_general_left_inverse_and_logdet,
    dot_general_right_inverse_and_logdet,
    parse_scan_problem,
    parse_while_problem,
    prepare_cond_branch_problem,
    read_state_values,
    scan_reverse_indices,
    select_cond_branch_jaxpr,
    solve_nested_values_and_state,
    state_from_vars,
    verify_while_candidate,
    _values_equal,
)
from probjax.core.jaxpr_propagation.utils import ProcessingRuleFactory
from probjax.core.registry import (
    Context,
    ProcessedResult,
    REGISTRY,
    register_univariate_inverse_logdet,
    register_bivariate_inverse_logdet,
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
    if context is None:
        return jnp.asarray(0.0)
    state = context.read_run_state(namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE)
    if state is None:
        return jnp.asarray(0.0)

    total = jnp.asarray(0.0)
    for var in outvars:
        if isinstance(var, Literal):
            continue
        total = total + jnp.asarray(state.get(var, 0.0))
    return total


def value_and_log_det_diagonal(f):
    """Autodiff fallback for computing value and log-det of the Jacobian diagonal."""
    grad_fn = jax.value_and_grad(f)

    def log_det_fn(*args, **kwargs):
        args_arrays = [jnp.array(arg) if jnp.ndim(arg) == 0 else arg for arg in args]
        args_arrays = jnp.broadcast_arrays(*args_arrays)
        n_dim = args_arrays[0].ndim
        vmaped_grad_fn = grad_fn
        for _ in range(n_dim):
            vmaped_grad_fn = jax.vmap(vmaped_grad_fn)
        value, det = vmaped_grad_fn(*args_arrays, **kwargs)

        log_det = jnp.log(jnp.abs(det) + 1e-10)
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

# log: d/dy[exp(y)] = exp(y) = y (since x = exp(y)) => log|Jacobian| = log(|out|) = y_sum?
# Actually: x = exp(y), dx/dy = exp(y) = x = out, so log|det| = sum(log(|out|))
# But we want d(input)/d(output) for inverse. If forward is log, inverse is exp.
# d/dy[exp(y)] = exp(y). So log|det| = sum(y) where y = output (of forward log)
register_univariate_inverse_logdet(
    jax.lax.log_p,
    jax.lax.exp_p,
    lambda out_val, in_val, params: jnp.sum(out_val),  # sum(y) where y is log output
)

# neg: d/dy[-y] = -1 => log|Jacobian| = 0
register_univariate_inverse_logdet(
    jax.lax.neg_p,
    jax.lax.neg_p,
    lambda out_val, in_val, params: jnp.asarray(0.0),
)

# copy: identity, log|Jacobian| = 0
register_univariate_inverse_logdet(
    jax.lax.copy_p,
    jax.lax.copy_p,
    lambda out_val, in_val, params: jnp.asarray(0.0),
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
            updates[var] = jnp.asarray(0.0)
    return ProcessedResult(eqn.invars, [in_val], updates)


# sqrt: x = y^2, d/dy[y^2] = 2y => log|det| = sum(log(2) + log(|y|))
def sqrt_inverse_fn(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.pow_p.bind(x, 2.0, **params)


register_univariate_inverse_logdet(
    jax.lax.sqrt_p,
    sqrt_inverse_fn,
    lambda out_val, in_val, params: jnp.sum(jnp.log(2.0) + jnp.log(jnp.abs(out_val))),
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

# tanh/atanh: d/dy[tanh(y)] = sech^2(y) = 1 - tanh^2(y)
# Since x = tanh(y), dx/dy = 1 - x^2, so log|det| = sum(log(1 - x^2))
# For inverse (forward=tanh), out = x = tanh(y), in = y = atanh(x)
register_univariate_inverse_logdet(
    jax.lax.tanh_p,
    jax.lax.atanh_p,
    lambda out_val, in_val, params: jnp.sum(jnp.log(1.0 - out_val**2 + 1e-10)),
)

# logistic (sigmoid): d/dy[logit(y)] = 1/(y*(1-y))
# For inverse of logistic, input is y = sigmoid(x), output is x
# d[logit(y)]/dy = 1/(y*(1-y)), so log|det| = -sum(log(y) + log(1-y))
register_univariate_inverse_logdet(
    jax.lax.logistic_p,
    lambda x, **params: jax.lax.log_p.bind(x) - jax.lax.log1p_p.bind(-x),
    lambda out_val, in_val, params: (
        -jnp.sum(jnp.log(out_val + 1e-10) + jnp.log(1.0 - out_val + 1e-10))
    ),
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
register_bivariate_inverse_logdet(
    jax.lax.add_p,
    jax.lax.sub_p,
    jax.lax.sub_p,
    lambda out_val, result, other, params: jnp.asarray(0.0),
    lambda out_val, result, other, params: jnp.asarray(0.0),
)

# sub: z = x - y. Solving for x: x = z + y, dx/dz = 1
# Solving for y: y = x - z, dy/dz = -1 => log|det| = 0
register_bivariate_inverse_logdet(
    jax.lax.sub_p,
    jax.lax.add_p.bind,
    lambda x, y, **params: jax.lax.sub_p.bind(y, x, **params),
    lambda out_val, result, other, params: jnp.asarray(0.0),
    lambda out_val, result, other, params: jnp.asarray(0.0),
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

    if lhs is None:
        missing_var = eqn.invars[0]
        missing_value, log_abs_det = dot_general_left_inverse_and_logdet(
            out, rhs, **eqn.params
        )
    else:
        missing_var = eqn.invars[1]
        missing_value, log_abs_det = dot_general_right_inverse_and_logdet(
            out, lhs, **eqn.params
        )

    updates = {}
    if not isinstance(missing_var, Literal):
        previous = _sum_previous_log_dets(context, eqn.outvars)
        updates[missing_var] = previous + jnp.asarray(log_abs_det)

    return ProcessedResult([missing_var], [missing_value], updates)


@REGISTRY.rule(jax.lax.cond_p, Context.INVERSE_LOGDET)
def invert_cond_and_logdet(eqn, known_invars, known_outvars, context=None):
    if any(out is None for out in known_outvars):
        return None
    branch = select_cond_branch_jaxpr(eqn, known_invars)
    target_sub_vars, target_outer_vars, known_vars, known_vals = (
        prepare_cond_branch_problem(
            eqn,
            branch,
            known_invars,
            known_outvars,
        )
    )

    if not target_sub_vars:
        return ProcessedResult([], [], {})

    outer_state = {}
    if context is not None:
        state = context.read_run_state(namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE)
        if state is not None:
            outer_state = state

    initial_state = {}
    for branch_outvar, outer_outvar in zip(
        branch.jaxpr.outvars, eqn.outvars, strict=False
    ):
        if isinstance(outer_outvar, Literal):
            continue
        if outer_outvar in outer_state:
            initial_state[branch_outvar] = outer_state[outer_outvar]

    nested_values, nested_state = solve_nested_values_and_state(
        jaxpr=branch.jaxpr,
        consts=branch.consts,
        known_vars=known_vars,
        known_vals=known_vals,
        target_vars=target_sub_vars,
        process_eqn=_make_logabsdet_processing_rule(
            INVERSE_AND_LOGABSDET_STATE_NAMESPACE
        ),
        cost_fn=_get_inverse_cost_fn(),
        reducer=inverse_and_logabsdet_state_reducer,
        initial_state=initial_state,
        state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
    )

    if any(v is None for v in nested_values):
        raise NotImplementedError("cond inverse+logdet could not recover branch inputs")

    nested_state = {} if nested_state is None else nested_state
    updates = {}
    for sub_var, outer_var in zip(target_sub_vars, target_outer_vars, strict=False):
        if isinstance(outer_var, Literal):
            continue
        if sub_var not in nested_state:
            raise NotImplementedError(
                "cond inverse+logdet missing branch logdet update"
            )
        updates[outer_var] = nested_state[sub_var]

    return ProcessedResult(target_outer_vars, nested_values, updates)


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

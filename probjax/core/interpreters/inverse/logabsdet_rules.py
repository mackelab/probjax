import jax
import jax.numpy as jnp
from jax.extend.core import Literal

from probjax.core.interpreters.inverse.registry import inverse_cost_fn
from probjax.core.interpreters.inverse.rules import (
    dot_general_left_inverse_and_logdet,
    dot_general_right_inverse_and_logdet,
    read_state_values,
    parse_scan_problem,
    parse_while_problem,
    prepare_cond_branch_problem,
    scan_reverse_indices,
    select_cond_branch_jaxpr,
    solve_nested_values_and_state,
    state_from_vars,
    verify_while_candidate,
    _values_equal,
)
from probjax.core.jaxpr_propagation.utils import ProcessingRuleFactory

CUSTOM_INVERSE_AND_LOG_DET_RULES = {}
INVERSE_AND_LOGABSDET_STATE_NAMESPACE = "inverse_and_logabsdet.log_dets"
_LOGABSDET_PROCESSING_RULE_FACTORY: ProcessingRuleFactory | None = None


def register_inverse_and_log_det_rule(key):
    def decorator(func):
        nonlocal key
        CUSTOM_INVERSE_AND_LOG_DET_RULES[key] = func
        return func

    return decorator


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


@register_inverse_and_log_det_rule(jax.lax.dot_general_p)
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

    return [missing_var], [missing_value], updates


@register_inverse_and_log_det_rule(jax.lax.cond_p)
def invert_cond_and_logdet(eqn, known_invars, known_outvars, context=None):
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
        return [], [], {}

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
        cost_fn=inverse_cost_fn,
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

    return target_outer_vars, nested_values, updates


@register_inverse_and_log_det_rule(jax.lax.scan_p)
def invert_scan_and_logdet(eqn, known_invars, known_outvars, context=None):
    problem = parse_scan_problem(eqn, known_invars, known_outvars)
    if problem["ys_outvars"]:
        raise NotImplementedError(
            "scan inverse+logdet currently supports carry-only scans"
        )

    missing_indices = [
        i for i, value in enumerate(problem["known_carry_in_vals"]) if value is None
    ]
    if not missing_indices:
        return [], [], {}

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
            cost_fn=inverse_cost_fn,
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

    return output_vars, output_vals, updates


@register_inverse_and_log_det_rule(jax.lax.while_p)
def invert_while_and_logdet(eqn, known_invars, known_outvars, context=None):
    problem = parse_while_problem(eqn, known_invars, known_outvars)

    missing_indices = [
        i for i, value in enumerate(problem["known_state_in_vals"]) if value is None
    ]
    if not missing_indices:
        return [], [], {}

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
            cost_fn=inverse_cost_fn,
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

    return output_vars, output_vals, updates


def value_and_log_det_diagonal(f):
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
    del env, eqn, context
    base_state = {} if state is None else dict(state)
    if not eqn_state:
        return base_state
    merged_state = dict(base_state)
    merged_state.update(eqn_state)
    return merged_state

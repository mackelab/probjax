import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax._src.util import safe_map


CUSTOM_INVERSE_PROCESSING_RULES = {}


def register_inverse_rule(key):
    def decorator(func):
        nonlocal key
        CUSTOM_INVERSE_PROCESSING_RULES[key] = func
        return func

    return decorator


def select_cond_branch_jaxpr(eqn, known_invars):
    if "branches" not in eqn.params:
        raise NotImplementedError("cond inverse requires branch jaxprs")
    if not known_invars:
        raise NotImplementedError("cond inverse requires known inputs")

    branch_index = known_invars[0]
    if branch_index is None:
        raise NotImplementedError("cond inverse requires known branch index")

    index_array = jnp.asarray(branch_index)
    if index_array.shape != ():
        raise NotImplementedError("cond inverse requires scalar branch index")

    index = int(index_array.item())
    branches = eqn.params["branches"]
    if index < 0 or index >= len(branches):
        raise NotImplementedError("cond inverse branch index out of range")

    return branches[index]


def prepare_cond_branch_problem(eqn, branch, known_invars, known_outvars):
    known_operands = known_invars[1:]
    branch_invars = branch.jaxpr.invars
    branch_outvars = branch.jaxpr.outvars

    if len(branch_invars) != len(known_operands):
        raise NotImplementedError("cond inverse operand arity mismatch")
    if len(branch_outvars) != len(known_outvars):
        raise NotImplementedError("cond inverse output arity mismatch")

    known_vars = []
    known_vals = []
    missing_indices = []

    for i, value in enumerate(known_operands):
        if value is None:
            missing_indices.append(i)
        else:
            known_vars.append(branch_invars[i])
            known_vals.append(value)

    for branch_outvar, outval in zip(branch_outvars, known_outvars, strict=False):
        if outval is None:
            raise NotImplementedError("cond inverse requires known outputs")
        known_vars.append(branch_outvar)
        known_vals.append(outval)

    target_sub_vars = [branch_invars[i] for i in missing_indices]
    target_outer_vars = [eqn.invars[1 + i] for i in missing_indices]

    return target_sub_vars, target_outer_vars, known_vars, known_vals


def scan_reverse_indices(length, reverse):
    if reverse:
        return range(length)
    return range(length - 1, -1, -1)


def parse_scan_problem(eqn, known_invars, known_outvars):
    params = eqn.params
    if "jaxpr" not in params:
        raise NotImplementedError("scan inverse requires nested jaxpr")

    body = params["jaxpr"]
    num_consts = params["num_consts"]
    num_carry = params["num_carry"]
    length = params["length"]
    reverse = params["reverse"]

    const_invars = list(eqn.invars[:num_consts])
    carry_invars = list(eqn.invars[num_consts : num_consts + num_carry])
    xs_invars = list(eqn.invars[num_consts + num_carry :])

    known_const_vals = list(known_invars[:num_consts])
    known_carry_in_vals = list(known_invars[num_consts : num_consts + num_carry])
    known_xs_vals = list(known_invars[num_consts + num_carry :])

    carry_outvars = list(eqn.outvars[:num_carry])
    ys_outvars = list(eqn.outvars[num_carry:])
    known_carry_out_vals = list(known_outvars[:num_carry])
    known_ys_out_vals = list(known_outvars[num_carry:])

    if any(v is None for v in known_const_vals):
        raise NotImplementedError("scan inverse requires known scan constants")
    if any(v is None for v in known_xs_vals):
        raise NotImplementedError("scan inverse requires known scan sequence inputs")
    if any(v is None for v in known_carry_out_vals):
        raise NotImplementedError("scan inverse requires known final carry")
    if any(v is None for v in known_ys_out_vals):
        raise NotImplementedError("scan inverse requires known scan outputs")

    body_invars = list(body.jaxpr.invars)
    body_outvars = list(body.jaxpr.outvars)

    expected_invars = num_consts + num_carry + len(xs_invars)
    expected_outvars = num_carry + len(ys_outvars)
    if len(body_invars) != expected_invars:
        raise NotImplementedError("scan inverse body input arity mismatch")
    if len(body_outvars) != expected_outvars:
        raise NotImplementedError("scan inverse body output arity mismatch")

    body_const_invars = body_invars[:num_consts]
    body_carry_invars = body_invars[num_consts : num_consts + num_carry]
    body_x_invars = body_invars[num_consts + num_carry :]
    body_carry_outvars = body_outvars[:num_carry]
    body_y_outvars = body_outvars[num_carry:]

    for value in known_xs_vals:
        if jnp.asarray(value).shape[0] != length:
            raise NotImplementedError("scan inverse sequence length mismatch")
    for value in known_ys_out_vals:
        if jnp.asarray(value).shape[0] != length:
            raise NotImplementedError("scan inverse output length mismatch")

    return {
        "body": body,
        "num_consts": num_consts,
        "num_carry": num_carry,
        "length": length,
        "reverse": reverse,
        "const_invars": const_invars,
        "carry_invars": carry_invars,
        "xs_invars": xs_invars,
        "carry_outvars": carry_outvars,
        "ys_outvars": ys_outvars,
        "known_const_vals": known_const_vals,
        "known_carry_in_vals": known_carry_in_vals,
        "known_xs_vals": known_xs_vals,
        "known_carry_out_vals": known_carry_out_vals,
        "known_ys_out_vals": known_ys_out_vals,
        "body_const_invars": body_const_invars,
        "body_carry_invars": body_carry_invars,
        "body_x_invars": body_x_invars,
        "body_carry_outvars": body_carry_outvars,
        "body_y_outvars": body_y_outvars,
    }


def parse_while_problem(eqn, known_invars, known_outvars):
    params = eqn.params
    if "body_jaxpr" not in params or "cond_jaxpr" not in params:
        raise NotImplementedError("while inverse requires body and cond jaxprs")

    cond_nconsts = params["cond_nconsts"]
    body_nconsts = params["body_nconsts"]
    state_offset = cond_nconsts + body_nconsts

    cond_const_invars = list(eqn.invars[:cond_nconsts])
    body_const_invars = list(eqn.invars[cond_nconsts:state_offset])
    state_invars = list(eqn.invars[state_offset:])

    known_cond_const_vals = list(known_invars[:cond_nconsts])
    known_body_const_vals = list(known_invars[cond_nconsts:state_offset])
    known_state_in_vals = list(known_invars[state_offset:])
    known_state_out_vals = list(known_outvars)

    if any(v is None for v in known_cond_const_vals):
        raise NotImplementedError("while inverse requires known cond constants")
    if any(v is None for v in known_body_const_vals):
        raise NotImplementedError("while inverse requires known body constants")
    if any(v is None for v in known_state_out_vals):
        raise NotImplementedError("while inverse requires known final loop state")

    cond = params["cond_jaxpr"]
    body = params["body_jaxpr"]
    n_state = len(state_invars)
    if len(eqn.outvars) != n_state:
        raise NotImplementedError("while inverse state arity mismatch")

    if len(cond.jaxpr.invars) != cond_nconsts + n_state:
        raise NotImplementedError("while inverse cond input arity mismatch")
    if len(cond.jaxpr.outvars) != 1:
        raise NotImplementedError("while inverse cond must return one predicate")
    if len(body.jaxpr.invars) != body_nconsts + n_state:
        raise NotImplementedError("while inverse body input arity mismatch")
    if len(body.jaxpr.outvars) != n_state:
        raise NotImplementedError("while inverse body output arity mismatch")

    body_state_invars = list(body.jaxpr.invars[body_nconsts:])

    return {
        "cond": cond,
        "body": body,
        "cond_nconsts": cond_nconsts,
        "body_nconsts": body_nconsts,
        "state_offset": state_offset,
        "cond_const_invars": cond_const_invars,
        "body_const_invars": body_const_invars,
        "state_invars": state_invars,
        "state_outvars": list(eqn.outvars),
        "known_cond_const_vals": known_cond_const_vals,
        "known_body_const_vals": known_body_const_vals,
        "known_state_in_vals": known_state_in_vals,
        "known_state_out_vals": known_state_out_vals,
        "body_state_invars": body_state_invars,
    }


def _as_python_bool(value):
    array = jnp.asarray(value)
    if array.shape != ():
        raise NotImplementedError("while inverse requires scalar cond predicate")
    return bool(array.item())


def _values_equal(left, right):
    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != right_array.shape:
        return False

    dtype = jnp.result_type(left_array.dtype, right_array.dtype)
    if jnp.issubdtype(dtype, jnp.inexact):
        return bool(jnp.allclose(left_array, right_array, atol=1e-6, rtol=1e-6))
    return bool(jnp.array_equal(left_array, right_array))


def _evaluate_closed_jaxpr(closed_jaxpr, *args):
    return jax_core.eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args)


def verify_while_candidate(problem, candidate_state, num_steps):
    cond = problem["cond"]
    body = problem["body"]
    cond_consts = problem["known_cond_const_vals"]
    body_consts = problem["known_body_const_vals"]
    expected_out_state = problem["known_state_out_vals"]

    state = list(candidate_state)
    for _ in range(num_steps):
        predicate_out = _evaluate_closed_jaxpr(cond, *cond_consts, *state)
        if not _as_python_bool(predicate_out[0]):
            return False
        state = list(_evaluate_closed_jaxpr(body, *body_consts, *state))

    predicate_out = _evaluate_closed_jaxpr(cond, *cond_consts, *state)
    if _as_python_bool(predicate_out[0]):
        return False

    return all(
        _values_equal(value, expected)
        for value, expected in zip(state, expected_out_state, strict=False)
    )


@register_inverse_rule(jax.lax.concatenate_p)
def invert_concat(eqn, known_invars, known_outvars):
    dim = eqn.params["dimension"]
    out = known_outvars[0]
    in_avals = safe_map(lambda x: x.aval, eqn.invars)
    split_dimensions = safe_map(lambda x: x.shape[dim], in_avals)
    split_indices = np.cumsum(split_dimensions)[:-1].tolist()

    in_vars = jnp.split(
        out,
        split_indices,
        axis=dim,
    )
    return eqn.invars, in_vars


@register_inverse_rule(jax.lax.squeeze_p)
def invert_squeeze(eqn, known_invars, known_outvars):
    in_shape = eqn.invars[0].aval.shape
    out = known_outvars[0]
    return [eqn.invars[0]], [out.reshape(in_shape)]


@register_inverse_rule(jax.lax.broadcast_in_dim_p)
def invert_broadcast_in_dim(eqn, known_invars, known_outvars):
    in_shape = eqn.invars[0].aval.shape
    out = known_outvars[0]
    return [eqn.invars[0]], [out.reshape(in_shape)]


@register_inverse_rule(jax.lax.rev_p)
def invert_rev(eqn, known_invars, known_outvars):
    return eqn.invars, [eqn.primitive.bind(*known_outvars, **eqn.params)]


@register_inverse_rule(jax.lax.gather_p)
def invert_gather(eqn, known_invars, known_outvars):
    input, index = known_invars
    out = known_outvars[0]

    if input is None:
        input_aval = eqn.invars[0].aval
        input = jnp.zeros(input_aval.shape, input_aval.dtype)

    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive.get_bind_params(params)
    del subfuns

    gather_numdim = bind_params["dimension_numbers"]
    scatter_numdim = jax.lax.ScatterDimensionNumbers(
        gather_numdim.offset_dims,
        gather_numdim.collapsed_slice_dims,
        gather_numdim.collapsed_slice_dims,
    )

    out = out.reshape(eqn.outvars[0].aval.shape)
    input = jax.lax.scatter(input, index, out, scatter_numdim)

    return [eqn.invars[0]], [input]


@register_inverse_rule(jax.lax.scatter_p)
def invert_scatter(eqn, known_invars, known_outvars):
    index = known_invars[1]
    assert index is not None, "Cannot invert scatter without index!"

    out = known_outvars[0]

    scatter_numdim = eqn.params["dimension_numbers"]
    gather_numdim = jax.lax.GatherDimensionNumbers(
        scatter_numdim.update_window_dims,
        scatter_numdim.inserted_window_dims,
        scatter_numdim.scatter_dims_to_operand_dims,
    )

    operand_ndim = out.ndim
    slice_sizes = [1] * operand_ndim

    collapsed_dims = set(gather_numdim.collapsed_slice_dims)
    window_operand_dims = [i for i in range(operand_ndim) if i not in collapsed_dims]
    update_shape = eqn.invars[2].aval.shape
    update_window_dims = tuple(sorted(scatter_numdim.update_window_dims))

    for operand_dim, update_dim in zip(
        window_operand_dims,
        update_window_dims,
        strict=False,
    ):
        if update_dim < len(update_shape):
            slice_sizes[operand_dim] = update_shape[update_dim]

    update_val = jax.lax.gather(out, index, gather_numdim, tuple(slice_sizes))
    update_val = jnp.reshape(update_val, eqn.invars[2].aval.shape)

    return [eqn.invars[0], eqn.invars[2]], [out, update_val]


@register_inverse_rule(jax.lax.select_n_p)
def invert_select_n(eqn, known_invars, known_outvars):
    out = known_outvars[0]
    which = known_invars[:1]
    cases = known_invars[1:]

    in_avals = safe_map(lambda x: x.aval, eqn.invars[1:])

    new_cases = []
    for c, aval in zip(cases, in_avals, strict=False):
        if c is None:
            new_cases.append(out.astype(aval.dtype))
        else:
            new_cases.append(c)

    return (
        eqn.invars,
        which + new_cases,
    )


@register_inverse_rule(jax.lax.reshape_p)
def invert_reshape(eqn, _, known_outvars):
    out = known_outvars[0]
    in_aval = eqn.invars[0].aval
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive.get_bind_params(params)
    bind_params["new_sizes"] = in_aval.shape
    return [eqn.invars[0]], [primitive.bind(out, *subfuns, **bind_params)]


@register_inverse_rule(jax.lax.convert_element_type_p)
def invert_convert_element_type(eqn, known_invars, known_outvars):
    out = known_outvars[0]
    in_aval = eqn.invars[0].aval
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive.get_bind_params(params)
    bind_params["new_dtype"] = in_aval.dtype
    return [eqn.invars[0]], [primitive.bind(out, *subfuns, **bind_params)]


@register_inverse_rule(jax.lax.bitcast_convert_type_p)
def invert_bitcast_convert_type(eqn, known_invars, known_outvars):
    out = known_outvars[0]
    in_aval = eqn.invars[0].aval
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive.get_bind_params(params)
    bind_params["new_dtype"] = in_aval.dtype
    return [eqn.invars[0]], [primitive.bind(out, *subfuns, **bind_params)]


@register_inverse_rule(jax.lax.transpose_p)
def invert_transpose(eqn, known_invars, known_outvars):
    out = known_outvars[0]
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive.get_bind_params(params)
    permutation = bind_params["permutation"]
    inverse_permutation = tuple(np.argsort(permutation).tolist())
    bind_params["permutation"] = inverse_permutation
    return [eqn.invars[0]], [primitive.bind(out, *subfuns, **bind_params)]


@register_inverse_rule(jax.lax.slice_p)
def invert_slice(eqn, known_invars, known_outvars):
    input = known_invars[0]
    start_index = eqn.params["start_indices"]
    invar = eqn.invars[0]
    in_aval = invar.aval
    if input is None:
        input = jnp.zeros(in_aval.shape, in_aval.dtype)
    out1 = known_outvars[0]
    while out1.ndim < input.ndim:
        out1 = jnp.expand_dims(out1, axis=-1)
    new_input = jax.lax.dynamic_update_slice(input, out1, start_index)
    return [invar], [new_input]


@register_inverse_rule(jax.lax.dynamic_slice_p)
def invert_dynamic_slice(eqn, known_invars, known_outvars):
    input = known_invars[0]
    start_indices = known_invars[1:]

    invar = eqn.invars[0]
    in_aval = invar.aval

    if input is None:
        input = jnp.full(in_aval.shape, jnp.nan, dtype=in_aval.dtype)

    out1 = known_outvars[0]
    new_input = jax.lax.dynamic_update_slice(input, out1, start_indices)

    return [invar], [new_input]


@register_inverse_rule(jax.lax.split_p)
def invert_split(eqn, known_invars, known_outvars):
    params = eqn.params
    invar = eqn.invars[0]
    assert len(known_outvars) == len(eqn.outvars), (
        "Cannot invert split without all outputs!"
    )
    axis = params["axis"]
    sizes = params["sizes"]

    assert all(
        o.shape[axis] == s for o, s in zip(known_outvars, sizes, strict=False)
    ), "Output shapes do not match the sizes!"

    return [invar], [jnp.concatenate(known_outvars, axis=axis)]


@register_inverse_rule(jax.lax.cond_p)
def invert_cond(eqn, known_invars, known_outvars, context=None):
    del context
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
        return [], []

    from probjax.core.interpreters.inverse.interpreter import InverseProcessingRule
    from probjax.core.interpreters.inverse.registry import inverse_cost_fn
    from probjax.core.jaxpr_propagation.propagate import propagate

    target_vals = propagate(
        branch.jaxpr,
        branch.consts,
        known_vars,
        known_vals,
        target_sub_vars,
        process_eqn=InverseProcessingRule(),
        cost_fn=inverse_cost_fn,
        process_all_eqns=True,
    )

    if any(v is None for v in target_vals):
        raise NotImplementedError("cond inverse could not recover branch inputs")

    return target_outer_vars, target_vals


@register_inverse_rule(jax.lax.scan_p)
def invert_scan(eqn, known_invars, known_outvars, context=None):
    del context
    problem = parse_scan_problem(eqn, known_invars, known_outvars)

    missing_indices = [
        i for i, value in enumerate(problem["known_carry_in_vals"]) if value is None
    ]
    if not missing_indices:
        return [], []

    from probjax.core.interpreters.inverse.interpreter import InverseProcessingRule
    from probjax.core.interpreters.inverse.registry import inverse_cost_fn
    from probjax.core.jaxpr_propagation.propagate import propagate

    current_carry = list(problem["known_carry_out_vals"])
    for index in scan_reverse_indices(problem["length"], problem["reverse"]):
        x_step_vals = [value[index] for value in problem["known_xs_vals"]]
        y_step_vals = [value[index] for value in problem["known_ys_out_vals"]]

        known_vars = (
            list(problem["body_const_invars"])
            + list(problem["body_x_invars"])
            + list(problem["body_carry_outvars"])
            + list(problem["body_y_outvars"])
        )
        known_vals = (
            list(problem["known_const_vals"])
            + x_step_vals
            + current_carry
            + y_step_vals
        )

        recovered_carry = propagate(
            problem["body"].jaxpr,
            problem["body"].consts,
            known_vars,
            known_vals,
            problem["body_carry_invars"],
            process_eqn=InverseProcessingRule(),
            cost_fn=inverse_cost_fn,
            process_all_eqns=True,
        )
        if any(v is None for v in recovered_carry):
            raise NotImplementedError("scan inverse could not recover carry inputs")
        current_carry = list(recovered_carry)

    output_vars = [problem["carry_invars"][i] for i in missing_indices]
    output_vals = [current_carry[i] for i in missing_indices]
    return output_vars, output_vals


@register_inverse_rule(jax.lax.while_p)
def invert_while(eqn, known_invars, known_outvars, context=None):
    del context
    problem = parse_while_problem(eqn, known_invars, known_outvars)

    missing_indices = [
        i for i, value in enumerate(problem["known_state_in_vals"]) if value is None
    ]
    if not missing_indices:
        return [], []

    anchor_indices = [
        i for i, value in enumerate(problem["known_state_in_vals"]) if value is not None
    ]
    if not anchor_indices:
        raise NotImplementedError(
            "while inverse requires at least one known input state"
        )

    from probjax.core.interpreters.inverse.interpreter import InverseProcessingRule
    from probjax.core.interpreters.inverse.registry import inverse_cost_fn
    from probjax.core.jaxpr_propagation.propagate import propagate

    max_reverse_steps = 10000
    current_state = list(problem["known_state_out_vals"])
    recovered_state = None

    for reverse_steps in range(max_reverse_steps + 1):
        anchors_match = all(
            _values_equal(current_state[i], problem["known_state_in_vals"][i])
            for i in anchor_indices
        )
        if anchors_match and verify_while_candidate(
            problem, current_state, reverse_steps
        ):
            recovered_state = list(current_state)
            break

        if reverse_steps == max_reverse_steps:
            break

        known_vars = list(problem["body_const_invars"]) + list(
            problem["body"].jaxpr.outvars
        )
        known_vals = list(problem["known_body_const_vals"]) + current_state
        previous_state = propagate(
            problem["body"].jaxpr,
            problem["body"].consts,
            known_vars,
            known_vals,
            problem["body_state_invars"],
            process_eqn=InverseProcessingRule(),
            cost_fn=inverse_cost_fn,
            process_all_eqns=True,
        )
        if any(v is None for v in previous_state):
            raise NotImplementedError("while inverse could not recover previous state")
        current_state = list(previous_state)

    if recovered_state is None:
        raise NotImplementedError(
            "while inverse could not determine loop iteration count"
        )

    output_vars = [problem["state_invars"][i] for i in missing_indices]
    output_vals = [recovered_state[i] for i in missing_indices]
    return output_vars, output_vals

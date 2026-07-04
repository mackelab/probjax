# Consolidated inverse rules module: unary, binary, and tensor/custom rules.
# All rules are registered in the unified REGISTRY.

from __future__ import annotations

import importlib
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core
from jax._src.util import safe_map
from jax.extend.core import Literal

from probjax.core.jaxpr_propagation import propagate
from probjax.core.jaxpr_propagation.utils import (
    CostFunction,
    Knowness,
    ProcessingRuleFactory,
    primitive_bind_params,
)
from probjax.core.registry import (
    Context,
    ProcessedResult,
    REGISTRY,
    register_bivariate_inverse,
    register_univariate_inverse,
)

# =============================================================================
# pjit primitive import (varies by JAX version)
# =============================================================================

try:
    pjit_mod = importlib.import_module("jax.experimental.pjit")
except ImportError:
    pjit_mod = None

pjit_p = getattr(cast(Any, pjit_mod), "pjit_p", None)
if pjit_p is None:
    # JAX 0.7+
    from jax._src.pjit import jit_p as pjit_p

# =============================================================================
# Helper functions for inverse computation
# =============================================================================


def integer_pow_inverse(x, **params):
    y = params.pop("y")
    return jax.lax.pow_p.bind(x, 1 / y, **params)


def logit(x, **params):
    return jax.lax.log_p.bind(x) - jax.lax.log1p_p.bind(-x)  # type: ignore


def sqrt_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.pow_p.bind(x, 2.0, **params)


def rsqrt_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return 1.0 / jax.lax.pow_p.bind(x, 2.0, **params)


def cbrt_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.pow_p.bind(x, 3.0, **params)


def asin_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.asin_p.bind(x, **params)


def acos_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.acos_p.bind(x, **params)


def atan_inverse(x, **params):
    params = dict(params)
    params.pop("accuracy", None)
    return jax.lax.atan_p.bind(x, **params)


def sin_inverse(x, **params):
    params = dict(params)
    params.setdefault("accuracy", None)
    return jax.lax.sin_p.bind(x, **params)


def cos_inverse(x, **params):
    params = dict(params)
    params.setdefault("accuracy", None)
    return jax.lax.cos_p.bind(x, **params)


def tan_inverse(x, **params):
    params = dict(params)
    params.setdefault("accuracy", None)
    return jax.lax.tan_p.bind(x, **params)


# =============================================================================
# Register univariate inverse rules
# =============================================================================

# Map of forward primitive -> inverse function/primitive
_UNIVARIATE_INVERSES = {
    jax.lax.sin_p: asin_inverse,
    jax.lax.asin_p: sin_inverse,
    jax.lax.cos_p: acos_inverse,
    jax.lax.acos_p: cos_inverse,
    jax.lax.tan_p: atan_inverse,
    jax.lax.atan_p: tan_inverse,
    jax.lax.tanh_p: jax.lax.atanh_p,
    jax.lax.atanh_p: jax.lax.tanh_p,
    jax.lax.sinh_p: jax.lax.asinh_p,
    jax.lax.asinh_p: jax.lax.sinh_p,
    jax.lax.cosh_p: jax.lax.acosh_p,
    jax.lax.acosh_p: jax.lax.cosh_p,
    jax.lax.exp_p: jax.lax.log_p,
    jax.lax.exp2_p: lambda x, **params: jnp.log2(x),
    jax.lax.log_p: jax.lax.exp_p,
    jax.lax.sqrt_p: sqrt_inverse,
    jax.lax.rsqrt_p: rsqrt_inverse,
    jax.lax.cbrt_p: cbrt_inverse,
    jax.lax.neg_p: jax.lax.neg_p,
    jax.lax.copy_p: jax.lax.copy_p,
    jax.lax.log1p_p: jax.lax.expm1_p,
    jax.lax.expm1_p: jax.lax.log1p_p,
    jax.lax.erf_p: jax.lax.erf_inv_p,
    jax.lax.erf_inv_p: jax.lax.erf_p,
    jax.lax.conj_p: jax.lax.conj_p,
    jax.lax.real_p: jax.lax.real_p,
    jax.lax.imag_p: jax.lax.imag_p,
    jax.lax.logistic_p: logit,
    jax.lax.integer_pow_p: integer_pow_inverse,
}

for forward_prim, inv_fn in _UNIVARIATE_INVERSES.items():
    register_univariate_inverse(forward_prim, inv_fn)


# =============================================================================
# Register bivariate inverse rules
# =============================================================================


def _inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for i, axis in enumerate(permutation):
        inverse[axis] = i
    return tuple(inverse)


def _prod(shape):
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _transpose_if_needed(x, permutation):
    if permutation == tuple(range(x.ndim)):
        return x
    return jnp.transpose(x, permutation)


def _dot_general_left_shapes(out, rhs, dimension_numbers):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contracting = tuple(lhs_contracting)
    rhs_contracting = tuple(rhs_contracting)
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)

    if len(lhs_contracting) != len(rhs_contracting):
        raise NotImplementedError(
            "dot_general inverse requires matched contracting ranks"
        )
    if len(lhs_batch) != len(rhs_batch):
        raise NotImplementedError("dot_general inverse requires matched batch ranks")

    rhs_free_axes = tuple(
        i for i in range(rhs.ndim) if i not in rhs_contracting and i not in rhs_batch
    )
    batch_shape = tuple(rhs.shape[i] for i in rhs_batch)
    rhs_free_shape = tuple(rhs.shape[i] for i in rhs_free_axes)
    lhs_contract_shape = tuple(rhs.shape[i] for i in rhs_contracting)

    out_shape = tuple(out.shape)
    min_rank = len(batch_shape) + len(rhs_free_shape)
    if len(out_shape) < min_rank:
        raise NotImplementedError(
            "dot_general output rank is incompatible with inversion"
        )
    if tuple(out_shape[: len(batch_shape)]) != batch_shape:
        raise NotImplementedError("dot_general output batch shape is incompatible")
    if rhs_free_shape and tuple(out_shape[-len(rhs_free_shape) :]) != rhs_free_shape:
        raise NotImplementedError("dot_general output free shape is incompatible")

    if rhs_free_shape:
        lhs_free_shape = out_shape[len(batch_shape) : -len(rhs_free_shape)]
    else:
        lhs_free_shape = out_shape[len(batch_shape) :]

    return {
        "lhs_contracting": lhs_contracting,
        "rhs_contracting": rhs_contracting,
        "lhs_batch": lhs_batch,
        "rhs_batch": rhs_batch,
        "rhs_free_axes": rhs_free_axes,
        "batch_shape": batch_shape,
        "lhs_free_shape": tuple(lhs_free_shape),
        "lhs_contract_shape": lhs_contract_shape,
        "rhs_free_shape": rhs_free_shape,
    }


def _dot_general_right_shapes(out, lhs, dimension_numbers):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    lhs_contracting = tuple(lhs_contracting)
    rhs_contracting = tuple(rhs_contracting)
    lhs_batch = tuple(lhs_batch)
    rhs_batch = tuple(rhs_batch)

    if len(lhs_contracting) != len(rhs_contracting):
        raise NotImplementedError(
            "dot_general inverse requires matched contracting ranks"
        )
    if len(lhs_batch) != len(rhs_batch):
        raise NotImplementedError("dot_general inverse requires matched batch ranks")

    lhs_free_axes = tuple(
        i for i in range(lhs.ndim) if i not in lhs_contracting and i not in lhs_batch
    )
    batch_shape = tuple(lhs.shape[i] for i in lhs_batch)
    lhs_free_shape = tuple(lhs.shape[i] for i in lhs_free_axes)
    rhs_contract_shape = tuple(lhs.shape[i] for i in lhs_contracting)

    out_shape = tuple(out.shape)
    prefix_shape = batch_shape + lhs_free_shape
    if len(out_shape) < len(prefix_shape):
        raise NotImplementedError(
            "dot_general output rank is incompatible with inversion"
        )
    if tuple(out_shape[: len(prefix_shape)]) != prefix_shape:
        raise NotImplementedError("dot_general output shape is incompatible")

    rhs_free_shape = tuple(out_shape[len(prefix_shape) :])

    return {
        "lhs_contracting": lhs_contracting,
        "rhs_contracting": rhs_contracting,
        "lhs_batch": lhs_batch,
        "rhs_batch": rhs_batch,
        "lhs_free_axes": lhs_free_axes,
        "batch_shape": batch_shape,
        "lhs_free_shape": lhs_free_shape,
        "rhs_contract_shape": rhs_contract_shape,
        "rhs_free_shape": rhs_free_shape,
    }


def dot_general_left_inverse_and_logdet(out, rhs, **params):
    shapes = _dot_general_left_shapes(out, rhs, params["dimension_numbers"])
    batch_shape = shapes["batch_shape"]
    lhs_free_shape = shapes["lhs_free_shape"]
    lhs_contract_shape = shapes["lhs_contract_shape"]
    rhs_free_shape = shapes["rhs_free_shape"]

    batch_size = _prod(batch_shape)
    lhs_free_size = _prod(lhs_free_shape)
    contract_size = _prod(lhs_contract_shape)
    rhs_free_size = _prod(rhs_free_shape)

    if contract_size != rhs_free_size:
        raise NotImplementedError(
            "dot_general inverse for lhs requires square contracted/free rhs dimensions"
        )

    rhs_permutation = (
        shapes["rhs_batch"] + shapes["rhs_contracting"] + shapes["rhs_free_axes"]
    )
    rhs_canon = _transpose_if_needed(rhs, rhs_permutation).reshape(
        batch_size,
        contract_size,
        rhs_free_size,
    )
    out_canon = out.reshape(batch_size, lhs_free_size, rhs_free_size)

    lhs_canon = jax.vmap(
        lambda rhs_matrix, out_matrix: jnp.linalg.solve(rhs_matrix.T, out_matrix.T).T
    )(rhs_canon, out_canon)

    _, rhs_logabsdet = jnp.linalg.slogdet(rhs_canon)
    log_abs_det = -jnp.asarray(lhs_free_size, dtype=rhs_logabsdet.dtype) * rhs_logabsdet
    log_abs_det = jnp.sum(log_abs_det)

    lhs_rank = (
        len(shapes["lhs_batch"]) + len(lhs_free_shape) + len(shapes["lhs_contracting"])
    )
    lhs_free_axes = tuple(
        i
        for i in range(lhs_rank)
        if i not in shapes["lhs_batch"] and i not in shapes["lhs_contracting"]
    )
    lhs_permutation = shapes["lhs_batch"] + lhs_free_axes + shapes["lhs_contracting"]
    lhs_canon_shape = batch_shape + lhs_free_shape + lhs_contract_shape
    lhs = lhs_canon.reshape(lhs_canon_shape)
    lhs = _transpose_if_needed(lhs, _inverse_permutation(lhs_permutation))

    return lhs, log_abs_det


def dot_general_right_inverse_and_logdet(out, lhs, **params):
    shapes = _dot_general_right_shapes(out, lhs, params["dimension_numbers"])
    batch_shape = shapes["batch_shape"]
    lhs_free_shape = shapes["lhs_free_shape"]
    rhs_contract_shape = shapes["rhs_contract_shape"]
    rhs_free_shape = shapes["rhs_free_shape"]

    batch_size = _prod(batch_shape)
    lhs_free_size = _prod(lhs_free_shape)
    contract_size = _prod(rhs_contract_shape)
    rhs_free_size = _prod(rhs_free_shape)

    if lhs_free_size != contract_size:
        raise NotImplementedError(
            "dot_general inverse for rhs requires square free/contracted lhs dimensions"
        )

    lhs_permutation = (
        shapes["lhs_batch"] + shapes["lhs_free_axes"] + shapes["lhs_contracting"]
    )
    lhs_canon = _transpose_if_needed(lhs, lhs_permutation).reshape(
        batch_size,
        lhs_free_size,
        contract_size,
    )
    out_canon = out.reshape(batch_size, lhs_free_size, rhs_free_size)

    rhs_canon = jax.vmap(
        lambda lhs_matrix, out_matrix: jnp.linalg.solve(lhs_matrix, out_matrix)
    )(lhs_canon, out_canon)

    _, lhs_logabsdet = jnp.linalg.slogdet(lhs_canon)
    log_abs_det = -jnp.asarray(rhs_free_size, dtype=lhs_logabsdet.dtype) * lhs_logabsdet
    log_abs_det = jnp.sum(log_abs_det)

    rhs_rank = (
        len(shapes["rhs_batch"]) + len(rhs_free_shape) + len(shapes["rhs_contracting"])
    )
    rhs_free_axes = tuple(
        i
        for i in range(rhs_rank)
        if i not in shapes["rhs_batch"] and i not in shapes["rhs_contracting"]
    )
    rhs_permutation = shapes["rhs_batch"] + shapes["rhs_contracting"] + rhs_free_axes
    rhs_canon_shape = batch_shape + rhs_contract_shape + rhs_free_shape
    rhs = rhs_canon.reshape(rhs_canon_shape)
    rhs = _transpose_if_needed(rhs, _inverse_permutation(rhs_permutation))

    return rhs, log_abs_det


def dot_general_left_inverse(out, rhs, **params):
    lhs, _ = dot_general_left_inverse_and_logdet(out, rhs, **params)
    return lhs


def dot_general_right_inverse(out, lhs, **params):
    rhs, _ = dot_general_right_inverse_and_logdet(out, lhs, **params)
    return rhs


# Map of binary primitive -> (left_inverse, right_inverse)
_BIVARIATE_INVERSES = {
    jax.lax.mul_p: (jax.lax.div_p, jax.lax.div_p),
    jax.lax.div_p: (
        jax.lax.mul_p,
        lambda x, y, **params: jax.lax.div_p.bind(y, x, **params),
    ),
    jax.lax.add_p: (jax.lax.sub_p, jax.lax.sub_p),
    jax.lax.sub_p: (
        jax.lax.add_p.bind,
        lambda x, y, **params: jax.lax.sub_p.bind(y, x, **params),
    ),
    jax.lax.pow_p: (
        lambda x, y, **params: jax.lax.pow_p.bind(x, 1.0 / y, **params),
        lambda x, y, **params: jax.lax.log_p.bind(x) / jax.lax.log_p.bind(y),  # type: ignore
    ),
    jax.lax.dot_general_p: (
        dot_general_left_inverse,
        dot_general_right_inverse,
    ),
}

for prim, (left_inv, right_inv) in _BIVARIATE_INVERSES.items():
    register_bivariate_inverse(prim, left_inv, right_inv)


# =============================================================================
# Nested propagation helpers
# =============================================================================

_INVERSE_PROCESSING_RULE_FACTORY: ProcessingRuleFactory | None = None
_INVERSE_COST_FN: CostFunction | None = None


def set_inverse_processing_rule_factory(factory):
    global _INVERSE_PROCESSING_RULE_FACTORY
    _INVERSE_PROCESSING_RULE_FACTORY = factory


def set_inverse_cost_fn(cost_fn):
    global _INVERSE_COST_FN
    _INVERSE_COST_FN = cost_fn


def _make_inverse_processing_rule():
    if _INVERSE_PROCESSING_RULE_FACTORY is None:
        raise RuntimeError("Inverse processing rule factory has not been configured.")
    return _INVERSE_PROCESSING_RULE_FACTORY()


def _get_inverse_cost_fn():
    if _INVERSE_COST_FN is None:
        raise RuntimeError("Inverse cost function has not been configured.")
    return _INVERSE_COST_FN


def solve_nested_values(
    *,
    jaxpr,
    consts,
    known_vars,
    known_vals,
    target_vars,
    process_eqn,
    cost_fn: CostFunction,
) -> list:
    return propagate(
        jaxpr,
        consts,
        known_vars,
        known_vals,
        target_vars,
        process_eqn=process_eqn,
        cost_fn=cost_fn,
        process_all_eqns=True,
    )


def solve_nested_values_and_state(
    *,
    jaxpr,
    consts,
    known_vars,
    known_vals,
    target_vars,
    process_eqn,
    cost_fn: CostFunction,
    reducer,
    initial_state: dict,
    state_namespace: str,
) -> tuple[list, dict | None]:
    return propagate(
        jaxpr,
        consts,
        known_vars,
        known_vals,
        target_vars,
        process_eqn=process_eqn,
        cost_fn=cost_fn,
        process_all_eqns=True,
        reducer=reducer,
        initial_state=initial_state,
        return_state=True,
        state_namespace=state_namespace,
    )


def state_from_vars(vars_, values) -> dict:
    return {
        var: value
        for var, value in zip(vars_, values, strict=False)
        if not isinstance(var, Literal)
    }


def read_state_values(state, vars_, *, default_factory) -> list:
    mapping = {} if state is None else state
    return [
        mapping.get(var, default_factory())
        if not isinstance(var, Literal)
        else default_factory()
        for var in vars_
    ]


# =============================================================================
# Custom inverse rules (for control flow primitives)
# =============================================================================


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


# =============================================================================
# Register custom inverse rules for control flow primitives
# =============================================================================


@REGISTRY.rule(jax.lax.concatenate_p, Context.INVERSE)
def invert_concat(eqn, known_invars, known_outvars):
    del known_invars
    dim = eqn.params["dimension"]
    out = known_outvars[0]
    if out is None:
        return None
    in_avals = safe_map(lambda x: x.aval, eqn.invars)
    split_dimensions = safe_map(lambda x: x.shape[dim], in_avals)
    split_indices = np.cumsum(split_dimensions)[:-1].tolist()

    in_vars = jnp.split(out, split_indices, axis=dim)
    return ProcessedResult(eqn.invars, in_vars)


@REGISTRY.rule(jax.lax.squeeze_p, Context.INVERSE)
def invert_squeeze(eqn, known_invars, known_outvars):
    del known_invars
    out = known_outvars[0]
    if out is None:
        return None
    in_shape = eqn.invars[0].aval.shape
    return ProcessedResult([eqn.invars[0]], [out.reshape(in_shape)])


@REGISTRY.rule(jax.lax.broadcast_in_dim_p, Context.INVERSE)
def invert_broadcast_in_dim(eqn, known_invars, known_outvars):
    del known_invars
    out = known_outvars[0]
    if out is None:
        return None
    in_shape = eqn.invars[0].aval.shape
    return ProcessedResult([eqn.invars[0]], [out.reshape(in_shape)])


@REGISTRY.rule(jax.lax.rev_p, Context.INVERSE)
def invert_rev(eqn, known_invars, known_outvars):
    del known_invars
    if known_outvars[0] is None:
        return None
    return ProcessedResult(
        eqn.invars, [eqn.primitive.bind(*known_outvars, **eqn.params)]
    )


@REGISTRY.rule(jax.lax.gather_p, Context.INVERSE)
def invert_gather(eqn, known_invars, known_outvars):
    input_val, index = known_invars
    out = known_outvars[0]

    if out is None:
        return None

    if input_val is None:
        input_aval = eqn.invars[0].aval
        input_val = jnp.zeros(input_aval.shape, input_aval.dtype)

    primitive = eqn.primitive
    params = eqn.params
    _, bind_params = primitive_bind_params(primitive, params)

    gather_numdim = bind_params["dimension_numbers"]
    scatter_numdim = jax.lax.ScatterDimensionNumbers(
        gather_numdim.offset_dims,
        gather_numdim.collapsed_slice_dims,
        gather_numdim.collapsed_slice_dims,
    )

    out = out.reshape(eqn.outvars[0].aval.shape)
    input_val = jax.lax.scatter(input_val, index, out, scatter_numdim)

    return ProcessedResult([eqn.invars[0]], [input_val])


@REGISTRY.rule(jax.lax.scatter_p, Context.INVERSE)
def invert_scatter(eqn, known_invars, known_outvars):
    index = known_invars[1]
    assert index is not None, "Cannot invert scatter without index!"

    out = known_outvars[0]
    if out is None:
        return None

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

    return ProcessedResult([eqn.invars[0], eqn.invars[2]], [out, update_val])


@REGISTRY.rule(jax.lax.select_n_p, Context.INVERSE)
def invert_select_n(eqn, known_invars, known_outvars):
    out = known_outvars[0]
    if out is None:
        return None
    which = known_invars[0]
    if which is None:
        # Can't invert without knowing which case was selected
        return None
    cases = known_invars[1:]
    in_avals = safe_map(lambda x: x.aval, eqn.invars[1:])

    # For select_n(which, case0, case1, ...), we have: out = cases[which]
    # When we know `which`, we can precisely set only the selected case to the output value.
    # Non-selected cases remain UNKNOWN - their values will be computed by FORWARD
    # from the now-known selected case value.
    #
    # Example: select_n(idx=1, neg(x), x) with output y
    # - We set only x = y (since idx=1 selected x)
    # - neg's forward pass then computes neg(x) = neg(y)

    # Check if which is a scalar (all elements same) or varies per element
    which_array = jnp.asarray(which)
    first_idx = which_array.flatten()[0]

    # For now, handle the simple case where which is uniform (all same index)
    # This covers the common case of jnp.select with a uniform condition
    if which_array.ndim == 0 or jnp.all(which_array == first_idx):
        # Uniform selection - only one case was selected everywhere
        selected_idx = int(first_idx)

        resolved_vars = []
        resolved_vals = []

        for i, (c, aval) in enumerate(zip(cases, in_avals, strict=False)):
            if c is None:
                if i == selected_idx:
                    # This is THE selected case - set to COMPLETE output value
                    resolved_vars.append(eqn.invars[1 + i])
                    resolved_vals.append(out.astype(aval.dtype))
                # Non-selected cases: leave as UNKNOWN (don't add to result)
            # Already-known cases: don't overwrite

        if not resolved_vars:
            return None
        return ProcessedResult(resolved_vars, resolved_vals)
    else:
        # Non-uniform selection (different indices for different elements)
        # Fall back to setting all unknown cases as PARTIAL placeholders
        new_cases = []
        resolved_vars = []
        for i, (c, aval) in enumerate(zip(cases, in_avals, strict=False)):
            if c is None:
                # This is a placeholder value - mark as PARTIAL so forward can fix it
                resolved_vars.append(eqn.invars[1 + i])
                new_cases.append(Knowness.partial(out.astype(aval.dtype)))

        if not resolved_vars:
            return None
        return ProcessedResult(resolved_vars, new_cases)


@REGISTRY.rule(jax.lax.reshape_p, Context.INVERSE)
def invert_reshape(eqn, known_invars, known_outvars):
    del known_invars
    out = known_outvars[0]
    if out is None:
        return None
    in_aval = eqn.invars[0].aval
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive_bind_params(primitive, params)
    bind_params["new_sizes"] = in_aval.shape
    return ProcessedResult(
        [eqn.invars[0]], [primitive.bind(*subfuns, out, **bind_params)]
    )


@REGISTRY.rule(jax.lax.convert_element_type_p, Context.INVERSE)
def invert_convert_element_type(eqn, known_invars, known_outvars):
    del known_invars
    out = known_outvars[0]
    if out is None:
        return None
    in_aval = eqn.invars[0].aval
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive_bind_params(primitive, params)
    bind_params["new_dtype"] = in_aval.dtype
    return ProcessedResult(
        [eqn.invars[0]], [primitive.bind(*subfuns, out, **bind_params)]
    )


@REGISTRY.rule(jax.lax.bitcast_convert_type_p, Context.INVERSE)
def invert_bitcast_convert_type(eqn, known_invars, known_outvars):
    del known_invars
    out = known_outvars[0]
    if out is None:
        return None
    in_aval = eqn.invars[0].aval
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive_bind_params(primitive, params)
    bind_params["new_dtype"] = in_aval.dtype
    return ProcessedResult(
        [eqn.invars[0]], [primitive.bind(*subfuns, out, **bind_params)]
    )


@REGISTRY.rule(jax.lax.transpose_p, Context.INVERSE)
def invert_transpose(eqn, known_invars, known_outvars):
    del known_invars
    out = known_outvars[0]
    if out is None:
        return None
    primitive = eqn.primitive
    params = eqn.params
    subfuns, bind_params = primitive_bind_params(primitive, params)
    permutation = bind_params["permutation"]
    inverse_permutation = tuple(np.argsort(permutation).tolist())
    bind_params["permutation"] = inverse_permutation
    return ProcessedResult(
        [eqn.invars[0]], [primitive.bind(*subfuns, out, **bind_params)]
    )


@REGISTRY.rule(jax.lax.slice_p, Context.INVERSE)
def invert_slice(eqn, known_invars, known_outvars):
    input_val = known_invars[0]
    out = known_outvars[0]
    if out is None:
        return None
    start_index = eqn.params["start_indices"]
    limit_indices = eqn.params["limit_indices"]
    strides = eqn.params.get("strides")
    invar = eqn.invars[0]
    in_aval = invar.aval

    # Check if this is a full slice (covers the entire input)
    # A full slice has start_index all zeros, limit_indices equal to input shape,
    # and either no strides or all strides equal to 1
    is_full_slice = (
        all(s == 0 for s in start_index)
        and tuple(limit_indices) == tuple(in_aval.shape)
        and (strides is None or all(s == 1 for s in strides))
    )

    # Track if we're creating a partial reconstruction
    is_partial = input_val is None and not is_full_slice
    if input_val is None:
        input_val = jnp.zeros(in_aval.shape, in_aval.dtype)

    out1 = out
    while out1.ndim < input_val.ndim:
        out1 = jnp.expand_dims(out1, axis=-1)
    new_input = jax.lax.dynamic_update_slice(input_val, out1, start_index)

    # Return PARTIAL if we created placeholder values, COMPLETE otherwise
    if is_partial:
        return ProcessedResult([invar], [Knowness.partial(new_input)])
    return ProcessedResult([invar], [new_input])


@REGISTRY.rule(jax.lax.dynamic_slice_p, Context.INVERSE)
def invert_dynamic_slice(eqn, known_invars, known_outvars):
    input_val = known_invars[0]
    start_indices = known_invars[1:]
    out = known_outvars[0]
    if out is None:
        return None

    invar = eqn.invars[0]
    in_aval = invar.aval

    # Track if we're creating a partial reconstruction
    is_partial = input_val is None
    if input_val is None:
        input_val = jnp.full(in_aval.shape, jnp.nan, dtype=in_aval.dtype)

    new_input = jax.lax.dynamic_update_slice(input_val, out, start_indices)

    # Return PARTIAL if we created placeholder values, COMPLETE otherwise
    if is_partial:
        return ProcessedResult([invar], [Knowness.partial(new_input)])
    return ProcessedResult([invar], [new_input])


@REGISTRY.rule(jax.lax.split_p, Context.INVERSE)
def invert_split(eqn, known_invars, known_outvars):
    del known_invars
    params = eqn.params
    invar = eqn.invars[0]
    if any(out is None for out in known_outvars):
        return None
    assert len(known_outvars) == len(eqn.outvars), (
        "Cannot invert split without all outputs!"
    )
    axis = params["axis"]
    sizes = params["sizes"]

    assert all(
        o.shape[axis] == s for o, s in zip(known_outvars, sizes, strict=False)
    ), "Output shapes do not match the sizes!"

    return ProcessedResult([invar], [jnp.concatenate(known_outvars, axis=axis)])


@REGISTRY.rule(jax.lax.split_p, Context.INVERSE_LOGDET)
def invert_split_logdet(eqn, known_invars, known_outvars):
    """Inverse of split with log-det = 0 (volume-preserving)."""
    del known_invars
    params = eqn.params
    invar = eqn.invars[0]
    if any(out is None for out in known_outvars):
        return None
    axis = params["axis"]
    concatenated = jnp.concatenate(known_outvars, axis=axis)
    return ProcessedResult([invar], [concatenated], {invar: jnp.asarray(0.0)})


@REGISTRY.rule(jax.lax.cond_p, Context.INVERSE)
def invert_cond(eqn, known_invars, known_outvars):
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
        return ProcessedResult([], [])

    target_vals = solve_nested_values(
        jaxpr=branch.jaxpr,
        consts=branch.consts,
        known_vars=known_vars,
        known_vals=known_vals,
        target_vars=target_sub_vars,
        process_eqn=_make_inverse_processing_rule(),
        cost_fn=_get_inverse_cost_fn(),
    )

    if any(v is None for v in target_vals):
        raise NotImplementedError("cond inverse could not recover branch inputs")

    return ProcessedResult(target_outer_vars, target_vals)


@REGISTRY.rule(jax.lax.scan_p, Context.INVERSE)
def invert_scan(eqn, known_invars, known_outvars):
    if any(out is None for out in known_outvars):
        return None
    problem = parse_scan_problem(eqn, known_invars, known_outvars)

    missing_indices = [
        i for i, value in enumerate(problem["known_carry_in_vals"]) if value is None
    ]
    if not missing_indices:
        return ProcessedResult([], [])

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

        recovered_carry = solve_nested_values(
            jaxpr=problem["body"].jaxpr,
            consts=problem["body"].consts,
            known_vars=known_vars,
            known_vals=known_vals,
            target_vars=problem["body_carry_invars"],
            process_eqn=_make_inverse_processing_rule(),
            cost_fn=_get_inverse_cost_fn(),
        )
        if any(v is None for v in recovered_carry):
            raise NotImplementedError("scan inverse could not recover carry inputs")
        current_carry = list(recovered_carry)

    output_vars = [problem["carry_invars"][i] for i in missing_indices]
    output_vals = [current_carry[i] for i in missing_indices]
    return ProcessedResult(output_vars, output_vals)


@REGISTRY.rule(jax.lax.while_p, Context.INVERSE)
def invert_while(eqn, known_invars, known_outvars):
    if any(out is None for out in known_outvars):
        return None
    problem = parse_while_problem(eqn, known_invars, known_outvars)

    missing_indices = [
        i for i, value in enumerate(problem["known_state_in_vals"]) if value is None
    ]
    if not missing_indices:
        return ProcessedResult([], [])

    anchor_indices = [
        i for i, value in enumerate(problem["known_state_in_vals"]) if value is not None
    ]
    if not anchor_indices:
        raise NotImplementedError(
            "while inverse requires at least one known input state"
        )

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
        previous_state = solve_nested_values(
            jaxpr=problem["body"].jaxpr,
            consts=problem["body"].consts,
            known_vars=known_vars,
            known_vals=known_vals,
            target_vars=problem["body_state_invars"],
            process_eqn=_make_inverse_processing_rule(),
            cost_fn=_get_inverse_cost_fn(),
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
    return ProcessedResult(output_vars, output_vals)

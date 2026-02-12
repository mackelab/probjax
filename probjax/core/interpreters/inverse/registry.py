import importlib
from typing import Any, cast

import jax
import jax.numpy as jnp

try:
    pjit_mod = importlib.import_module("jax.experimental.pjit")
except ImportError:
    pjit_mod = None

pjit_p = getattr(cast(Any, pjit_mod), "pjit_p", None)
if pjit_p is None:
    # JaX 0.7
    from jax._src.pjit import jit_p as pjit_p

from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p
from probjax.core.interpreters.inverse.rules_binary import BIVARIATE_INVERSE_REGISTRY
from probjax.core.interpreters.inverse.rules_tensor import (
    CUSTOM_INVERSE_PROCESSING_RULES,
    register_inverse_rule,
)
from probjax.core.interpreters.inverse.rules_unary import UNIVARIATE_INVERSE_REGISTRY


def is_univariate(eqn) -> bool:
    return len(eqn.invars) == 1 and len(eqn.outvars) == 1


def is_bivariate(eqn) -> bool:
    return (
        len(eqn.invars) == 2
        and len(eqn.outvars) == 1
        and eqn.primitive in BIVARIATE_INVERSE_REGISTRY
    )


def has_registered_inverse(eqn, known_invars, known_outvars) -> bool:
    primitive = eqn.primitive

    if primitive is custom_inverse_call_p:
        inv_argnum = eqn.params["inv_argnum"]
        cond1 = known_invars[inv_argnum] is False
        cond2 = all(known_invars[:inv_argnum]) and all(known_invars[inv_argnum + 1 :])
        return cond1 and cond2

    return (
        primitive in UNIVARIATE_INVERSE_REGISTRY
        or (primitive in BIVARIATE_INVERSE_REGISTRY and any(known_invars))
        or primitive in CUSTOM_INVERSE_PROCESSING_RULES
    )


def inverse_cost_fn(eqn, known_invars, known_outvars):
    known_invars = tuple(known_invars)
    known_outvars = tuple(known_outvars)

    if eqn.primitive is jax.lax.cond_p and (not known_invars or not known_invars[0]):
        return jnp.inf

    if eqn.primitive is jax.lax.scan_p:
        num_consts = eqn.params["num_consts"]
        num_carry = eqn.params["num_carry"]
        known_const = all(known_invars[:num_consts])
        known_xs = all(known_invars[num_consts + num_carry :])
        known_carry_out = all(known_outvars[:num_carry])
        if not (known_const and known_xs and known_carry_out):
            return jnp.inf

    if eqn.primitive is jax.lax.while_p:
        cond_nconsts = eqn.params["cond_nconsts"]
        body_nconsts = eqn.params["body_nconsts"]
        state_offset = cond_nconsts + body_nconsts
        known_cond_consts = all(known_invars[:cond_nconsts])
        known_body_consts = all(known_invars[cond_nconsts:state_offset])
        known_state_out = all(known_outvars)
        has_known_anchor = any(known_invars[state_offset:])
        if not (
            known_cond_consts
            and known_body_consts
            and known_state_out
            and has_known_anchor
        ):
            return jnp.inf

    if eqn.primitive is jax.lax.gather_p or eqn.primitive is jax.lax.slice_p:
        return 1.0
    if eqn.primitive is pjit_p and all(known_outvars):
        return 1.5
    if all(known_invars) and not any(known_outvars):
        return 0
    if all(known_outvars) and has_registered_inverse(eqn, known_invars, known_outvars):
        return 0.5
    return jnp.inf


__all__ = [
    "BIVARIATE_INVERSE_REGISTRY",
    "CUSTOM_INVERSE_PROCESSING_RULES",
    "UNIVARIATE_INVERSE_REGISTRY",
    "has_registered_inverse",
    "inverse_cost_fn",
    "is_bivariate",
    "is_univariate",
    "register_inverse_rule",
]

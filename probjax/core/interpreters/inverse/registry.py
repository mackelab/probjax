"""
Inverse registry utilities.

This module provides cost functions and utility functions for the inverse
interpreter. The actual rule registration is done in the unified REGISTRY
from probjax.core.registry.

For backwards compatibility, this module exports shim objects that wrap
the unified REGISTRY to provide the old API.
"""

import importlib
from typing import Any, cast

import jax
import jax.numpy as jnp

from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p
from probjax.core.registry import Context, REGISTRY

# pjit primitive import (varies by JAX version)
try:
    pjit_mod = importlib.import_module("jax.experimental.pjit")
except ImportError:
    pjit_mod = None

pjit_p = getattr(cast(Any, pjit_mod), "pjit_p", None)
if pjit_p is None:
    # JAX 0.7+
    from jax._src.pjit import jit_p as pjit_p


# =============================================================================
# Utility Functions
# =============================================================================


def has_registered_inverse(eqn, known_invars, known_outvars) -> bool:
    """
    Check if an equation has a registered inverse rule.

    This is used by the cost function to determine if an equation can be
    inverted given the current known values.
    """
    primitive = eqn.primitive

    # Special case for custom_inverse_call_p
    if primitive is custom_inverse_call_p:
        inv_argnum = eqn.params["inv_argnum"]
        cond1 = known_invars[inv_argnum] is False
        cond2 = all(known_invars[:inv_argnum]) and all(known_invars[inv_argnum + 1 :])
        return cond1 and cond2

    # Check unified registry
    return REGISTRY.has_rule(primitive, Context.INVERSE)


# =============================================================================
# Cost Function
# =============================================================================


def inverse_cost_fn(eqn, known_invars, known_outvars):
    """
    Cost function for inverse propagation.

    Returns the cost of processing an equation given the known values.
    Lower cost means the equation should be processed earlier.
    """
    known_invars = tuple(known_invars)
    known_outvars = tuple(known_outvars)

    # Special cases for control flow primitives
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

    # Cost based on what's known
    if eqn.primitive is jax.lax.gather_p or eqn.primitive is jax.lax.slice_p:
        return 1.0
    if eqn.primitive is pjit_p and all(known_outvars):
        return 1.5
    if all(known_invars) and not any(known_outvars):
        return 0
    if all(known_outvars) and has_registered_inverse(eqn, known_invars, known_outvars):
        return 0.5
    return jnp.inf


# =============================================================================
# Backwards Compatibility Shims
# =============================================================================


class _RegistryShim:
    """
    Shim that wraps the unified REGISTRY to provide backwards-compatible
    dict-like access for old code that uses UNIVARIATE_INVERSE_REGISTRY, etc.

    This is deprecated and will be removed in a future version.
    """

    def __init__(self, context: str):
        self.context = context

    def __contains__(self, primitive):
        return REGISTRY.has_rule(primitive, self.context)

    def __getitem__(self, primitive):
        rule = REGISTRY.get(primitive, self.context)
        if rule is None:
            raise KeyError(f"No rule registered for {primitive} in {self.context}")
        return rule

    def get(self, primitive, default=None):
        rule = REGISTRY.get(primitive, self.context)
        return rule if rule is not None else default


# Backwards compatibility: these shims provide dict-like access to rules
# DEPRECATED: Use REGISTRY.has_rule(prim, Context.INVERSE) instead
UNIVARIATE_INVERSE_REGISTRY = _RegistryShim(Context.INVERSE)
BIVARIATE_INVERSE_REGISTRY = _RegistryShim(Context.INVERSE)
CUSTOM_INVERSE_PROCESSING_RULES = _RegistryShim(Context.INVERSE)


def register_inverse_rule(primitive, rule):
    """
    Register an inverse rule for a primitive.

    DEPRECATED: Use @REGISTRY.rule(primitive, Context.INVERSE) instead.
    """
    REGISTRY.register(primitive, Context.INVERSE, rule)


__all__ = [
    # Main exports
    "has_registered_inverse",
    "inverse_cost_fn",
    # Backwards compatibility (deprecated)
    "BIVARIATE_INVERSE_REGISTRY",
    "CUSTOM_INVERSE_PROCESSING_RULES",
    "UNIVARIATE_INVERSE_REGISTRY",
    "register_inverse_rule",
]

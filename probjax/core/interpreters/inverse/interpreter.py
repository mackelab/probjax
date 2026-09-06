"""
Inverse interpreter using unified rule registry.

This module provides the InverseProcessingRule class that processes JAXPR
equations to compute inverse values using rules registered in the global
REGISTRY.
"""

import jax
import jax.numpy as jnp
from jax._src import core as jax_core

from probjax.core.custom_primitives.contracts import parse_custom_inverse_call_params
from probjax.core.custom_primitives.custom_inverse import (
    custom_inverse,
    custom_inverse_call_p,
    custom_inverse_enabled,
)
from probjax.core.jaxpr_propagation.utils import (
    KnownessLevel,
    ProcessingRule,
)
from probjax.core.registry import Context, ProcessedResult, REGISTRY


def maybe_inverse_custom_inverse(
    fun,
    *,
    static_argnums=(),
    invertible_arg=None,
):
    """
    Create an inverse wrapper for a custom_inverse function.

    If `fun` is already a custom_inverse instance with defined inverse functions,
    this returns a new custom_inverse that inverts the inverse (i.e., recovers
    the original forward function behavior).

    Returns None when custom-inverse handling is disabled (see
    ``disable_custom_inverse``), so callers fall back to structural inversion.
    """
    if not custom_inverse_enabled():
        return None
    if not isinstance(fun, custom_inverse):
        return None

    configured_static = tuple(fun.static_argnums or ())
    requested_static = tuple(sorted(static_argnums or ()))
    if requested_static and requested_static != configured_static:
        raise ValueError(
            "For custom_inverse inputs, static_argnums must match the "
            "custom_inverse configuration."
        )

    if invertible_arg is not None and invertible_arg != fun.inv_argnum:
        raise ValueError(
            "For custom_inverse inputs, invertible_arg must match "
            "custom_inverse.inv_argnum."
        )

    if fun.inv_fun is None or fun.inv_fun_and_log_det is None:
        raise AttributeError(
            "Inverse not defined on custom_inverse input. "
            "Use definv/definv_and_logdet first."
        )

    inverse_wrapper = custom_inverse(
        fun.inv_fun,
        inv_argnum=fun.inv_argnum,
        static_argnums=fun.static_argnums,
    )
    inverse_wrapper.definv(fun.fun)

    def inverse_inverse_and_logdet(*args, **kwargs):
        # Prefer a direct forward value/logdet if available.
        if fun.value_and_logdet_fun is not None:
            return fun.value_and_logdet(*args, **kwargs)

        # Otherwise use the inverse branch and negate its logdet.
        y = fun.fun(*args, **kwargs)
        inv_args = list(args)
        inv_args[fun.inv_argnum] = y
        _, inv_logdet = fun.inv_and_logdet(*tuple(inv_args), **kwargs)
        return y, -jnp.asarray(inv_logdet)

    inverse_wrapper.definv_and_logdet(inverse_inverse_and_logdet)
    inverse_wrapper.defvalue_and_logdet(fun.inv_and_logdet)
    return inverse_wrapper


class InverseProcessingRule(ProcessingRule):
    """
    Processing rule that computes inverse values using the unified registry.

    For each equation, this rule uses Knowness levels to determine whether
    to run FORWARD or INVERSE:

    1. If any output is COMPLETE: prefer INVERSE (don't overwrite authoritative values)
    2. If all inputs known and outputs can be overwritten: prefer FORWARD
    3. If all outputs known: try INVERSE
    4. If all inputs known: try FORWARD
    5. Handle custom_inverse_call_p specially using its inverse jaxpr
    """

    def __call__(
        self, eqn, known_invars, known_outvars, context=None
    ) -> ProcessedResult | None:
        # Check if this is a custom_inverse_call primitive
        if eqn.primitive is custom_inverse_call_p:
            return self._process_custom_inverse_call(eqn, known_invars, known_outvars)

        all_inputs_known = all(v is not None for v in known_invars)
        all_outputs_known = all(v is not None for v in known_outvars)

        # Get knowness levels from context if available
        # This allows us to distinguish PARTIAL from COMPLETE outputs
        output_knowness_levels = self._get_output_knowness_levels(eqn, context)
        input_knowness_levels = self._get_input_knowness_levels(eqn, context)

        any_output_complete = any(
            level == KnownessLevel.COMPLETE for level in output_knowness_levels
        )
        all_outputs_complete = all(
            level == KnownessLevel.COMPLETE for level in output_knowness_levels
        )
        all_outputs_can_overwrite = all(
            level != KnownessLevel.COMPLETE for level in output_knowness_levels
        )
        # Check if ALL inputs are COMPLETE (authoritative, no placeholders)
        all_inputs_complete = all(
            level == KnownessLevel.COMPLETE for level in input_knowness_levels
        )

        # When BOTH inputs AND outputs are known, use knowness to decide:
        # - If any output is COMPLETE (authoritative), prefer INVERSE
        # - If outputs are PARTIAL and ALL inputs are COMPLETE, FORWARD can fix them
        if all_inputs_known and all_outputs_known:
            if any_output_complete:
                # At least one output is authoritative - use INVERSE to compute inputs
                result = REGISTRY.process(
                    eqn, known_invars, known_outvars, Context.INVERSE
                )
                if result is not None:
                    return result
            elif all_outputs_can_overwrite and all_inputs_complete:
                # Outputs are partial/placeholder AND inputs are fully known
                # FORWARD can fix the placeholder outputs
                result = REGISTRY.process(
                    eqn, known_invars, known_outvars, Context.FORWARD
                )
                if result is not None:
                    return result

        # Try inverse rule if all outputs are known AND all outputs are COMPLETE
        # (If any output is PARTIAL, inverse would use placeholder values)
        # Exception: The "any_output_complete" case above already handled mixed cases
        if all_outputs_known and all_outputs_complete:
            result = REGISTRY.process(eqn, known_invars, known_outvars, Context.INVERSE)
            if result is not None:
                return result

        # Try forward rule if all inputs are known AND all inputs are COMPLETE
        # (If any input is PARTIAL, forward would produce garbage from placeholder values)
        if all_inputs_known and all_inputs_complete:
            result = REGISTRY.process(eqn, known_invars, known_outvars, Context.FORWARD)
            if result is not None:
                return result

        # Fallback: Try inverse if outputs are known but inputs are not all known
        # This handles cases like scatter where we need to extract values from
        # a PARTIAL output array (e.g., dynamic_slice wrote valid data at scatter indices)
        if all_outputs_known and not all_inputs_known:
            result = REGISTRY.process(eqn, known_invars, known_outvars, Context.INVERSE)
            if result is not None:
                return result

        # Cannot process this equation
        return None

    def _get_input_knowness_levels(self, eqn, context) -> list[KnownessLevel]:
        """Get the knowness levels for input variables."""
        if context is None or not hasattr(context, "env"):
            # Fallback: assume all known values are COMPLETE
            return [KnownessLevel.COMPLETE] * len(eqn.invars)

        env = context.env
        levels = []
        for var in eqn.invars:
            level = env.get_knowness_level(var)
            levels.append(level)
        return levels

    def _get_output_knowness_levels(self, eqn, context) -> list[KnownessLevel]:
        """Get the knowness levels for output variables."""
        if context is None or not hasattr(context, "env"):
            # Fallback: assume all known values are COMPLETE
            return [KnownessLevel.COMPLETE] * len(eqn.outvars)

        env = context.env
        levels = []
        for var in eqn.outvars:
            level = env.get_knowness_level(var)
            levels.append(level)
        return levels

    def _process_custom_inverse_call(
        self, eqn, known_invars, known_outvars
    ) -> ProcessedResult | None:
        """
        Handle custom_inverse_call_p by evaluating its inverse jaxpr.
        """
        # If outputs not known, cannot compute inverse
        if not all(v is not None for v in known_outvars):
            # Try forward if all inputs known
            if all(v is not None for v in known_invars):
                result = REGISTRY.process(
                    eqn, known_invars, known_outvars, Context.FORWARD
                )
                if result is not None:
                    return result
            return None

        custom_params = parse_custom_inverse_call_params(eqn.params)
        inverse_jaxpr = custom_params.inverse_jaxpr_thunk()

        jaxpr = inverse_jaxpr.jaxpr
        consts = inverse_jaxpr.literals

        target_indices = set(custom_params.target_in_indices)
        inputs = []
        outputs_inserted = False
        for index, value in enumerate(known_invars):
            if index in target_indices:
                if not outputs_inserted:
                    inputs.extend(known_outvars)
                    outputs_inserted = True
                continue
            if value is None:
                return None
            inputs.append(value)

        out = jax_core.eval_jaxpr(jaxpr, consts, *inputs)

        invars = [eqn.invars[index] for index in custom_params.target_in_indices]
        vals = list(out[: len(invars)])

        return ProcessedResult(invars, vals)

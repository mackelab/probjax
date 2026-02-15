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
)
from probjax.core.jaxpr_propagation.utils import ProcessingRule
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
    """
    if not isinstance(fun, custom_inverse):
        return None

    configured_static = tuple(fun.static_argnums or ())
    requested_static = tuple(static_argnums or ())
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

    For each equation, this rule:
    1. If all outputs are known: try to apply an INVERSE rule to recover inputs
    2. If all inputs are known: run the primitive forward to compute outputs
    3. Handle custom_inverse_call_p specially using its inverse jaxpr
    """

    def __call__(
        self, eqn, known_invars, known_outvars, context=None
    ) -> ProcessedResult | None:
        # Check if this is a custom_inverse_call primitive
        if eqn.primitive is custom_inverse_call_p:
            return self._process_custom_inverse_call(eqn, known_invars, known_outvars)

        all_inputs_known = all(v is not None for v in known_invars)
        all_outputs_known = all(v is not None for v in known_outvars)

        # When BOTH inputs AND outputs are known, prefer FORWARD processing.
        # This handles the "resolve conflicts" case where an intermediate variable
        # was set to a placeholder value by an inverse rule, and the forward pass
        # should overwrite it with the correct computed value.
        if all_inputs_known and all_outputs_known:
            result = REGISTRY.process(eqn, known_invars, known_outvars, Context.FORWARD)
            if result is not None:
                return result

        # Try inverse rule if all outputs are known
        if all_outputs_known:
            result = REGISTRY.process(eqn, known_invars, known_outvars, Context.INVERSE)
            if result is not None:
                return result

        # Try forward rule if all inputs are known
        if all_inputs_known:
            result = REGISTRY.process(eqn, known_invars, known_outvars, Context.FORWARD)
            if result is not None:
                return result

        # Cannot process this equation
        return None

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

        # Build inputs: use known_invars where available, else use output value
        inputs = [v if v is not None else known_outvars[0] for v in known_invars]

        out = jax_core.eval_jaxpr(jaxpr, consts, *inputs)

        # Return only the variables that were unknown
        invars = [
            eqn.invars[i] for i in range(len(eqn.invars)) if known_invars[i] is None
        ]
        vals = [out[0] for i in range(len(eqn.invars)) if known_invars[i] is None]

        return ProcessedResult(invars, vals)

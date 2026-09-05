"""
Inverse+LogAbsDet interpreter using unified rule registry.

This module provides the InverseAndLogAbsDetProcessingRule class that processes
JAXPR equations to compute inverse values along with log-determinant tracking
using rules registered in the global REGISTRY.
"""

import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax.extend.core import Literal

from probjax.core.custom_primitives.contracts import parse_custom_inverse_call_params
from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p
from probjax.core.interpreters.inverse.interpreter import InverseProcessingRule
from probjax.core.interpreters.inverse.logabsdet_rules import (
    INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
    is_elementwise_primitive,
    value_and_log_det_diagonal,
)
from probjax.core.interpreters.inverse.utils import is_inexact_value
from probjax.core.registry import Context, ProcessedResult, REGISTRY


class InverseAndLogAbsDetProcessingRule(InverseProcessingRule):
    """
    Processing rule that computes inverse values with log-determinant tracking.

    This extends InverseProcessingRule to additionally track the log absolute
    determinant of the Jacobian for each transformation. The log-determinants
    are accumulated in the context's run state.

    For each equation, this rule:
    1. If a INVERSE_LOGDET rule exists and all outputs are known: use it
    2. If only INVERSE rule exists: fall back to autodiff for log-det
    3. If all inputs known: run forward (with zero log-det contribution)
    4. Handle custom_inverse_call_p specially using its inverse+logdet jaxpr
    """

    def __init__(self, state_namespace: str = INVERSE_AND_LOGABSDET_STATE_NAMESPACE):
        super().__init__()
        self.state_namespace = state_namespace

    def _read_log_dets(self, context) -> dict:
        """Read accumulated log-determinants from context."""
        if context is None:
            return {}
        state = context.read_run_state(namespace=self.state_namespace)
        if state is None:
            return {}
        return state

    def _read_nested_log_dets(self, context) -> dict:
        """Read nested log-determinants from transient state."""
        if context is None:
            return {}
        nested_state = context.read_transient_state(self.state_namespace, default=None)
        if nested_state is None:
            return {}
        return nested_state

    @staticmethod
    def _sum_log_dets(log_dets: dict, vars_) -> jax.Array:
        """Sum log-determinants for the given variables."""
        total = jnp.asarray(0.0)
        for v in vars_:
            if isinstance(v, Literal):
                continue
            total = total + jnp.asarray(log_dets.get(v, 0.0))
        return total

    def __call__(
        self, eqn, known_invars, known_outvars, context=None
    ) -> ProcessedResult | None:
        # Handle custom_inverse_call_p specially
        if eqn.primitive is custom_inverse_call_p:
            return self._process_custom_inverse_call_with_logdet(
                eqn, known_invars, known_outvars, context
            )

        # Try INVERSE_LOGDET rule if all outputs are known
        if all(v is not None for v in known_outvars):
            result = REGISTRY.process(
                eqn, known_invars, known_outvars, Context.INVERSE_LOGDET
            )
            if result is not None:
                # Add previous log-dets from output variables to the rule's state
                return self._add_previous_logdets_to_state(
                    eqn,
                    result.resolved_vars,
                    result.resolved_vals,
                    result.state or {},
                    context,
                )

            # Fall back to INVERSE rule + autodiff for log-det
            result = REGISTRY.process(eqn, known_invars, known_outvars, Context.INVERSE)
            if result is not None:
                return self._add_autodiff_logdet(
                    eqn, known_invars, known_outvars, result, context
                )

        # Try forward rule if all inputs are known
        if all(v is not None for v in known_invars):
            result = REGISTRY.process(eqn, known_invars, known_outvars, Context.FORWARD)
            if result is not None:
                return ProcessedResult(result.resolved_vars, result.resolved_vals, {})

        # Cannot process this equation
        return None

    def _add_previous_logdets_to_state(
        self, eqn, resolved_vars, resolved_vals, state, context
    ) -> ProcessedResult:
        """
        Add previous log-dets from output variables to the state.

        This implements the chain rule: log|dx/dz| = log|dx/dy| + log|dy/dz|
        The rule provides log|dx/dy|, and we need to add the accumulated
        log|dy/dz| from the output variables.
        """
        log_dets = self._read_log_dets(context)
        previous = self._sum_log_dets(log_dets, eqn.outvars)

        updated_state = {}
        for var in resolved_vars:
            if isinstance(var, Literal):
                continue
            local_logdet = state.get(var, jnp.asarray(0.0))
            updated_state[var] = previous + jnp.asarray(local_logdet)

        return ProcessedResult(resolved_vars, resolved_vals, updated_state)

    def _add_autodiff_logdet(
        self, eqn, known_invars, known_outvars, inverse_result, context
    ) -> ProcessedResult:
        """
        Compute log-determinant via autodiff for an inverse result.

        This is used as a fallback when no explicit INVERSE_LOGDET rule exists.
        """
        if not is_elementwise_primitive(eqn.primitive):
            raise NotImplementedError(
                f"no usable INVERSE_LOGDET rule for '{eqn.primitive.name}'. The "
                "autodiff fallback assumes a diagonal Jacobian, which is wrong "
                "for a primitive that is not elementwise, so it is refused "
                "rather than guessed. Register a rule with REGISTRY.rule("
                f"<{eqn.primitive.name}_p>, Context.INVERSE_LOGDET), or "
                "register_rearrangement_inverse_logdet if it only moves "
                "elements around. (A rearrangement rule that declines because "
                "the input is only partially recovered also lands here: "
                "log-determinants under partial propagation are unsupported.)"
            )

        resolved_vars = inverse_result.resolved_vars
        resolved_vals = inverse_result.resolved_vals

        log_dets = self._read_log_dets(context)
        previous = self._sum_log_dets(log_dets, eqn.outvars)



        # Compute log-det via autodiff if values are inexact
        if len(resolved_vals) == 1:
            out_val = known_outvars[0]
            in_val = resolved_vals[0]

            if is_inexact_value(out_val) and is_inexact_value(in_val):
                # Build a function that computes the inverse for autodiff
                rule = REGISTRY.get(eqn.primitive, Context.INVERSE)
                if rule is not None:

                    def inverse_fn(*args):
                        # Re-execute the inverse to get value for autodiff
                        result = rule(eqn, known_invars, args)
                        if isinstance(result, ProcessedResult):
                            return jnp.sum(result.resolved_vals[0])
                        elif isinstance(result, tuple):
                            return jnp.sum(result[1][0])
                        return jnp.asarray(0.0)

                    eval_fn = value_and_log_det_diagonal(inverse_fn)
                    _, log_abs_det = eval_fn(*known_outvars)
                else:
                    log_abs_det = jnp.asarray(0.0)
            else:
                log_abs_det = jnp.asarray(0.0)
        else:
            log_abs_det = jnp.asarray(0.0)

        # Build updates
        updates = {}
        for var in resolved_vars:
            if not isinstance(var, Literal):
                updates[var] = previous + log_abs_det

        return ProcessedResult(resolved_vars, resolved_vals, updates)

    def _process_custom_inverse_call_with_logdet(
        self, eqn, known_invars, known_outvars, context
    ) -> ProcessedResult | None:
        """
        Handle custom_inverse_call_p by evaluating its inverse+logdet jaxpr.
        """
        # If outputs not known, cannot compute inverse
        if not all(v is not None for v in known_outvars):
            # Try forward if all inputs known
            if all(v is not None for v in known_invars):
                result = REGISTRY.process(
                    eqn, known_invars, known_outvars, Context.FORWARD
                )
                if result is not None:
                    return ProcessedResult(
                        result.resolved_vars, result.resolved_vals, {}
                    )
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
        result_vals = list(out[: len(invars)])

        # Extract log-det from output (last element)
        log_abs_det = jnp.sum(out[-1])

        log_dets = self._read_log_dets(context)
        previous = self._sum_log_dets(log_dets, eqn.outvars)

        updates = {}
        for index, var in enumerate(invars):
            if isinstance(var, Literal):
                continue
            updates[var] = previous + log_abs_det if index == 0 else jnp.asarray(0.0)

        return ProcessedResult(invars, result_vals, updates)

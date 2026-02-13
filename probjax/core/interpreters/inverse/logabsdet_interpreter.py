import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax.extend.core import Literal, Primitive

from probjax.core.custom_primitives.contracts import parse_custom_inverse_call_params
from probjax.core.interpreters.common import apply_rule
from probjax.core.interpreters.inverse.dispatch import (
    DispatchAction,
    classify_primitive,
    compute_knownness,
    select_dispatch_action,
)
from probjax.core.interpreters.inverse.interpreter import InverseProcessingRule
from probjax.core.interpreters.inverse.registry import (
    BIVARIATE_INVERSE_REGISTRY,
    CUSTOM_INVERSE_PROCESSING_RULES,
    UNIVARIATE_INVERSE_REGISTRY,
)
from probjax.core.interpreters.inverse.logabsdet_rules import (
    CUSTOM_INVERSE_AND_LOG_DET_RULES,
    INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
    value_and_log_det_diagonal,
)


_ACTION_METHODS = {
    DispatchAction.CUSTOM_CALL: "_custom_inverse_call_with_logdet",
    DispatchAction.CUSTOM_RULE: "_custom_rule_with_logdet",
    DispatchAction.PJIT_MISSING_INPUTS: "_pjit_with_logdet",
    DispatchAction.UNIVARIATE: "_univariate_inverse_with_logdet",
    DispatchAction.BIVARIATE: "_bivariate_inverse_with_logdet",
}


def _is_inexact_value(value) -> bool:
    return jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact)


class InverseAndLogAbsDetProcessingRule(InverseProcessingRule):
    def __init__(self, state_namespace: str = INVERSE_AND_LOGABSDET_STATE_NAMESPACE):
        super().__init__()
        self.state_namespace = state_namespace

    def _read_log_dets(self, context) -> dict:
        if context is None:
            return {}
        state = context.read_run_state(namespace=self.state_namespace)
        if state is None:
            return {}
        return state

    def _read_nested_log_dets(self, context) -> dict:
        if context is None:
            return {}
        nested_state = context.read_transient_state(self.state_namespace, default=None)
        if nested_state is None:
            return {}
        return nested_state

    @staticmethod
    def _sum_log_dets(log_dets: dict, vars_) -> jax.Array:
        total = jnp.asarray(0.0)
        for v in vars_:
            if isinstance(v, Literal):
                continue
            total = total + jnp.asarray(log_dets.get(v, 0.0))
        return total

    def __call__(self, eqn, known_invars, known_outvars, context=None):
        knownness = compute_knownness(known_invars, known_outvars)

        if eqn.primitive in CUSTOM_INVERSE_AND_LOG_DET_RULES and knownness.out_all:
            return apply_rule(
                CUSTOM_INVERSE_AND_LOG_DET_RULES[eqn.primitive],
                eqn,
                known_invars,
                known_outvars,
                context=context,
            )

        primitive_kind = classify_primitive(
            eqn,
            custom_rules=CUSTOM_INVERSE_PROCESSING_RULES,
            univariate_registry=UNIVARIATE_INVERSE_REGISTRY,
            bivariate_registry=BIVARIATE_INVERSE_REGISTRY,
        )
        action = select_dispatch_action(
            primitive_kind,
            knownness,
            prefer_resolve_conflict=False,
        )

        if action is DispatchAction.FORWARD:
            return self._default_forward_processing(eqn, known_invars, known_outvars)

        method_name = _ACTION_METHODS.get(action)
        if method_name is not None:
            method = getattr(self, method_name)
            return method(eqn, known_invars, known_outvars, context=context)

        raise NotImplementedError(f"Cannot invert {eqn}")

    def _univariate_inverse_with_logdet(
        self, eqn, known_invars, known_outvars, context=None
    ):
        del known_invars
        primitive = eqn.primitive
        if primitive not in UNIVARIATE_INVERSE_REGISTRY:
            raise NotImplementedError(f"{primitive} is not invertible!")

        inv_primitive = UNIVARIATE_INVERSE_REGISTRY[primitive]
        if isinstance(inv_primitive, Primitive):
            subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
            invars = inv_primitive.bind(*subfuns, *known_outvars, **bind_params)

            if _is_inexact_value(known_outvars[0]) and _is_inexact_value(invars):

                def f(*args):
                    subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
                    invars_local = inv_primitive.bind(*subfuns, *args, **bind_params)
                    return jnp.sum(invars_local)

                eval_fn = value_and_log_det_diagonal(f)
                invars, log_abs_det = eval_fn(*known_outvars)
            else:
                log_abs_det = jnp.asarray(0.0)
        else:
            invars = inv_primitive(*known_outvars, **eqn.params)
            if _is_inexact_value(known_outvars[0]) and _is_inexact_value(invars):
                eval_fn = value_and_log_det_diagonal(
                    lambda *args: jnp.sum(inv_primitive(*args, **eqn.params))
                )
                invars, log_abs_det = eval_fn(*known_outvars)
            else:
                log_abs_det = jnp.asarray(0.0)

        if not isinstance(invars, list):
            invars = [invars]

        log_dets = self._read_log_dets(context)
        previous = jnp.asarray(log_dets.get(eqn.outvars[0], 0.0))
        updates = {}
        if not isinstance(eqn.invars[0], Literal):
            updates[eqn.invars[0]] = previous + log_abs_det

        return eqn.invars, invars, updates

    def _bivariate_inverse_with_logdet(
        self, eqn, known_invars, known_outvars, context=None
    ):
        primitive = eqn.primitive
        input1 = known_outvars[0]
        left_inverse = known_invars[0] is None
        input2 = known_invars[1] if left_inverse else known_invars[0]

        (left_inverse_fn, right_inverse_fn) = BIVARIATE_INVERSE_REGISTRY[primitive]
        inv_primitive = left_inverse_fn if left_inverse else right_inverse_fn

        if isinstance(inv_primitive, Primitive):
            subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
            invars = inv_primitive.bind(*subfuns, input1, input2, **bind_params)

            if _is_inexact_value(input1) and _is_inexact_value(invars):

                def f(*args):
                    subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
                    return inv_primitive.bind(*subfuns, *args, **bind_params)

                eval_fn = value_and_log_det_diagonal(f)
                invars, log_abs_det = eval_fn(input1, input2)
            else:
                log_abs_det = jnp.asarray(0.0)
        else:
            invars = inv_primitive(input1, input2, **eqn.params)
            if _is_inexact_value(input1) and _is_inexact_value(invars):
                eval_fn = value_and_log_det_diagonal(
                    lambda *args: jnp.sum(inv_primitive(*args, **eqn.params))
                )
                invars, log_abs_det = eval_fn(input1, input2)
            else:
                log_abs_det = jnp.asarray(0.0)

        log_dets = self._read_log_dets(context)
        previous = jnp.asarray(log_dets.get(eqn.outvars[0], 0.0))
        updates = {}
        if left_inverse:
            if not isinstance(eqn.invars[0], Literal):
                updates[eqn.invars[0]] = previous + log_abs_det
            return [eqn.invars[0]], [invars], updates

        if not isinstance(eqn.invars[1], Literal):
            updates[eqn.invars[1]] = previous + log_abs_det
        return [eqn.invars[1]], [invars], updates

    def _pjit_with_logdet(self, eqn, known_invars, known_outvars, context=None):
        del known_invars, known_outvars
        if "jaxpr" in eqn.params:
            jaxpr = eqn.params["jaxpr"]
        else:
            jaxpr = eqn.params["call_jaxpr"]

        sub_invars = jaxpr.jaxpr.invars
        sub_outvars = jaxpr.jaxpr.outvars
        subvars = sub_invars + sub_outvars
        vars_ = eqn.invars + eqn.outvars

        log_dets = self._read_log_dets(context)
        nested_log_dets = self._read_nested_log_dets(context)
        lookup = dict(log_dets)
        lookup.update(nested_log_dets)

        updates = {}
        for v_sub, v in zip(subvars, vars_, strict=False):
            if not isinstance(v, Literal) and v_sub in lookup:
                updates[v] = lookup[v_sub]

        previous = jnp.asarray(0.0)
        for v in eqn.outvars:
            if isinstance(v, Literal):
                continue
            previous = previous + jnp.asarray(updates.get(v, lookup.get(v, 0.0)))

        for v in eqn.invars:
            if not isinstance(v, Literal):
                if v not in updates:
                    updates[v] = previous

        return [], [], updates

    def _custom_rule_with_logdet(self, eqn, known_invars, known_outvars, context=None):
        primitive = eqn.primitive
        if primitive not in CUSTOM_INVERSE_PROCESSING_RULES:
            raise NotImplementedError(f"{primitive} is not invertible!")

        outvars, outs = apply_rule(
            CUSTOM_INVERSE_PROCESSING_RULES[primitive],
            eqn,
            known_invars,
            known_outvars,
            context=context,
        )

        log_dets = self._read_log_dets(context)
        previous = self._sum_log_dets(log_dets, eqn.outvars)
        updates = {v: previous for v in outvars if not isinstance(v, Literal)}

        return outvars, outs, updates

    def _custom_inverse_call_with_logdet(
        self,
        eqn,
        known_invars,
        known_outvars,
        context=None,
    ):
        custom_params = parse_custom_inverse_call_params(eqn.params)
        inverse_jaxpr = custom_params.inverse_jaxpr_thunk()
        jaxpr = inverse_jaxpr.jaxpr
        consts = inverse_jaxpr.literals
        inputs = [v if v is not None else known_outvars[0] for v in known_invars]
        out = jax_core.eval_jaxpr(
            jaxpr,
            consts,
            *inputs,
        )
        invars = [
            eqn.invars[i] for i in range(len(eqn.invars)) if known_invars[i] is None
        ]
        inputs = [out[0] for i in range(len(eqn.invars)) if known_invars[i] is None]
        log_abs_det = jnp.sum(out[-1])

        log_dets = self._read_log_dets(context)
        previous = self._sum_log_dets(log_dets, eqn.outvars)
        updates = {
            v: previous + log_abs_det for v in eqn.invars if not isinstance(v, Literal)
        }

        return invars, out[:-1], updates

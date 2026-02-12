import jax
from jax._src import core as jax_core
from jax.extend.core import Primitive

from probjax.core.interpreters.common import apply_rule
from probjax.core.interpreters.inverse.dispatch import (
    DispatchAction,
    classify_primitive,
    compute_knownness,
    select_dispatch_action,
)
from probjax.core.interpreters.inverse.registry import (
    BIVARIATE_INVERSE_REGISTRY,
    CUSTOM_INVERSE_PROCESSING_RULES,
    UNIVARIATE_INVERSE_REGISTRY,
)
from probjax.core.jaxpr_propagation.utils import ProcessingRule


_ACTION_METHODS = {
    DispatchAction.CUSTOM_CALL: "_default_custom_inverse_call_apply",
    DispatchAction.RESOLVE_CONFLICT: "_default_resolve_conflicts",
    DispatchAction.UNIVARIATE: "_default_univariate_inverse",
    DispatchAction.BIVARIATE: "_default_bivariate_inverse",
    DispatchAction.FORWARD: "_default_forward_processing",
}


class InverseProcessingRule(ProcessingRule):
    def __call__(self, eqn, known_invars, known_outvars, context=None):
        knownness = compute_knownness(known_invars, known_outvars)
        primitive_kind = classify_primitive(
            eqn,
            custom_rules=CUSTOM_INVERSE_PROCESSING_RULES,
            univariate_registry=UNIVARIATE_INVERSE_REGISTRY,
            bivariate_registry=BIVARIATE_INVERSE_REGISTRY,
        )
        action = select_dispatch_action(
            primitive_kind,
            knownness,
            prefer_resolve_conflict=True,
        )

        if action is DispatchAction.CUSTOM_RULE:
            return apply_rule(
                CUSTOM_INVERSE_PROCESSING_RULES[eqn.primitive],
                eqn,
                known_invars,
                known_outvars,
                context=context,
            )
        if action is DispatchAction.PJIT_MISSING_INPUTS:
            return None

        method_name = _ACTION_METHODS.get(action)
        if method_name is not None:
            method = getattr(self, method_name)
            return method(eqn, known_invars, known_outvars)

        raise NotImplementedError(f"Cannot invert {eqn}")

    def _default_univariate_inverse(self, eqn, known_invars, known_outvars):
        del known_invars
        primitive = eqn.primitive
        if primitive not in UNIVARIATE_INVERSE_REGISTRY:
            raise NotImplementedError(f"{primitive} is not invertible!")

        inv_primitive = UNIVARIATE_INVERSE_REGISTRY[primitive]
        if isinstance(inv_primitive, Primitive):
            subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
            invars = inv_primitive.bind(*subfuns, *known_outvars, **bind_params)
        else:
            invars = inv_primitive(*known_outvars, **eqn.params)

        if not isinstance(invars, list):
            invars = [invars]

        return eqn.invars, invars

    def _default_bivariate_inverse(self, eqn, known_invars, known_outvars):
        primitive = eqn.primitive
        input1 = known_outvars[0]
        left_inverse = known_invars[0] is None
        input2 = known_invars[1] if left_inverse else known_invars[0]

        (left_inverse_fn, right_inverse_fn) = BIVARIATE_INVERSE_REGISTRY[primitive]
        inv_primitive = left_inverse_fn if left_inverse else right_inverse_fn

        if isinstance(inv_primitive, Primitive):
            subfuns, bind_params = inv_primitive.get_bind_params(eqn.params)
            missing_invar = inv_primitive.bind(*subfuns, input1, input2, **bind_params)
        else:
            missing_invar = inv_primitive(input1, input2, **eqn.params)

        if left_inverse:
            return [eqn.invars[0]], [missing_invar]
        return [eqn.invars[1]], [missing_invar]

    def _default_resolve_conflicts(self, eqn, known_invars, known_outvars):
        outvars, outvals = self._default_forward_processing(
            eqn,
            known_invars,
            known_outvars,
        )
        return outvars, outvals

    def _default_forward_processing(self, eqn, known_invars, known_outvars):
        del known_outvars
        primitive = eqn.primitive
        subfuns, bind_params = primitive.get_bind_params(eqn.params)
        outvals = primitive.bind(*subfuns, *known_invars, **bind_params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        return eqn.outvars, outvals  # type: ignore

    def _default_custom_inverse_call_apply(self, eqn, known_invars, known_outvars):
        inverse_jaxpr = eqn.params["inverse_jaxpr_thunk"]()

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
        return invars, inputs

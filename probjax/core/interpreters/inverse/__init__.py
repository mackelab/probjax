from probjax.core.interpreters.inverse.interpreter import (
    InverseProcessingRule,
    maybe_inverse_custom_inverse,
)
from probjax.core.interpreters.inverse.logabsdet_interpreter import (
    InverseAndLogAbsDetProcessingRule,
)
from probjax.core.interpreters.inverse.logabsdet_rules import (
    CUSTOM_INVERSE_AND_LOG_DET_RULES,
    INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
    inverse_and_logabsdet_state_reducer,
    register_inverse_and_log_det_rule,
    set_logabsdet_processing_rule_factory,
    value_and_log_det_diagonal,
)
from probjax.core.interpreters.inverse.registry import (
    BIVARIATE_INVERSE_REGISTRY,
    CUSTOM_INVERSE_PROCESSING_RULES,
    UNIVARIATE_INVERSE_REGISTRY,
    has_registered_inverse,
    inverse_cost_fn,
    is_bivariate,
    is_univariate,
    register_inverse_rule,
)
from probjax.core.interpreters.inverse.rules import (
    set_inverse_cost_fn,
    set_inverse_processing_rule_factory,
)

_WIRED = False


def configure_inverse_wiring() -> None:
    global _WIRED
    if _WIRED:
        return
    set_inverse_processing_rule_factory(InverseProcessingRule)
    set_inverse_cost_fn(inverse_cost_fn)
    set_logabsdet_processing_rule_factory(InverseAndLogAbsDetProcessingRule)
    _WIRED = True


configure_inverse_wiring()

__all__ = [
    "BIVARIATE_INVERSE_REGISTRY",
    "CUSTOM_INVERSE_AND_LOG_DET_RULES",
    "CUSTOM_INVERSE_PROCESSING_RULES",
    "INVERSE_AND_LOGABSDET_STATE_NAMESPACE",
    "InverseAndLogAbsDetProcessingRule",
    "InverseProcessingRule",
    "UNIVARIATE_INVERSE_REGISTRY",
    "has_registered_inverse",
    "inverse_and_logabsdet_state_reducer",
    "inverse_cost_fn",
    "is_bivariate",
    "is_univariate",
    "maybe_inverse_custom_inverse",
    "register_inverse_and_log_det_rule",
    "register_inverse_rule",
    "value_and_log_det_diagonal",
    "configure_inverse_wiring",
]

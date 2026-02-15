"""
Inverse interpreter package.

This package provides the inverse interpreter for computing inverse values
from JAXPR equations, along with optional log-determinant tracking.

The main classes are:
- InverseProcessingRule: Computes inverse values
- InverseAndLogAbsDetProcessingRule: Computes inverse values with log-det tracking

All rules are registered in the unified REGISTRY from probjax.core.registry.
"""

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
    register_inverse_rule,
)
from probjax.core.interpreters.inverse.rules import (
    set_inverse_cost_fn,
    set_inverse_processing_rule_factory,
)

# Import rules modules to trigger registration
import probjax.core.interpreters.inverse.rules  # noqa: F401
import probjax.core.interpreters.inverse.logabsdet_rules  # noqa: F401

_WIRED = False


def configure_inverse_wiring() -> None:
    """
    Configure the inverse interpreter wiring.

    This sets up the processing rule factories and cost functions needed
    for nested propagation in control flow primitives.
    """
    global _WIRED
    if _WIRED:
        return
    set_inverse_processing_rule_factory(InverseProcessingRule)
    set_inverse_cost_fn(inverse_cost_fn)
    set_logabsdet_processing_rule_factory(InverseAndLogAbsDetProcessingRule)
    _WIRED = True


configure_inverse_wiring()

__all__ = [
    # Main interpreter classes
    "InverseProcessingRule",
    "InverseAndLogAbsDetProcessingRule",
    # Helper functions
    "maybe_inverse_custom_inverse",
    "has_registered_inverse",
    "inverse_cost_fn",
    "value_and_log_det_diagonal",
    "inverse_and_logabsdet_state_reducer",
    "configure_inverse_wiring",
    # State namespace
    "INVERSE_AND_LOGABSDET_STATE_NAMESPACE",
    # Backwards compatibility (deprecated)
    "BIVARIATE_INVERSE_REGISTRY",
    "CUSTOM_INVERSE_AND_LOG_DET_RULES",
    "CUSTOM_INVERSE_PROCESSING_RULES",
    "UNIVARIATE_INVERSE_REGISTRY",
    "register_inverse_and_log_det_rule",
    "register_inverse_rule",
]

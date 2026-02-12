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

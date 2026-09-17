"""Shared helpers for probabilistic-program interpretation rules."""

from typing import Any, Optional

from jax.extend.core import JaxprEqn

from probjax.core.custom_primitives.contracts import parse_random_variable_call_params
from probjax.core.custom_primitives.random_variable import rv_p


def rv_site_name(eqn: JaxprEqn) -> Optional[str]:
    """Site name if ``eqn`` is a random-variable site, else ``None``."""
    if eqn.primitive is not rv_p:
        return None
    return parse_random_variable_call_params(eqn.params).name


def rv_site_params(eqn: JaxprEqn) -> Optional[Any]:
    """Site params if ``eqn`` is a random-variable site, else ``None``."""
    if eqn.primitive is not rv_p:
        return None
    return parse_random_variable_call_params(eqn.params)

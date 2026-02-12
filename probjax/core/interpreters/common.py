import inspect
from typing import Any, Callable


def supports_context_argument(func: Callable, minimum_positional: int = 4) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    return has_varargs or len(positional_params) >= minimum_positional


def apply_rule(
    rule: Callable,
    eqn: Any,
    known_invars,
    known_outvars,
    context: Any = None,
):
    if context is not None and supports_context_argument(rule):
        return rule(eqn, known_invars, known_outvars, context)
    return rule(eqn, known_invars, known_outvars)

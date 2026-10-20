"""
Utilities for inverse interpreter computations.

This module provides helper functions used by the inverse interpreter
and log-det tracking machinery.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
from jax._src import core as jax_core


def scan_arity(params: dict) -> tuple[int, int]:
    """Return ``(num_consts, num_carry)`` of a ``scan`` equation.

    JAX < 0.11 stores them as params; newer versions only keep the
    ``ft_in`` layout of (consts, carry, xs).
    """
    if "num_consts" in params:
        return params["num_consts"], params["num_carry"]
    consts, carry, _ = params["ft_in"].unpack()
    return len(consts), len(carry)


def is_inexact_value(value: Any) -> bool:
    """
    Check if a value has an inexact (floating-point or complex) dtype.

    This is used to determine whether log-det contributions should be
    computed for a variable - we only track log-det for continuous
    (inexact) values.

    Args:
        value: The value to check

    Returns:
        True if the value has an inexact dtype (float or complex),
        False otherwise (integers, booleans, etc.)
    """
    return jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact)


def eval_custom_inverse_jaxpr(
    eqn: Any,
    known_invars: Sequence[Any],
    known_outvars: Sequence[Any],
) -> tuple[list[Any], Any] | None:
    """Evaluate a ``custom_inverse_call`` equation's inverse jaxpr.

    The inverse takes the known outputs in place of the target inputs, so the
    outputs are spliced into the input list at the first target position.
    Returns the target variables and the inverse jaxpr's outputs, or ``None``
    when a non-target input is unknown and the inverse cannot be evaluated.
    """
    from probjax.core.custom_primitives.contracts import (
        parse_custom_inverse_call_params,
    )

    custom_params = parse_custom_inverse_call_params(eqn.params)
    inverse_jaxpr = custom_params.inverse_jaxpr_thunk()

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

    out = jax_core.eval_jaxpr(inverse_jaxpr.jaxpr, inverse_jaxpr.literals, *inputs)

    invars = [eqn.invars[index] for index in custom_params.target_in_indices]
    return invars, out

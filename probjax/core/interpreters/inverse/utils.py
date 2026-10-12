"""
Utilities for inverse interpreter computations.

This module provides helper functions used by the inverse interpreter
and log-det tracking machinery.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


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

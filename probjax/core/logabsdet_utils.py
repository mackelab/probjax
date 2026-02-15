"""
Utilities for log absolute determinant computations.

This module provides helper functions used by the inverse interpreter
and log-det tracking machinery.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


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

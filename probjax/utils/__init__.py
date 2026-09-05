"""Utility functions for numerical computation and optimization.

This module provides various utility functions including:

- ODE/SDE integration
- Linear algebra utilities
- Interpolation functions
- Graph utilities
- Special mathematical functions
- Root finding and optimization

Example
-------
>>> from probjax.utils import odeint
>>> import jax.numpy as jnp
>>>
>>> # Solve an ODE
>>> def f(t, y): return -y
>>> y0 = jnp.array([1.0])
>>> ts = jnp.linspace(0, 1, 100)
>>> solution = odeint(f, y0, ts)
"""

# ODE/SDE integration
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint

# Linear algebra
from probjax.utils.linalg import (
    cholesky_update,
    mv_diag_or_dense,
)

# Interpolation
from probjax.utils.interpolation import (
    linear_interpolation,
    polynomial_interpolation,
)

# Graph utilities
from probjax.utils.graph import (
    find_ancestors_jax,
    faithfull_mask,
)

# Special functions
from probjax.utils.special import (
    betaincinv,
    digammainv,
    gammaincinv,
)

# Solvers and root finding
from probjax.utils.solver import (
    newton_raphson,
    root,
)

# Function utilities
from probjax.utils.functions import (
    split_drift,
)

# JAX utilities
from probjax.utils.jaxutils import (
    ravel_args,
)

# Typing
from probjax.utils.typing import (
    Array,
    ArrayLike,
    PyTree,
    RngKey,
)

__all__ = [
    # ODE/SDE integration
    "odeint",
    "sdeint",
    # Linear algebra
    "cholesky_update",
    "mv_diag_or_dense",
    # Interpolation
    "linear_interpolation",
    "polynomial_interpolation",
    # Graph utilities
    "find_ancestors_jax",
    "faithfull_mask",
    # Special functions
    "betaincinv",
    "digammainv",
    "gammaincinv",
    # Solvers
    "newton_raphson",
    "root",
    # Function utilities
    "split_drift",
    # JAX utilities
    "ravel_args",
    # Typing
    "Array",
    "ArrayLike",
    "PyTree",
    "RngKey",
]

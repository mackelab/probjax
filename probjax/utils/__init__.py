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
>>> def f(t, y):
...     return -y
>>> y0 = jnp.array([1.0])
>>> ts = jnp.linspace(0, 1, 100)
>>> solution = odeint(f, y0, ts)
"""

# ODE/SDE integration
# Function utilities
from probjax.utils.functions import (
    split_drift,
)

# Graph utilities
from probjax.utils.graph import (
    faithfull_mask,
    find_ancestors_jax,
)

# Runtime type guards
from probjax.utils.guards import (
    call_if_callable,
    ensure_int,
    ensure_tuple,
    get_field,
    has_attrs,
    is_array_like,
    is_mapping,
    is_tracing,
    normalize_devices,
    normalize_rng,
    params_to_dict,
    replace_field,
)

# Interpolation
from probjax.utils.interpolation import (
    linear_interpolation,
    polynomial_interpolation,
)

# JAX utilities
from probjax.utils.jaxutils import (
    ravel_args,
)

# Linear algebra
from probjax.utils.linalg import (
    cholesky_update,
    mv_diag_or_dense,
)
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint

# Solvers and root finding
from probjax.utils.solver import (
    newton_raphson,
    root,
)

# Special functions
from probjax.utils.special import (
    betainccinv,
    betaincinv,
    digammainv,
    gammainccinv,
    gammaincinv,
    log1mexp,
    logdiffexp,
)

# Typing
from probjax.utils.typing import (
    Array,
    ArrayLike,
    ArrayLikeTree,
    ArrayTree,
    PyTree,
    RngKey,
    Scalar,
    ScalarLike,
    Shape,
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
    "betainccinv",
    "gammainccinv",
    "log1mexp",
    "logdiffexp",
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
    "ArrayTree",
    "ArrayLikeTree",
    "PyTree",
    "RngKey",
    "Scalar",
    "ScalarLike",
    "Shape",
]

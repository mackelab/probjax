from typing import NamedTuple

import jax.numpy as jnp


class AdaptiveParams(NamedTuple):
    """Parameters for adaptive ODE integration."""

    rtol: float = 1e-4
    atol: float = 1e-5
    mxstep: int = jnp.inf
    dtmin: float = 0.0
    dtmax: float = jnp.inf
    maxerror: float = 1.2
    safety: float = 0.95
    ifactor: float = 10.0
    dfactor: float = 0.1
    error_norm: float = 2

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class GeometricPath:
    """Geometric (tempered) path: logprior + t * loglikelihood."""

    is_geometric: bool = True

    def initial_param(self, **_):
        return 0.0

    def logdensity_fn(
        self, tempering_param, *, logprior_fn: Callable, loglikelihood_fn: Callable, **_
    ):
        def logdensity(x):
            return logprior_fn(x) + tempering_param * loglikelihood_fn(x)

        return logdensity


@dataclass(frozen=True)
class PartialPosteriorsPath:
    """Partial posterior (data tempering) path."""

    is_geometric: bool = False

    def initial_param(self, *, num_datapoints: int, **_):
        return jnp.zeros(num_datapoints)

    def logdensity_fn(
        self,
        tempering_param,
        *,
        partial_logposterior_factory: Callable,
        **_,
    ):
        return partial_logposterior_factory(tempering_param)

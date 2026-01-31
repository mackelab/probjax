from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array


@runtime_checkable
class FlowPairTrainingConfigProtocol(Protocol):
    """Training-time sampling for (t, r) pairs (for mean flow matching)."""

    t_min: float
    t_max: float

    def sample_times_pair(self, rng, shape: Tuple[int, ...]) -> tuple[Array, Array]: ...


@dataclass
class SigmoidPairFlowTrainingConfig(FlowPairTrainingConfigProtocol):
    """
    Pair sampling mirroring MeanFlowMatcher.noise_schedule.
    """

    percent_rt: float = 0.25
    mu_rt: float = -0.4
    scale_rt: float = 1.0
    mu_t: float = 0.0
    scale_t: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def sample_times_pair(self, rng, shape: Tuple[int, ...]) -> tuple[Array, Array]:
        batch_size = shape[0]
        batch_size_different = int(batch_size * self.percent_rt)
        batch_size_same = batch_size - batch_size_different

        rng_t, rng_r, rng_tr = jax.random.split(rng, 3)

        t1 = jax.nn.sigmoid(
            jax.random.normal(rng_t, (batch_size_different,) + shape[1:] + (1,))
            * self.scale_rt
            - self.mu_rt
        )
        r1 = jax.nn.sigmoid(
            jax.random.normal(rng_r, (batch_size_different,) + shape[1:] + (1,))
            * self.scale_rt
            - self.mu_rt
        )
        r1 = jnp.clip(t1 + r1, a_min=self.t_min, a_max=self.t_max)

        t2 = r2 = jax.nn.sigmoid(
            jax.random.normal(rng_tr, (batch_size_same,) + shape[1:] + (1,))
            * self.scale_t
            - self.mu_t
        )

        t = jnp.concatenate([t1, t2], axis=0)
        r = jnp.concatenate([r1, r2], axis=0)

        t = jnp.clip(t, self.t_min, self.t_max)
        r = jnp.clip(r, self.t_min, self.t_max)
        return t, r

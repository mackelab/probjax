"""Shared contract for generative model families.

Every generative model in :mod:`probjax.nn.generative` — normalizing flows,
flow matching, denoising diffusion, discrete diffusion — trains via
``loss(rng, data)`` and exposes itself as a
:class:`probjax.stats.base.DistributionAPI` via ``as_distribution()``.
Training loops and inference adapters can target this protocol instead of a
concrete family.
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Array

from probjax.stats.base import DistributionAPI
from probjax.utils.typing import RngKey

__all__ = ["GenerativeModelProtocol"]


@runtime_checkable
class GenerativeModelProtocol(Protocol):
    def loss(self, rng: RngKey, data: Array, *args, **kwargs) -> Array:
        """Monte-Carlo training loss for a batch of data."""
        ...

    def as_distribution(self, event_shape=None, **kwargs) -> DistributionAPI:
        """View this model as a distribution (sampling, and density if tractable)."""
        ...

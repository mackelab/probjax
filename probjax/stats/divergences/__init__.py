"""
Statistical divergences (:mod:`probjax.stats.divergences`)
=========================================================

This module provides implementations of various statistical divergences between
probability distributions.

Available divergences
-------------------
.. autosummary::
   :toctree: generated/

   kl_divergence
   wasserstein_distance
   sliced_wasserstein_distance
   max_slice_wasserstein_distance
"""

from probjax.stats.divergences.kl import kl_divergence
from probjax.stats.divergences.wasserstein import (
    max_slice_wasserstein_distance,
    sliced_wasserstein_distance,
    wasserstein_distance,
)

__all__ = [
    "kl_divergence",
    "wasserstein_distance",
    "sliced_wasserstein_distance",
    "max_slice_wasserstein_distance",
]

"""Twist functions for twisted SMC with a frozen diffusion model.

A twist is a lookahead: ``log_twist(t, particles)`` approximates, for each
particle in a batch, the log of the expected terminal potential given the
noisy state ``x_t``. SMC uses twist *differences* between successive steps
as incremental weights, so any positive twist keeps the final target exact;
better twists only improve particle efficiency.

This module is intentionally model-light: twists receive a ``denoise_fn``
callable ``(t, x_batch) -> xhat_batch`` (e.g. a bound
:meth:`DiffusionDenoiser.denoise`) instead of importing the model class.
Future twist sources (MC marginal estimates a la RDSMC, learned flow
functions a la Reinforced SMC) implement the same :class:`TwistFn`
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array, PyTree


@runtime_checkable
class TwistFn(Protocol):
    """Per-particle log-twist evaluated on a whole particle batch."""

    def log_twist(self, t: Array, particles: PyTree[Array]) -> Array:
        """Return shape ``(num_particles,)`` log-twist values at time ``t``."""
        ...


def _batch_size(particles: PyTree[Array]) -> int:
    return int(jax.tree.leaves(particles)[0].shape[0])


@dataclass(frozen=True)
class NullTwist:
    """Zero twist: reduces twisted SMC to prior simulation + final weighting."""

    def log_twist(self, t: Array, particles: PyTree[Array]) -> Array:
        n = _batch_size(particles)
        return jnp.zeros((n,))


@dataclass(frozen=True)
class DenoisedPotentialTwist:
    """Twist via the Tweedie/denoised estimate (TDS-style).

    ``log_twist(t, x_t) = temperature * log_potential_fn(xhat_0(x_t))``
    where ``xhat_0`` is the model posterior mean. The runner evaluates the
    final step (``t == t_min``) on the clean particles directly, so the
    terminal target stays exact even when denoising is imperfect.

    Args:
        denoise_fn: ``(t, x_batch) -> xhat_batch`` over the particle batch.
        log_potential_fn: single-particle ``xhat_0 -> scalar`` (vmapped
            internally over the batch).
        temperature: scales the potential (``1.0`` = exact tilt).
    """

    denoise_fn: Callable[[Array, PyTree[Array]], PyTree[Array]]
    log_potential_fn: Callable[[PyTree[Array]], Array]
    temperature: float = 1.0

    def log_twist(self, t: Array, particles: PyTree[Array]) -> Array:
        xhat = self.denoise_fn(t, particles)
        values = jax.vmap(self.log_potential_fn)(xhat)
        return self.temperature * jnp.asarray(values, dtype=jnp.float32)


__all__ = ["TwistFn", "NullTwist", "DenoisedPotentialTwist"]

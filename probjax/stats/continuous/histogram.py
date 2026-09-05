"""
Histogram distributions (:mod:`probjax.stats.histogram`)
=========================================================

Piecewise-constant densities on a uniform bin grid, parameterised by per-bin
logits so a neural network can emit them directly.

Two variants:

* :data:`histogram` — supported on ``[low, high]`` and exactly zero outside.
  Maximally flexible on a compact domain, but a hard constraint: any sample
  outside the range has zero likelihood, so it cannot be trained on data that
  escapes the grid. Choose the bounds from the data, as for the bounded splines.
* :data:`tailed_histogram` — the same grid with exponential tails glued on, so
  the support is all of R. This is the safe default when the range is not known
  in advance.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_continuous
from probjax.stats.constraints import Interval, real, strict_positive
from probjax.utils.typing import Array, RngKey

__all__ = ["histogram", "tailed_histogram"]

_EPS = 1e-12


def _bin_probs(logits: Array) -> Array:
    return jax.nn.softmax(jnp.asarray(logits), axis=-1)


def _grid(low, high, num_bins: int):
    """Uniform bin edges and width, broadcasting over the batch."""
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    width = (high - low) / num_bins
    steps = jnp.arange(num_bins + 1, dtype=jnp.result_type(low, high, jnp.float32))
    edges = low[..., None] + width[..., None] * steps
    return edges, width


def _bin_index(x, low, high, num_bins: int) -> Array:
    """Index of the bin containing ``x``, clipped into range."""
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    u = (jnp.asarray(x) - low) / (high - low)
    idx = jnp.floor(u * num_bins).astype(jnp.int32)
    return jnp.clip(idx, 0, num_bins - 1)


def _align(x, logits):
    """Broadcast ``x`` and per-bin ``logits`` to a common batch shape.

    ``logits`` carries a trailing bin axis that is *not* a batch axis, so the
    usual elementwise broadcast against ``x`` does not apply: the batch part is
    ``logits.shape[:-1]``. Lining them up explicitly keeps ``take_along_axis``
    well defined for any mix of scalar and batched parameters.
    """
    x = jnp.asarray(x)
    logits = jnp.asarray(logits)
    num_bins = logits.shape[-1]
    batch = jnp.broadcast_shapes(x.shape, logits.shape[:-1])
    return (
        jnp.broadcast_to(x, batch),
        jnp.broadcast_to(logits, batch + (num_bins,)),
        num_bins,
    )


class histogram_gen(rv_continuous):
    """Piecewise-constant density on ``[low, high]``.

    Parameters
    ----------
    logits : array_like
        Unnormalised per-bin log-masses, shape ``(..., num_bins)``. Bin masses
        are ``softmax(logits)`` and the density in bin ``b`` is
        ``mass_b / bin_width``.
    low, high : array_like
        Domain endpoints, ``low < high``.

    Notes
    -----
    The density is exactly zero outside ``[low, high]``, so ``logpdf`` returns
    ``-inf`` there. That is the definition of the family rather than a defect,
    but it means training data must lie inside the domain; use
    :data:`tailed_histogram` when it might not.
    """

    parameters = {"logits": real, "low": real, "high": real}

    @classmethod
    def param_sizes(cls, num_bins: int = 32, **kwargs) -> dict:
        """Trailing size of each parameter, given the bin count."""
        del kwargs
        if num_bins < 1:
            raise ValueError(f"num_bins must be positive; got {num_bins}.")
        return {"logits": num_bins, "low": 1, "high": 1}

    @classmethod
    def support(cls, logits=None, low=0.0, high=1.0, **kwargs):
        """Support is the closed domain ``[low, high]``."""
        return Interval(low, high)

    @classmethod
    def logpdf(cls, x, logits=None, low=0.0, high=1.0, **kwargs):
        """Log-density; ``-inf`` outside ``[low, high]``."""
        del kwargs
        x, logits, num_bins = _align(x, logits)
        _, width = _grid(low, high, num_bins)

        idx = _bin_index(x, low, high, num_bins)
        log_mass = jax.nn.log_softmax(logits, axis=-1)
        log_mass_at = jnp.take_along_axis(log_mass, idx[..., None], axis=-1)[..., 0]

        inside = (x >= jnp.asarray(low)) & (x <= jnp.asarray(high))
        return jnp.where(inside, log_mass_at - jnp.log(width), -jnp.inf)

    @classmethod
    def cdf(cls, x, logits=None, low=0.0, high=1.0, **kwargs):
        """Piecewise-linear CDF."""
        del kwargs
        x, logits, num_bins = _align(x, logits)
        probs = _bin_probs(logits)
        _, width = _grid(low, high, num_bins)

        cum = jnp.cumsum(probs, axis=-1)
        below = jnp.concatenate([jnp.zeros_like(cum[..., :1]), cum[..., :-1]], axis=-1)

        idx = _bin_index(x, low, high, num_bins)
        cum_below = jnp.take_along_axis(below, idx[..., None], axis=-1)[..., 0]
        mass_at = jnp.take_along_axis(probs, idx[..., None], axis=-1)[..., 0]

        edge = jnp.asarray(low) + width * idx
        frac = jnp.clip((x - edge) / width, 0.0, 1.0)
        inner = cum_below + mass_at * frac
        return jnp.where(x <= jnp.asarray(low), 0.0, jnp.where(x >= jnp.asarray(high), 1.0, inner))

    @classmethod
    def ppf(cls, q, logits=None, low=0.0, high=1.0, **kwargs):
        """Inverse of the piecewise-linear CDF."""
        del kwargs
        q, logits, num_bins = _align(q, logits)
        q = jnp.clip(q, 0.0, 1.0)
        probs = _bin_probs(logits)
        _, width = _grid(low, high, num_bins)

        cum = jnp.cumsum(probs, axis=-1)
        below = jnp.concatenate([jnp.zeros_like(cum[..., :1]), cum[..., :-1]], axis=-1)
        # first bin whose cumulative mass reaches q
        idx = jnp.clip(
            jnp.sum((cum[..., :] < q[..., None]).astype(jnp.int32), axis=-1),
            0,
            num_bins - 1,
        )
        cum_below = jnp.take_along_axis(below, idx[..., None], axis=-1)[..., 0]
        mass_at = jnp.take_along_axis(probs, idx[..., None], axis=-1)[..., 0]

        frac = (q - cum_below) / jnp.maximum(mass_at, _EPS)
        return jnp.asarray(low) + width * (idx + jnp.clip(frac, 0.0, 1.0))

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        logits=None,
        low=0.0,
        high=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Pick a bin by its mass, then a uniform point inside it."""
        del kwargs
        logits = jnp.asarray(logits)
        num_bins = logits.shape[-1]
        _, width = _grid(low, high, num_bins)

        batch_shape = jnp.broadcast_shapes(
            logits.shape[:-1], jnp.shape(low), jnp.shape(high)
        )
        out_shape = tuple(shape) + batch_shape

        key_bin, key_u = random.split(rng)
        full_logits = jnp.broadcast_to(logits, out_shape + (num_bins,))
        idx = random.categorical(key_bin, full_logits, axis=-1)
        u = random.uniform(key_u, out_shape)
        return jnp.asarray(low) + width * (idx + u)

    @classmethod
    def mean(cls, logits=None, low=0.0, high=1.0, **kwargs):
        del kwargs
        probs = _bin_probs(logits)
        num_bins = probs.shape[-1]
        _, width = _grid(low, high, num_bins)
        centres = jnp.asarray(low)[..., None] + width[..., None] * (
            jnp.arange(num_bins) + 0.5
        )
        return jnp.sum(probs * centres, axis=-1)


class tailed_histogram_gen(histogram_gen):
    """Piecewise-constant density on ``[low, high]`` with exponential tails.

    A fraction ``sigmoid(tail_logit)`` of the total mass is placed in two
    exponential tails, split evenly, glued at ``low`` and ``high``; the rest is
    distributed over the bins as in :data:`histogram`. The support is all of R,
    which makes this the safe variant when the data range is not known ahead of
    time.

    Parameters
    ----------
    logits : array_like
        Unnormalised per-bin log-masses, shape ``(..., num_bins)``.
    low, high : array_like
        Endpoints of the binned region.
    tail_logit : array_like
        Logit of the total mass assigned to the two tails.
    left_rate, right_rate : array_like
        Exponential decay rates of the tails, strictly positive.
    """

    parameters = {
        "logits": real,
        "low": real,
        "high": real,
        "tail_logit": real,
        "left_rate": strict_positive,
        "right_rate": strict_positive,
    }

    @classmethod
    def param_sizes(cls, num_bins: int = 32, **kwargs) -> dict:
        del kwargs
        if num_bins < 1:
            raise ValueError(f"num_bins must be positive; got {num_bins}.")
        return {
            "logits": num_bins,
            "low": 1,
            "high": 1,
            "tail_logit": 1,
            "left_rate": 1,
            "right_rate": 1,
        }

    @classmethod
    def support(cls, **kwargs):
        """Support is the whole real line."""
        return real

    @classmethod
    def _split_mass(cls, tail_logit):
        """(mass in the two tails each, mass in the binned region)."""
        tail = jax.nn.sigmoid(jnp.asarray(tail_logit))
        return 0.5 * tail, 1.0 - tail

    @classmethod
    def logpdf(
        cls,
        x,
        logits=None,
        low=0.0,
        high=1.0,
        tail_logit=-5.0,
        left_rate=1.0,
        right_rate=1.0,
        **kwargs,
    ):
        x = jnp.asarray(x)
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        half_tail, body = cls._split_mass(tail_logit)

        # `where` over an -inf branch still propagates a NaN gradient, so
        # evaluate the body on a clamped x and select the region afterwards.
        body_logpdf = histogram_gen.logpdf(
            jnp.clip(x, low, high), logits=logits, low=low, high=high
        )
        inside = jnp.log(jnp.maximum(body, _EPS)) + body_logpdf

        left_rate = jnp.asarray(left_rate)
        right_rate = jnp.asarray(right_rate)
        log_half = jnp.log(jnp.maximum(half_tail, _EPS))
        left = log_half + jnp.log(left_rate) - left_rate * (low - x)
        right = log_half + jnp.log(right_rate) - right_rate * (x - high)

        out = jnp.where(x < low, left, inside)
        return jnp.where(x > high, right, out)

    @classmethod
    def cdf(
        cls,
        x,
        logits=None,
        low=0.0,
        high=1.0,
        tail_logit=-5.0,
        left_rate=1.0,
        right_rate=1.0,
        **kwargs,
    ):
        x = jnp.asarray(x)
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        half_tail, body = cls._split_mass(tail_logit)

        body_cdf = histogram_gen.cdf(
            jnp.clip(x, low, high), logits=logits, low=low, high=high
        )
        inside = half_tail + body * body_cdf
        left = half_tail * jnp.exp(-jnp.asarray(left_rate) * (low - x))
        right = 1.0 - half_tail * jnp.exp(-jnp.asarray(right_rate) * (x - high))

        out = jnp.where(x < low, left, inside)
        return jnp.where(x > high, right, out)

    @classmethod
    def ppf(
        cls,
        q,
        logits=None,
        low=0.0,
        high=1.0,
        tail_logit=-5.0,
        left_rate=1.0,
        right_rate=1.0,
        **kwargs,
    ):
        q = jnp.clip(jnp.asarray(q), 0.0, 1.0)
        low = jnp.asarray(low)
        high = jnp.asarray(high)
        half_tail, body = cls._split_mass(tail_logit)

        # Invert each region analytically, then select.
        safe_left = jnp.clip(q / jnp.maximum(half_tail, _EPS), _EPS, 1.0)
        x_left = low + jnp.log(safe_left) / jnp.asarray(left_rate)

        safe_right = jnp.clip((1.0 - q) / jnp.maximum(half_tail, _EPS), _EPS, 1.0)
        x_right = high - jnp.log(safe_right) / jnp.asarray(right_rate)

        inner_q = jnp.clip((q - half_tail) / jnp.maximum(body, _EPS), 0.0, 1.0)
        x_mid = histogram_gen.ppf(inner_q, logits=logits, low=low, high=high)

        out = jnp.where(q < half_tail, x_left, x_mid)
        return jnp.where(q > 1.0 - half_tail, x_right, out)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        logits=None,
        low=0.0,
        high=1.0,
        tail_logit=-5.0,
        left_rate=1.0,
        right_rate=1.0,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Inverse-CDF sampling; every region of the quantile function is closed form."""
        logits = jnp.asarray(logits)
        batch_shape = jnp.broadcast_shapes(
            logits.shape[:-1],
            jnp.shape(low),
            jnp.shape(high),
            jnp.shape(tail_logit),
        )
        u = random.uniform(rng, tuple(shape) + batch_shape)
        return cls.ppf(
            u,
            logits=logits,
            low=low,
            high=high,
            tail_logit=tail_logit,
            left_rate=left_rate,
            right_rate=right_rate,
        )

    @classmethod
    def mean(cls, **kwargs):
        raise NotImplementedError(
            "tailed_histogram.mean is not implemented; use sample estimates."
        )


histogram = histogram_gen(name="histogram")
tailed_histogram = tailed_histogram_gen(name="tailed_histogram")

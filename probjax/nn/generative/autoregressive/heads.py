"""Conditional heads: what each ``p(x_i | x_<i)`` is.

A head turns the conditioner's ``num_params`` raw outputs for one dimension
into a univariate distribution. Every head is a few explicit lines: which
:mod:`probjax.stats` family it uses (:attr:`Head.dist`) and how raw outputs
become that family's parameters (:meth:`Head.params`). Each constraint is
neutral at zero -- a zero-initialised conditioner gives a standard normal, a
flat histogram, an identity spline -- so training starts from a sane density.

A new family is a subclass that sets ``dist`` and ``num_params`` and
implements ``params``; ``logpdf`` and ``sample`` follow from those.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
from jax import Array

from probjax.nn.generative.nflows.config import (
    _bin_positions,
    _component_offsets,
    _knot_slopes,
    _positive,
)
from probjax.stats import (
    categorical,
    histogram,
    logistic_mixture_kernel,
    mixture_kernel,
    norm,
    spline_normal,
    tailed_histogram,
)
from probjax.stats.base import rv_generic

__all__ = [
    "CategoricalHead",
    "Head",
    "HistogramHead",
    "LocScaleHead",
    "MixtureHead",
    "NormalHead",
    "SplineHead",
]


def _scale(raw: Array) -> Array:
    """Positive, bounded to ``[1e-1, 10]``-ish, and exactly 1 at ``raw == 0``."""
    return _positive(raw, 1e-3, "tanh", 10.0)


def _blocks(raw: Array, *sizes: int) -> list[Array]:
    """Split the trailing axis of ``raw`` into consecutive blocks."""
    edges = [0]
    for size in sizes:
        edges.append(edges[-1] + size)
    if edges[-1] != raw.shape[-1]:
        raise ValueError(f"expected {edges[-1]} raw parameters, got {raw.shape[-1]}.")
    return [raw[..., lo:hi] for lo, hi in zip(edges[:-1], edges[1:], strict=True)]


class Head:
    """Base class: ``dist``, ``num_params`` and ``params`` define a head.

    ``num_classes`` is ``None`` for continuous data; a discrete head sets it,
    which makes the data integer-valued and one-hot encodes the conditioner's
    input. ``support`` is ``(low, high)`` for a bounded family, so training
    data outside it can be rejected instead of scoring ``-inf``.
    """

    # Set by subclasses, as class attributes, fields or properties:
    #   dist        -- the probjax.stats family
    #   num_params  -- raw conditioner outputs per dimension
    num_classes = None
    support = None

    def params(self, raw: Array) -> dict[str, Any]:
        """``(..., num_params)`` raw outputs -> the family's parameters."""
        raise NotImplementedError

    def logpdf(self, raw: Array, x: Array) -> Array:
        return jnp.asarray(self.dist.logpdf(x, **self.params(raw)))

    def sample(self, key, raw: Array) -> Array:
        """One draw per leading index of ``raw``."""
        return jnp.asarray(self.dist._rvs_impl(key, **self.params(raw), shape=()))


@dataclass(frozen=True)
class LocScaleHead(Head):
    """Any location-scale family: ``LocScaleHead(laplace)``, ``LocScaleHead(t, {"df": 5.0})``.

    ``fixed`` supplies the family's other parameters as constants.
    """

    dist: rv_generic = norm
    fixed: Mapping[str, Any] = field(default_factory=dict)
    num_params = 2

    def params(self, raw):
        loc, scale = _blocks(raw, 1, 1)
        return {"loc": loc[..., 0], "scale": _scale(scale[..., 0]), **self.fixed}


@dataclass(frozen=True)
class NormalHead(LocScaleHead):
    """Gaussian conditionals -- the MADE head."""


@dataclass(frozen=True)
class MixtureHead(Head):
    """A ``num_components`` mixture: the flexible, KDE-like head.

    The component locations start spread over ``[-spread, spread]``. That is
    not cosmetic: from a zero-initialised conditioner every component would
    share a location, a scale *and* a gradient, so the mixture would stay a
    single kernel forever.
    """

    num_components: int = 10
    kernel: str = "norm"
    spread: float = 2.0

    def __post_init__(self):
        if self.kernel not in ("norm", "logistic"):
            raise ValueError(f"kernel must be 'norm' or 'logistic', got {self.kernel!r}.")

    @property
    def dist(self):
        return mixture_kernel if self.kernel == "norm" else logistic_mixture_kernel

    @property
    def num_params(self):
        return 3 * self.num_components

    def params(self, raw):
        k = self.num_components
        log_weights, locs, scales = _blocks(raw, k, k, k)
        offsets = _component_offsets(k, self.spread, raw.dtype)
        return {"log_weights": log_weights, "locs": locs + offsets, "scales": _scale(scales)}


@dataclass(frozen=True)
class HistogramHead(Head):
    """Piecewise-constant density on ``[low, high]``; ``tails`` adds exponential tails."""

    num_bins: int = 32
    low: float = -5.0
    high: float = 5.0
    tails: bool = True

    @property
    def dist(self):
        return tailed_histogram if self.tails else histogram

    @property
    def num_params(self):
        return self.num_bins + 3 if self.tails else self.num_bins

    @property
    def support(self):
        return None if self.tails else (self.low, self.high)

    def params(self, raw):
        bounds = {"low": self.low, "high": self.high}
        if not self.tails:
            (logits,) = _blocks(raw, self.num_bins)
            return {"logits": logits, **bounds}
        logits, tail, left, right = _blocks(raw, self.num_bins, 1, 1, 1)
        return {
            "logits": logits,
            **bounds,
            # Offset so the neutral state puts ~5% of the mass in the tails,
            # not sigmoid(0) = half of it.
            "tail_logit": tail[..., 0] - 3.0,
            "left_rate": _scale(left[..., 0]),
            "right_rate": _scale(right[..., 0]),
        }


@dataclass(frozen=True)
class SplineHead(Head):
    """A spline-warped normal: knots span ``[-latent_bound, latent_bound]`` and ``[-bound, bound]``.

    ``latent_bound`` defaults to ``bound``, so zero parameters give matching
    knots with unit slopes -- an identity spline, exactly a standard normal.
    The two boundary slopes, which set the tails, are damped by
    ``tail_logit_scale``; 1.0 leaves them undamped.
    """

    num_bins: int = 8
    bound: float = 5.0
    latent_bound: Optional[float] = None
    tail_logit_scale: float = 0.1
    dist = spline_normal

    def __post_init__(self):
        if not (math.isfinite(self.tail_logit_scale) and self.tail_logit_scale > 0):
            raise ValueError("tail_logit_scale must be finite and positive.")

    @property
    def num_params(self):
        # n + 1 knots spanning a fixed interval have n degrees of freedom.
        return 3 * self.num_bins + 1

    def params(self, raw):
        n = self.num_bins
        latent = self.bound if self.latent_bound is None else self.latent_bound
        x_widths, y_widths, slopes = _blocks(raw, n, n, n + 1)
        damping = jnp.ones(n + 1).at[jnp.array([0, -1])].set(self.tail_logit_scale)
        return {
            "x_pos": _bin_positions(x_widths, -latent, latent, 1e-4),
            "y_pos": _bin_positions(y_widths, -self.bound, self.bound, 1e-4),
            "knot_slopes": _knot_slopes(slopes * damping, 10.0),
        }


@dataclass(frozen=True)
class CategoricalHead(Head):
    """Categorical conditionals over integers ``0 .. num_classes - 1``."""

    num_classes: int
    dist = categorical

    @property
    def num_params(self):
        return self.num_classes

    def params(self, raw):
        return {"probs": jax.nn.softmax(raw, axis=-1)}

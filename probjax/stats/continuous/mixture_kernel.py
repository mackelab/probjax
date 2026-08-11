"""
Kernel mixture distribution (:mod:`probjax.stats.mixture_kernel`)
==================================================================

A location-scale mixture with a fixed number of components, parameterised by
*batched arrays* rather than a list of component objects.

This is the shape a neural network wants: one mixture per element, with all
components carried in a trailing axis, so a conditioner can emit
``(..., num_components)`` weights, locations and scales in one go.
:mod:`probjax.stats.mixture` is the general composition operator over a Python
list of frozen distributions and cannot express that.

Because the components are a fixed location-scale family the density, CDF and
sampler are all closed form; only the quantile function needs a solve.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

from probjax.stats.base import rv_continuous
from probjax.stats.constraints import real, strict_positive
from probjax.utils.typing import Array, ArrayLike, RngKey

__all__ = ["mixture_kernel", "logistic_mixture_kernel"]

_KERNELS = ("norm", "logistic")


def _kernel_logpdf(kind: str, z: Array) -> Array:
    """Standardised component log-density at ``z = (x - loc) / scale``."""
    if kind == "norm":
        return -0.5 * (z**2 + jnp.log(2.0 * jnp.pi))
    # logistic: log f(z) = -z - 2 log(1 + e^-z), written stably
    return -z - 2.0 * jax.nn.softplus(-z)


def _kernel_logcdf(kind: str, z: Array) -> Array:
    if kind == "norm":
        return jax.scipy.stats.norm.logcdf(z)
    return jax.nn.log_sigmoid(z)


def _kernel_standard_rvs(kind: str, rng: RngKey, shape: Tuple[int, ...]) -> Array:
    if kind == "norm":
        return random.normal(rng, shape)
    return random.logistic(rng, shape)


class mixture_kernel_gen(rv_continuous):
    """Mixture of ``num_components`` location-scale kernels.

    Parameters
    ----------
    log_weights : array_like
        Unnormalised component log-weights, shape ``(..., num_components)``.
        Only differences matter; they are normalised internally.
    locs : array_like
        Component locations, shape ``(..., num_components)``.
    scales : array_like
        Component scales, shape ``(..., num_components)``, strictly positive.

    Notes
    -----
    All three parameters share a trailing component axis, which is what makes
    this usable as a per-element conditional head. The support is the whole
    real line for either kernel.
    """

    parameters = {
        "log_weights": real,
        "locs": real,
        "scales": strict_positive,
    }

    #: Which location-scale kernel the components use. Every density method is
    #: a classmethod, and ``rv_generic.__init__`` assigns ``name`` on the class,
    #: so a variant is a *subclass* rather than a differently-configured
    #: instance -- see ``logistic_mixture_kernel_gen`` below.
    kernel: str = "norm"

    @classmethod
    def param_sizes(cls, num_components: int = 8, **kwargs) -> dict:
        """Trailing size of each parameter, given the component count."""
        del kwargs
        if num_components < 1:
            raise ValueError(f"num_components must be positive; got {num_components}.")
        return {
            "log_weights": num_components,
            "locs": num_components,
            "scales": num_components,
        }

    @classmethod
    def support(cls, log_weights=None, locs=None, scales=None, **kwargs):
        """Support of the kernel mixture."""
        return real

    @classmethod
    def _standardise(cls, x, locs, scales):
        x = jnp.asarray(x)
        locs = jnp.asarray(locs)
        scales = jnp.asarray(scales)
        return (x[..., None] - locs) / scales

    @classmethod
    def _log_mixing(cls, log_weights):
        return jax.nn.log_softmax(jnp.asarray(log_weights), axis=-1)

    @classmethod
    def _kernel_kind(cls, kwargs) -> str:
        return kwargs.get("kernel", cls.kernel)

    @classmethod
    def logpdf(cls, x, log_weights=None, locs=None, scales=None, **kwargs):
        """Log-density of the mixture at ``x``."""
        kind = cls._kernel_kind(kwargs)
        z = cls._standardise(x, locs, scales)
        log_comp = _kernel_logpdf(kind, z) - jnp.log(jnp.asarray(scales))
        return jax.nn.logsumexp(cls._log_mixing(log_weights) + log_comp, axis=-1)

    @classmethod
    def cdf(cls, x, log_weights=None, locs=None, scales=None, **kwargs):
        """Mixture CDF: the weighted sum of the component CDFs."""
        kind = cls._kernel_kind(kwargs)
        z = cls._standardise(x, locs, scales)
        log_cdf = cls._log_mixing(log_weights) + _kernel_logcdf(kind, z)
        return jnp.exp(jax.nn.logsumexp(log_cdf, axis=-1))

    @classmethod
    def logcdf(cls, x, log_weights=None, locs=None, scales=None, **kwargs):
        kind = cls._kernel_kind(kwargs)
        z = cls._standardise(x, locs, scales)
        return jax.nn.logsumexp(
            cls._log_mixing(log_weights) + _kernel_logcdf(kind, z), axis=-1
        )

    @classmethod
    def ppf(cls, q, log_weights=None, locs=None, scales=None, **kwargs):
        """Quantile function, by bisection on the (strictly increasing) CDF.

        A mixture CDF has no closed-form inverse. The solver is the same one the
        monotone bijectors use, so it inherits their NaN-safety and residual
        check rather than silently returning a bracket edge.
        """
        from probjax.stats.bijective.monotone import _solve_increasing

        q = jnp.asarray(q)

        def cdf_at(t):
            return cls.cdf(t, log_weights, locs, scales, **kwargs)

        root = _solve_increasing(cdf_at, jnp.clip(q, 1e-7, 1.0 - 1e-7))
        # The solver reports failure as NaN; the boundaries genuinely have no
        # finite root, so answer them directly rather than letting the clip
        # above pass off an interior point as the true quantile.
        root = jnp.where(q <= 0.0, -jnp.inf, root)
        return jnp.where(q >= 1.0, jnp.inf, root)

    @classmethod
    def _rvs_impl(
        cls,
        rng: RngKey,
        log_weights=None,
        locs=None,
        scales=None,
        shape: Tuple[int, ...] = (),
        **kwargs,
    ) -> Array:
        """Draw a component, then draw from it."""
        kind = cls._kernel_kind(kwargs)
        log_weights = jnp.asarray(log_weights)
        locs = jnp.asarray(locs)
        scales = jnp.asarray(scales)

        batch_shape = jnp.broadcast_shapes(
            log_weights.shape[:-1], locs.shape[:-1], scales.shape[:-1]
        )
        out_shape = tuple(shape) + batch_shape

        key_pick, key_draw = random.split(rng)
        # categorical reduces the trailing axis, so broadcast the logits up to
        # the full output shape first.
        logits = jnp.broadcast_to(
            cls._log_mixing(log_weights), out_shape + log_weights.shape[-1:]
        )
        idx = random.categorical(key_pick, logits, axis=-1)

        loc = jnp.take_along_axis(
            jnp.broadcast_to(locs, logits.shape), idx[..., None], axis=-1
        )[..., 0]
        scale = jnp.take_along_axis(
            jnp.broadcast_to(scales, logits.shape), idx[..., None], axis=-1
        )[..., 0]
        return _kernel_standard_rvs(kind, key_draw, out_shape) * scale + loc

    @classmethod
    def mean(cls, log_weights=None, locs=None, scales=None, **kwargs):
        """Mixture mean; both kernels have a zero-mean standardised form."""
        del kwargs
        weights = jnp.exp(cls._log_mixing(log_weights))
        return jnp.sum(weights * jnp.asarray(locs), axis=-1)

    @classmethod
    def var(cls, log_weights=None, locs=None, scales=None, **kwargs):
        """Law of total variance over the component indicator."""
        kind = cls._kernel_kind(kwargs)
        weights = jnp.exp(cls._log_mixing(log_weights))
        locs = jnp.asarray(locs)
        scales = jnp.asarray(scales)
        # variance of the standardised kernel
        unit_var = 1.0 if kind == "norm" else (jnp.pi**2) / 3.0
        mean = jnp.sum(weights * locs, axis=-1, keepdims=True)
        within = jnp.sum(weights * unit_var * scales**2, axis=-1)
        between = jnp.sum(weights * (locs - mean) ** 2, axis=-1)
        return within + between


class logistic_mixture_kernel_gen(mixture_kernel_gen):
    """Mixture of logistic kernels; heavier-tailed than the Gaussian variant."""

    kernel = "logistic"


mixture_kernel = mixture_kernel_gen(name="mixture_kernel")
logistic_mixture_kernel = logistic_mixture_kernel_gen(name="logistic_mixture_kernel")

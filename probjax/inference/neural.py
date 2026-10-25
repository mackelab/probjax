"""Bridge from :mod:`probjax.nn` models to the inference stack.

This is the one module in :mod:`probjax.inference` that knows about flax. It
splits a model into a pure function and its trainable parameters, so kernels
carry the weights in ``params`` and :func:`~probjax.inference.fit_adaptor` can
train them inside a compiled scan:

>>> proposal, proposal_params = density(maf(dim, 4, rngs=nnx.Rngs(0)))
>>> kernel = imh(logdensity_fn, proposal=proposal)
>>> result = warmup(key, kernel, kernel.init(key, x0), proposal_params, 2000)

It calls the model's own methods rather than ``model.as_dist()``: the
distribution view runs through ``jax.export``, which cannot be ``vmap``-ed,
and chains are.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from flax import nnx

from probjax.inference.base import Density

__all__ = ["bijection", "density"]


def _split(model: Any) -> tuple[Any, Any, Any]:
    # Only ``nnx.Param`` leaves are trained; buffers such as a flow's
    # standardisation statistics stay fixed.
    return nnx.split(model, nnx.Param, ...)


def density(model: Any) -> tuple[Density, Any]:
    """Split a tractable-density model into a :class:`Density` and its params.

    Args:
        model: a :mod:`probjax.nn` model with a log-density -- a normalizing
            flow (``maf``, ``nsf``, ...) or an autoregressive model.

    Returns:
        ``(density, params)``. Flow samples are reparameterised, so
        ``density.sample`` is differentiable in ``params``.
    """
    if not hasattr(model, "_logpdf"):
        raise TypeError(f"{type(model).__name__} has no tractable log-density.")
    graphdef, params, rest = _split(model)

    def logpdf(x: Any, params: Any) -> Any:
        return jnp.squeeze(nnx.merge(graphdef, params, rest)._logpdf(x))

    def sample(key: Any, params: Any) -> Any:
        merged = nnx.merge(graphdef, params, rest)
        if hasattr(merged, "transform"):
            return merged.transform(merged._sample_base(key, (), None))
        return merged.sample(key)

    return Density(sample, logpdf), params


def bijection(flow: Any) -> tuple[Callable[[Any, Any], tuple[Any, Any]], Any]:
    """Split a normalizing flow into its map ``z -> (x, log|det J|)`` and params.

    This is what :func:`~probjax.inference.neutra` needs.
    """
    graphdef, params, rest = _split(flow)

    def forward(z: Any, params: Any) -> tuple[Any, Any]:
        merged = nnx.merge(graphdef, params, rest)
        x = merged.transform(z)
        # The flow does not expose its Jacobian, but it does not have to:
        # log q(T(z)) = log base(z) - log|det J_T(z)|, which is exact and costs
        # one density call.
        log_det = jnp.sum(merged.base_dist.logpdf(z)) - jnp.squeeze(merged._logpdf(x))
        return x, log_det

    return forward, params

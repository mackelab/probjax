"""Bridge from :mod:`probjax.nn` models to the inference stack.

This is the one module in :mod:`probjax.inference` that knows about flax. It
splits a model into a pure function and its trainable parameters, so kernels
carry the weights in ``params`` and :func:`~probjax.inference.fit_adaptor` can
train them inside a compiled scan:

>>> proposal, proposal_params = density(maf(dim, 4, rngs=nnx.Rngs(0)))
>>> kernel = imh(logdensity_fn, proposal=proposal)
>>> result = warmup(key, kernel, kernel.init(key, x0), proposal_params, 2000)

It calls the model's plain methods (``logpdf``, ``sample_base``,
``sample_from_noise``) with the params passed explicitly, so kernels can
train them inside a compiled scan and ``vmap`` over chains.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.inference.base import Density
from probjax.inference.smc.diffusion import Guide

__all__ = ["bijection", "density", "diffusion_guide"]


def _split(model: Any) -> tuple[Any, Any, Any]:
    # Only ``nnx.Param`` leaves are trained; buffers such as a flow's
    # standardisation statistics stay fixed.
    return nnx.split(model, nnx.Param, ...)


def _batch_of_one(tree: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf[None], tree)


def _first(tree: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf[0], tree)


def density(model: Any) -> tuple[Density, Any]:
    """Split a tractable-density model into a :class:`Density` and its params.

    Args:
        model: a :mod:`probjax.nn` model with a log-density -- a normalizing
            flow (``maf``, ``nsf``, ...) or an autoregressive model.

    Returns:
        ``(density, params)``. Flow samples are reparameterised, so
        ``density.sample`` is differentiable in ``params``.
    """
    if not hasattr(model, "logpdf"):
        raise TypeError(f"{type(model).__name__} has no tractable log-density.")
    spec = model.as_dist().event_spec
    graphdef, params, rest = _split(model)

    def logpdf(x: Any, params: Any) -> Any:
        merged = nnx.merge(graphdef, params, rest)
        return merged.logpdf(_batch_of_one(x))[0]

    def sample(key: Any, params: Any) -> Any:
        merged = nnx.merge(graphdef, params, rest)
        # One key for both: a flow draws from the noise and ignores the key,
        # an autoregressive model draws from the key and ignores the noise.
        noise = merged.sample_base(key, (1,), spec)
        return _first(merged.sample_from_noise(noise, key))

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
        log_q = merged.logpdf(_batch_of_one(x))[0]
        log_det = jnp.sum(merged.base_dist.logpdf(z)) - log_q
        return x, log_det

    return forward, params


def diffusion_guide(
    model: Any,
    log_potential_fn: Callable[[Any], Any],
    *,
    event_spec: Any = None,
    twist: bool = True,
    inputs: dict[str, Any] | None = None,
) -> tuple[Guide, Any]:
    """Steer a pretrained diffusion model towards ``prior(x) * exp(V(x))``.

    The guide for :func:`~probjax.inference.diffusion_smc`, after the twisted
    diffusion sampler (Wu et al. 2023). The twist is ``V`` at the model's
    denoised estimate of the clean point, faded in as the noise goes, and its
    gradient steers the model's reverse SDE. The twist is a heuristic: it
    decides how many particles are needed, never what they converge to.

    Args:
        model: a trained diffusion model with a ``solver_cfg``, e.g. a
            :class:`~probjax.nn.generative.diffusion.model.DiffusionDenoiser`.
        log_potential_fn: ``V`` for one clean point; scale it to temper.
        event_spec: shape (or pytree of shapes) of one point; defaults to the
            model's.
        twist: ``False`` neither looks ahead nor steers: it simulates the
            prior and weights by the final potential, a baseline.
        inputs: named model inputs such as ``context``.

    Returns:
        ``(guide, params)``; the params are the model's weights.
    """
    inputs = dict(inputs or {})
    event_spec = model.event_spec if event_spec is None else event_spec
    if event_spec is None:
        raise TypeError("Pass event_spec, or set one on the model.")
    spec = model._normalize_event_spec(event_spec)
    graphdef, params, rest = _split(model)
    potential = jax.vmap(log_potential_fn)

    def sample_base(key: Any, t: Any, num_particles: int, params: Any) -> Any:
        merged = nnx.merge(graphdef, params, rest)
        particles = merged.sample_base(key, (num_particles,), spec, t_max=t)
        return particles, jnp.zeros(num_particles)

    def evaluate(key: Any, t: Any, particles: Any, params: Any) -> Any:
        del key
        merged = nnx.merge(graphdef, params, rest)
        drift_fn, diffusion_fn = merged._build_sde_drift_and_diffusion(inputs)
        alpha, sigma = merged.scale_fn(t), merged.std_fn(t)
        # Fade the twist in with the signal left at time t: at high noise the
        # denoised point is a poor stand-in for the clean one, and the full
        # potential there collapses the weights. The square fades in later
        # than the plain fraction, which measured better on sharp potentials.
        fade = (alpha**2 / (alpha**2 + sigma**2)) ** 2

        def log_twist_fn(particles):
            return fade * potential(merged.denoise(t, particles, **inputs))

        if twist:
            log_twist, pullback = jax.vjp(log_twist_fn, particles)
            (guidance,) = pullback(jnp.ones_like(log_twist))
        else:
            log_twist = jnp.zeros(jax.tree.leaves(particles)[0].shape[0])
            guidance = jax.tree.map(jnp.zeros_like, particles)
        return (
            log_twist,
            drift_fn(t, particles),
            diffusion_fn(t, particles),
            guidance,
        )

    def log_target(particles: Any, params: Any) -> Any:
        del params
        return potential(particles)

    return Guide(sample_base, evaluate, log_target), params

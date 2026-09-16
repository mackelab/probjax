"""Shared preamble helpers for the score-matching loss builders."""

import jax


def _require_rng(rng):
    assert rng is not None, "loss_fn does require rngs, pass them to function kwargs."


def _sample_eps(rng, shape):
    return jax.random.normal(rng, shape=shape)


def _resolve_weight(weight_fn, times):
    return weight_fn(times) if weight_fn is not None else None


def _pop_axis(kwargs, axis):
    return kwargs.pop("axis", axis)


def _perturb(mean_fn, std_fn, times, x, rng):
    """Diffuse ``x`` as ``mean + std * eps``; return ``(perturbed, eps)``."""
    mean = mean_fn(times, x)
    std = std_fn(times, x)
    eps = _sample_eps(rng, x.shape)
    return mean + std * eps, eps

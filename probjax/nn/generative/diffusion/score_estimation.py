"""Monte-Carlo score and marginal estimation via the denoising identity.

For an unnormalized target ``exp(log_target_fn(x_0))`` and a forward
noising ``x_t = alpha(t) * x_0 + sigma(t) * eps``, the denoising score
identity (DSI) writes the marginal score as a posterior expectation over
the denoising posterior
``pi(x_0 | x_t) propto exp(log_target_fn(x_0)) * N(x_t | alpha * x_0, s^2)``::

    grad_{x_t} log pi_t(x_t) = E[(alpha * x_0 - x_t) / s^2 | x_t].

:func:`is_score_and_marginal` estimates the score with importance sampling
from ``q(u | x_t) = N(x_t / alpha, (s / alpha)^2)`` and reuses the same
weights for an unbiased marginal estimate ``pi_hat(x_t) = mean(w)`` of
``Z * pi_t(x_t)``. Those byproducts are exactly what reverse-diffusion SMC
needs for its intermediate targets -- score proposals and twist targets
from one set of target evaluations.

Single-particle API throughout (pytrees supported); batch over particles
with :func:`jax.vmap`. Scale parameters come from the caller's schedule
(``model.scale_fn(t)``, ``model.std_fn(t)``), so every noise schedule in
:mod:`probjax.nn.generative.diffusion.config` works unchanged.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array, PyTree, RngKey

_LOG_2PI = jnp.log(2.0 * jnp.pi)


class ScoreEstimationInfo(NamedTuple):
    ess: Array  # inner importance-sample ESS
    log_marginal: Array  # log mean(w), unbiased for log(Z * pi_t(x_t))


def _sum_event_dims(leaf: Array) -> Array:
    return jnp.sum(leaf, axis=tuple(range(1, leaf.ndim))) if leaf.ndim > 1 else leaf


def _gaussian_logpdf_sum(x: PyTree[Array], mean: PyTree[Array], std: Array) -> Array:
    """Summed log ``N(x | mean, std^2)`` over event dims, result ``(M,)``."""
    total = None
    for xl, ml in zip(jax.tree.leaves(x), jax.tree.leaves(mean), strict=True):
        term = -0.5 * ((xl - ml) / std) ** 2 - jnp.log(std) - 0.5 * _LOG_2PI
        reduced = _sum_event_dims(term)
        total = reduced if total is None else total + reduced
    if total is None:
        raise ValueError("x and mean must contain at least one leaf.")
    return total


def is_score_and_marginal(
    key: RngKey,
    x_t: PyTree[Array],
    t: Array,
    *,
    log_target_fn: Callable[[PyTree[Array]], Array],
    alpha: Array,
    sigma: Array,
    num_inner: int,
    max_score_norm: float | None = None,
) -> tuple[PyTree[Array], Array, ScoreEstimationInfo]:
    """IS estimate of the DSI score and marginal at one noisy state.

    Args:
        key: RNG key for the inner importance samples.
        x_t: single-particle noisy state pytree (no batch axis).
        t: time (kept for API symmetry; ``alpha``/``sigma`` carry the values).
        log_target_fn: single-particle unnormalized log-density.
        alpha: forward scale at ``t`` (``schedule.scale(t)``).
        sigma: forward std at ``t`` (``schedule.std(t)``), must be positive.
        num_inner: inner importance samples ``M``.
        max_score_norm: optionally clip the global score norm (paper B.3.2).

    Returns:
        ``(score, log_marginal, info)`` with ``score`` a pytree like
        ``x_t`` and ``log_marginal`` the log mean importance weight.
    """
    del t  # alpha/sigma already encode the schedule at t.
    if num_inner < 1:
        raise ValueError("num_inner must be positive.")
    alpha = jnp.asarray(alpha, dtype=jnp.float32)
    sigma = jnp.asarray(sigma, dtype=jnp.float32)
    if not bool(jnp.all(sigma > 0)):
        raise ValueError("sigma must be positive.")

    x_leaves, treedef = jax.tree.flatten(x_t)
    prop_std = sigma / alpha
    prop_loc = treedef.unflatten([leaf / alpha for leaf in x_leaves])

    eps_keys = jax.random.split(key, len(x_leaves))
    eps = [
        jax.random.normal(k, (num_inner,) + leaf.shape, jnp.float32)
        for k, leaf in zip(eps_keys, x_leaves, strict=True)
    ]
    prop_loc_leaves = jax.tree.leaves(prop_loc)
    inner = treedef.unflatten([
        loc + prop_std * e for loc, e in zip(prop_loc_leaves, eps, strict=True)
    ])

    log_prior = jax.vmap(log_target_fn)(inner)
    log_lik = _gaussian_logpdf_sum(
        x_t, treedef.unflatten([alpha * leaf for leaf in jax.tree.leaves(inner)]), sigma
    )
    log_q = _gaussian_logpdf_sum(inner, prop_loc, prop_std)
    log_w = (
        jnp.asarray(log_prior, dtype=jnp.float32)
        + jnp.asarray(log_lik, dtype=jnp.float32)
        - jnp.asarray(log_q, dtype=jnp.float32)
    )

    log_marginal = jax.scipy.special.logsumexp(log_w) - jnp.log(num_inner)
    weights = jnp.exp(log_w - jax.scipy.special.logsumexp(log_w))

    inner_leaves = jax.tree.leaves(inner)
    x_leaves_b = x_leaves  # event-shaped, broadcast against (M, ...)

    def _weighted_mean(term_leaf: Array) -> Array:
        w = weights.reshape((num_inner,) + (1,) * (term_leaf.ndim - 1))
        return jnp.sum(w * term_leaf, axis=0)

    score = treedef.unflatten([
        _weighted_mean((alpha * u_leaf - x_leaf) / sigma**2)
        for u_leaf, x_leaf in zip(inner_leaves, x_leaves_b, strict=True)
    ])

    if max_score_norm is not None:
        norm = jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(score)))
        clipped = max_score_norm / jnp.maximum(norm, max_score_norm)
        score = jax.tree.map(lambda leaf: leaf * clipped, score)

    ess = jnp.exp(
        2.0 * jax.scipy.special.logsumexp(log_w)
        - jax.scipy.special.logsumexp(2.0 * log_w)
    )
    return score, log_marginal, ScoreEstimationInfo(ess=ess, log_marginal=log_marginal)


__all__ = ["ScoreEstimationInfo", "is_score_and_marginal"]

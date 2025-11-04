"""
Kullback-Leibler divergence implementations.
"""

import jax
import jax.numpy as jnp

from probjax.stats import (
    # Discrete distributions
    bernoulli,
    beta,
    categorical,
    cauchy,
    dirichlet,
    expon,
    gamma,
    laplace,
    multivariate_normal,
    # Continuous distributions
    norm,
    poisson,
    rv_continuous,
    rv_discrete,
    rv_generic,
)
from probjax.stats.divergences.base import divergence, register_divergence

__all__ = ["kl_divergence"]

NAME = "kl"


def kl_divergence(p, q, mc_samples=0, key=None):
    """Compute Kullback-Leibler divergence :math:`KL(p \\| q)` between two distributions.

    .. math::

        KL(p \\| q) = \\int p(x) \\log\frac {p(x)} {q(x)} \\,dx

    Args:
        p (rv_generic): First distribution
        q (rv_generic): Second distribution
        mc_samples (int): Number of samples to use for Monte Carlo approximation.
            Defaults to 0. Then only analytic expressions.
        key (jax.Array): Key for random number generation.
            Defaults to None. Only required if mc_samples > 0.

    Returns:
        jax.Array: A batch of KL divergences of shape `batch_shape`.
    """
    return divergence(NAME, p, q, mc_samples=mc_samples, key=key)


@register_divergence(NAME, rv_generic, rv_generic)
def _kl_generic(p, q, mc_samples=0, key=None):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL divergence between distributions with different event shapes not"
            "supported"
        )

    assert mc_samples >= 0, (
        "For general distributions we require mc_samples >= 0, to evaluate a Monte "
        "Carlo approximation of the KL divergence."
    )
    assert key is not None, "Key must be provided if mc_samples > 0"

    samples = p.rvs(key, (mc_samples,))
    log_prob_p = p.logpdf(samples)
    log_prob_q = q.logpdf(samples)
    return (log_prob_p - log_prob_q).mean(0)


@register_divergence(NAME, rv_continuous, rv_continuous)
def _kl_continuous_continuous(p, q, mc_samples=0, key=None):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL divergence between distributions with different event shapes not"
            "supported"
        )

    assert mc_samples >= 0, (
        "For general distributions we require mc_samples >= 0, to evaluate a Monte "
        "Carlo approximation of the KL divergence."
    )
    assert key is not None, "Key must be provided if mc_samples > 0"

    samples = p.rvs(key, (mc_samples,))
    log_prob_p = p.logpdf(samples)
    log_prob_q = q.logpdf(samples)
    return (log_prob_p - log_prob_q).mean(0)


@register_divergence(NAME, rv_discrete, rv_discrete)
def _kl_discrete_discrete(p, q, mc_samples=0, key=None):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "KL divergence between distributions with different event shapes not"
            "supported"
        )

    assert mc_samples >= 0, (
        "For general distributions we require mc_samples >= 0, to evaluate a Monte "
        "Carlo approximation of the KL divergence."
    )
    assert key is not None, "Key must be provided if mc_samples > 0"

    samples = p.rvs(key, (mc_samples,))
    log_prob_p = p.logpdf(samples)
    log_prob_q = q.logpdf(samples)
    return (log_prob_p - log_prob_q).mean(0)


@register_divergence(NAME, norm, norm)
def _kl_normal_normal(p, q, mc_samples=0, key=None):
    loc_p, scale_p = p.loc, p.scale
    loc_q, scale_q = q.loc, q.scale
    t1 = jnp.log(scale_q / scale_p)
    t2 = (scale_p**2 + (loc_p - loc_q) ** 2) / (2 * scale_q**2) - 0.5
    return t1 + t2


@register_divergence(NAME, multivariate_normal, multivariate_normal)
def _kl_mvn_mvn(p, q, mc_samples=0, key=None):
    loc_p, scale_p = p.loc, p.scale_tril
    loc_q, scale_q = q.loc, q.scale_tril
    t1 = jnp.linalg.slogdet(scale_p)[1] - jnp.linalg.slogdet(scale_q)[1]
    t2 = (
        jnp.trace(jnp.linalg.solve(scale_q, scale_p))
        + (loc_q - loc_p).T @ jnp.linalg.solve(scale_q, loc_q - loc_p)
        - p.event_shape[0]
    )
    return t1 + t2


@register_divergence(NAME, bernoulli, bernoulli)
def _kl_bernoulli_bernoulli(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    probs_q = q.probs
    t1 = probs_p * jnp.log(probs_p / probs_q)
    t2 = (1 - probs_p) * jnp.log((1 - probs_p) / (1 - probs_q))
    return t1 + t2


@register_divergence(NAME, categorical, categorical)
def _kl_categorical_categorical(p, q, mc_samples=0, key=None):
    probs_p = p.probs
    probs_q = q.probs
    return (probs_p * jnp.log(probs_p / probs_q)).sum(-1)


@register_divergence(NAME, dirichlet, dirichlet)
def _kl_dirichlet_dirichlet(p, q, mc_samples=0, key=None):
    alpha_p = p.concentration
    alpha_q = q.concentration
    t1 = jax.scipy.special.gammaln(alpha_p.sum(-1)) - jax.scipy.special.gammaln(
        alpha_q.sum(-1)
    )
    t2 = jax.scipy.special.gammaln(alpha_q).sum(-1) - jax.scipy.special.gammaln(
        alpha_p
    ).sum(-1)
    t3 = ((alpha_p - alpha_q) * (1 / alpha_q - 1 / alpha_p)).sum(-1)
    return t1 + t2 + t3


@register_divergence(NAME, gamma, gamma)
def _kl_gamma_gamma(p, q, mc_samples=0, key=None):
    alpha_p, beta_p = p.concentration, p.rate
    alpha_q, beta_q = q.concentration, q.rate
    t1 = jax.scipy.special.gammaln(alpha_p) - jax.scipy.special.gammaln(alpha_q)
    t2 = (alpha_p - alpha_q) * (jax.scipy.special.digamma(alpha_p) - beta_p / alpha_p)
    t3 = (beta_p - beta_q) * (alpha_p / beta_p - 1)
    return t1 + t2 + t3


@register_divergence(NAME, beta, beta)
def _kl_beta_beta(p, q, mc_samples=0, key=None):
    alpha_p, beta_p = p.concentration1, p.concentration0
    alpha_q, beta_q = q.concentration1, q.concentration0
    t1 = (
        jax.scipy.special.gammaln(alpha_p)
        + jax.scipy.special.gammaln(beta_p)
        - jax.scipy.special.gammaln(alpha_p + beta_p)
    )
    t2 = (
        jax.scipy.special.digamma(alpha_p) - jax.scipy.special.digamma(alpha_p + beta_p)
    ) * (alpha_p - alpha_q)
    t3 = (
        jax.scipy.special.digamma(beta_p) - jax.scipy.special.digamma(alpha_p + beta_p)
    ) * (beta_p - beta_q)
    return t1 + t2 + t3


@register_divergence(NAME, expon, expon)
def _kl_exponential_exponential(p, q, mc_samples=0, key=None):
    rate_p = p.rate
    rate_q = q.rate
    t1 = jnp.log(rate_p / rate_q)
    t2 = rate_p / rate_q - 1
    return t1 + t2


@register_divergence(NAME, laplace, laplace)
def _kl_laplace_laplace(p, q, mc_samples=0, key=None):
    loc_p, scale_p = p.loc, p.scale
    loc_q, scale_q = q.loc, q.scale
    t1 = jnp.log(scale_p / scale_q)
    t2 = (scale_p / scale_q) + (loc_p - loc_q).abs() / scale_q - 1
    return t1 + t2


@register_divergence(NAME, poisson, poisson)
def _kl_poisson_poisson(p, q, mc_samples=0, key=None):
    rate_p = p.rate
    rate_q = q.rate
    t1 = rate_p * jnp.log(rate_p / rate_q)
    t2 = rate_p - rate_q
    return t1 + t2


@register_divergence(NAME, cauchy, cauchy)
def _kl_cauchy_cauchy(p, q, mc_samples=0, key=None):
    loc_p, scale_p = p.loc, p.scale
    loc_q, scale_q = q.loc, q.scale
    t1 = jnp.log((scale_p + scale_q) ** 2 + (loc_p - loc_q) ** 2)
    t2 = jnp.log(4 * scale_p * scale_q)
    return t1 - t2


@register_divergence(NAME, dirichlet, dirichlet)
def _kl_dirichlet_dirichlet(p, q, mc_samples=0, key=None):
    alpha_p = p.concentration
    alpha_q = q.concentration
    t1 = jax.scipy.special.gammaln(alpha_p.sum(-1)) - jax.scipy.special.gammaln(
        alpha_q.sum(-1)
    )
    t2 = jax.scipy.special.gammaln(alpha_q).sum(-1) - jax.scipy.special.gammaln(
        alpha_p
    ).sum(-1)
    t3 = ((alpha_p - alpha_q) * (1 / alpha_q - 1 / alpha_p)).sum(-1)
    return t1 + t2 + t3

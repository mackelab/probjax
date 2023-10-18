import jax
import jax.numpy as jnp

from probjax.distributions.divergences.divergence import register_divergence, divergence
from probjax import distributions as dist

__all__ = ["wasserstein_distance"]

NAME = "wasserstein"


def wasserstein_distance(p, q, mc_samples=0, key=None, order=2):
    return divergence(NAME, p, q, mc_samples=mc_samples, key=key, order=order)


def _1d_wasserstein(p, q, mc_samples=0, key=None, order=2):
    eval_points = jnp.linspace(0, 1, mc_samples)
    f1 = p.icdf(eval_points)
    f2 = q.icdf(eval_points)
    dist = jnp.abs(f1 - f2) ** order
    return jnp.trapz(dist, eval_points)


def _wasserstein_generic(p, q, mc_samples=0, key=None, order=2):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "Wasserstein distance between distributions with different event shapes not supported"
        )

    assert (
        mc_samples >= 0
    ), "For general distirbutions we require mc_samples >= 0, to evaluate a Monte Carlo approximation of the Wasserstein distance."

    if sum(p.event_shape) > 1:
        raise ValueError(
            "Wasserstein distance between multivariate distributions not supported"
        )
    else:
        return _1d_wasserstein(p, q, mc_samples=mc_samples, key=key, order=order)

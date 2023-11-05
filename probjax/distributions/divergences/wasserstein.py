import jax
import jax.numpy as jnp

from probjax.distributions.divergences.divergence import register_divergence, divergence
from probjax import distributions as dist

from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

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

def _1d_wasserstein_without_cdf(p, q, mc_samples=0, key=None, order=2):
    samples_p = p.sample(key, (mc_samples,))
    samples_q = q.sample(key, (mc_samples,))
    sorted_samples_p = jnp.sort(samples_p)
    sorted_samples_q = jnp.sort(samples_q)
    dist = jnp.abs(sorted_samples_p - sorted_samples_q) ** order
    return jnp.mean(dist)

@jax.jit
def _ot_cost(x, y, order:int=2):
    geom = pointcloud.PointCloud(x, y, cost_fn = costs.PNormP(order))
    ot_prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    ot = solver(ot_prob)
    return ot.reg_ot_cost

def _wasserstein_generic(p, q, mc_samples=0, key=None, order=2):
    samples1 = p.sample(key, (mc_samples,))
    samples2 = q.sample(key, (mc_samples,))
    
    cost = _ot_cost(samples1, samples2, order=order)
    return (cost * order)**(1/order)


@register_divergence(NAME, dist.Distribution, dist.Distribution)
def _wasserstein_generic(p, q, mc_samples=0, key=None, order=2):
    if p.event_shape != q.event_shape:
        raise ValueError(
            "Wasserstein distance between distributions with different event shapes not supported"
        )

    assert (
        mc_samples >= 0
    ), "For general distirbutions we require mc_samples >= 0, to evaluate a Monte Carlo approximation of the Wasserstein distance."

    if sum(p.event_shape) > 1:
        return _wasserstein_generic(p, q, mc_samples=mc_samples, key=key, order=order)
    else:
        return _1d_wasserstein(p, q, mc_samples=mc_samples, key=key, order=order)


@register_divergence(NAME, dist.Normal, dist.Normal)
def _wasserstein_normal_normal(p, q, mc_samples=0, key=None, order=2):
    if order == 2:
        t1 = (p.mean - q.mean) ** 2
        t2 = p.variance + q.variance - 2 * jnp.sqrt(p.variance * q.variance)
        return t1 + t2
    else:
        return _wasserstein_generic(p, q, mc_samples=mc_samples, key=key, order=order)
    
@register_divergence(NAME, dist.MultivariateNormal, dist.MultivariateNormal)
def _wasserstein_multivariate_normal_multivariate_normal(p, q, mc_samples=0, key=None, order=2):
    if order == 2:
        t1 = jnp.linalg.norm(p.mean - q.mean) ** 2
        C1 = p.covariance_matrix
        C2 = q.covariance_matrix
        C2_sqrt = jnp.linalg.cholesky(C2)
        t2 = C1 + C2 - 2 * jnp.linalg.cholesky(C2_sqrt @ C1 @ C2_sqrt)
        t2 = jnp.linalg.trace(t2)
        return t1 + t2
    else:
        return _wasserstein_generic(p, q, mc_samples=mc_samples, key=key, order=order)

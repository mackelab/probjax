"""
Wasserstein distance implementations.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from probjax.stats import (
    # Discrete distributions
    multivariate_normal,
    # Continuous distributions
    norm,
    rv_generic,
)
from probjax.stats.divergences.base import divergence, register_divergence

__all__ = [
    "wasserstein_distance",
    "sliced_wasserstein_distance",
    "max_slice_wasserstein_distance",
]

NAME = "wasserstein"
NAME_SLICED = "sliced_wasserstein"
NAME_MAX_SLICE = "max_slice_wasserstein"


def wasserstein_distance(p, q, mc_samples=0, key=None, order=2, **kwargs):
    """Compute the Wasserstein distance between two distributions.

    Args:
        p (rv_generic): First distribution
        q (rv_generic): Second distribution
        mc_samples (int): Number of Monte Carlo samples
        key (jax.Array): JAX random key
        order (int): The order of the Wasserstein distance
        **kwargs: Additional keyword arguments

    Returns:
        jax.Array: The Wasserstein distance between the two distributions
    """
    return divergence(NAME, p, q, mc_samples=mc_samples, key=key, order=order, **kwargs)


def sliced_wasserstein_distance(
    p, q, num_slices=100, mc_samples=0, key=None, order=2, **kwargs
):
    """Compute the Sliced Wasserstein distance between two distributions.

    Args:
        p (rv_generic): First distribution
        q (rv_generic): Second distribution
        num_slices (int): Number of slices
        mc_samples (int): Number of Monte Carlo samples
        key (jax.Array): JAX random key
        order (int): The order of the Wasserstein distance
        **kwargs: Additional keyword arguments

    Returns:
        jax.Array: The Sliced Wasserstein distance between the two distributions
    """
    return divergence(
        NAME_SLICED,
        p,
        q,
        mc_samples=mc_samples,
        num_slices=num_slices,
        key=key,
        order=order,
        **kwargs,
    )


def max_slice_wasserstein_distance(
    p, q, num_slices=100, mc_samples=0, key=None, order=2, **kwargs
):
    """Compute the Max Sliced Wasserstein distance between two distributions.

    Args:
        p (rv_generic): First distribution
        q (rv_generic): Second distribution
        num_slices (int): Number of slices
        mc_samples (int): Number of Monte Carlo samples
        key (jax.Array): JAX random key
        order (int): The order of the Wasserstein distance
        **kwargs: Additional keyword arguments

    Returns:
        jax.Array: The Max Sliced Wasserstein distance between the two distributions
    """
    return divergence(
        NAME_MAX_SLICE,
        p,
        q,
        mc_samples=mc_samples,
        num_slices=num_slices,
        key=key,
        order=order,
        **kwargs,
    )


def _1d_wasserstein(p, q, mc_samples=0, key=None, order=2):
    eval_points = jnp.linspace(0, 1, mc_samples)
    f1 = p.ppf(eval_points)
    f2 = q.ppf(eval_points)
    dist = jnp.abs(f1 - f2) ** order
    return jnp.trapz(dist, eval_points)


def _1d_wasserstein_without_cdf(samples_p, samples_q, order=1):
    sorted_samples_p = jnp.sort(samples_p)
    sorted_samples_q = jnp.sort(samples_q)
    dist = jnp.abs(sorted_samples_p - sorted_samples_q) ** order
    return jnp.mean(dist)


@partial(jax.jit, static_argnums=(2,))
def _ot_cost(x, y, order: int = 2, epsilon: float = 0.1):
    geom = pointcloud.PointCloud(x, y, cost_fn=costs.PNormP(order), epsilon=epsilon)
    ot_prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    ot = solver(ot_prob)
    return ot.reg_ot_cost


def __sliced_wasserstein_generic(
    samples_p, samples_q, num_slices, order=1, key=None, **kwargs
):
    slices = jax.random.normal(key, (num_slices, samples_p.shape[-1]))
    slices = slices / jnp.linalg.norm(slices, axis=-1, keepdims=True)
    sliced_samples_p = jnp.dot(samples_p, slices.T).T
    sliced_samples_q = jnp.dot(samples_q, slices.T).T
    wasserstein_1d = jax.vmap(_1d_wasserstein_without_cdf)(
        sliced_samples_p, sliced_samples_q
    )
    return jnp.mean(wasserstein_1d)


@partial(jax.jit, static_argnums=(2, 3, 4))
def __max_slice_wasserstein_generic(
    samples_p, samples_q, order=1, tol=1e-8, num_init=10, key=None, **kwargs
):
    def loss_fn(theta):
        theta = theta / jnp.linalg.norm(theta)
        samples1_slice = jnp.dot(samples_p, theta)
        samples2_slice = jnp.dot(samples_q, theta)
        return -_1d_wasserstein_without_cdf(samples1_slice, samples2_slice, order=order)

    @jax.vmap
    def optimal_fn(key):
        theta_opt_result = minimize(
            loss_fn,
            jax.random.normal(key, shape=(samples_p.shape[-1],)),
            method="BFGS",
            tol=tol,
        )

        opt_value = -theta_opt_result.fun
        return opt_value

    keys = jax.random.split(key, (num_init,))
    opt_values = optimal_fn(keys)

    return jnp.max(opt_values)


def __wasserstein_generic(p, q, mc_samples=0, key=None, order=2, **kwargs):
    key1, key2 = jax.random.split(key, 2)
    samples1 = p.rvs(key1, (mc_samples,))
    samples2 = q.rvs(key2, (mc_samples,))

    epsilon = kwargs.get("epsilon", 0.1)
    cost = _ot_cost(samples1, samples2, order=order, epsilon=epsilon)
    return (cost * order) ** (1 / order)


@register_divergence(NAME_SLICED, rv_generic, rv_generic)
def _sliced_wasserstein_generic(
    p, q, num_slices=100, mc_samples=0, key=None, order=2, **kwargs
):
    key1, key2 = jax.random.split(key, 2)
    samples1 = p.rvs(key1, (mc_samples,))
    samples2 = q.rvs(key2, (mc_samples,))

    return __sliced_wasserstein_generic(
        samples1, samples2, num_slices, order=order, key=key, **kwargs
    )


@register_divergence(NAME_MAX_SLICE, rv_generic, rv_generic)
def _max_sliced_wasserstein(p, q, mc_samples=0, key=None, order=2, **kwargs):
    key1, key2, key3 = jax.random.split(key, 3)
    samples1 = p.rvs(key1, (mc_samples,))
    samples2 = q.rvs(key2, (mc_samples,))

    return __max_slice_wasserstein_generic(
        samples1, samples2, order=order, key=key3, **kwargs
    )


@register_divergence(NAME, rv_generic, rv_generic)
def _wasserstein_generic(p, q, mc_samples=0, key=None, order=2, **kwargs):
    """Compute the Wasserstein distance between two generic distributions.

    Args:
        p (rv_generic): First distribution
        q (rv_generic): Second distribution
        mc_samples (int): Number of Monte Carlo samples
        key (jax.Array): JAX random key
        order (int): The order of the Wasserstein distance
        **kwargs: Additional keyword arguments

    Returns:
        jax.Array: The Wasserstein distance between the two distributions
    """
    if p.event_shape != q.event_shape:
        raise ValueError(
            "Wasserstein distance between distributions with different event shapes "
            "not supported"
        )

    assert mc_samples >= 0, (
        "For general distributions we require mc_samples >= 0, to evaluate a Monte"
        "Carlo approximation of the Wasserstein distance."
    )

    if sum(p.event_shape) > 1:
        return __wasserstein_generic(
            p, q, mc_samples=mc_samples, key=key, order=order, **kwargs
        )
    else:
        return _1d_wasserstein(
            p, q, mc_samples=mc_samples, key=key, order=order, **kwargs
        )


@register_divergence(NAME, norm, norm)
def _wasserstein_normal_normal(p, q, mc_samples=0, key=None, order=2):
    if order == 2:
        t1 = (p.mean - q.mean) ** 2
        t2 = p.variance + q.variance - 2 * jnp.sqrt(p.variance * q.variance)
        return t1 + t2
    else:
        return __wasserstein_generic(p, q, mc_samples=mc_samples, key=key, order=order)


@register_divergence(NAME, multivariate_normal, multivariate_normal)
def _wasserstein_multivariate_normal_multivariate_normal(
    p, q, mc_samples=0, key=None, order=2
):
    if order == 2:
        t1 = jnp.linalg.norm(p.mean - q.mean) ** 2
        C1 = p.covariance_matrix
        C2 = q.covariance_matrix
        C2_sqrt = jnp.linalg.cholesky(C2)
        t2 = C1 + C2 - 2 * jnp.linalg.cholesky(C2_sqrt @ C1 @ C2_sqrt)
        t2 = jnp.linalg.trace(t2)
        return t1 + t2
    else:
        return __wasserstein_generic(p, q, mc_samples=mc_samples, key=key, order=order)

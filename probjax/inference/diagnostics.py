"""MCMC convergence diagnostics, re-exported from BlackJAX.

Thin wrapper around :mod:`blackjax.diagnostics` so ProbJax runners can report
R-hat, effective sample size, and HMC divergence concentration without vendoring
the numerics. Importing from here (rather than BlackJAX directly) keeps the
public surface ``probjax.inference.*`` and tracks upstream behavior changes —
e.g. since BlackJAX 1.7.0 ``effective_sample_size`` returns 0 for
numerically degenerate (zero-variance) chains and NaN if any draw is NaN.

The kernelized Stein discrepancy (``imq_kernel``,
``kernelized_stein_discrepancy``) is vendored from unreleased BlackJAX ``main``
(``080662f8`` + PyTree-native rework in ``832a1bde``) because the pinned
``blackjax==1.7.1`` does not ship it yet. Remove the vendored copy and
re-export from BlackJAX once the pin reaches a release containing it.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from blackjax.diagnostics import (
    DivergenceConcentrationReport,
    divergence_concentration,
    divergence_concentration_from_counts,
    effective_sample_size,
    ess_bulk,
    ess_tail,
    format_divergence_warning,
    pareto_khat,
    potential_scale_reduction,
    rhat,
)
from blackjax.types import Array, ArrayLikeTree
from jax.flatten_util import ravel_pytree

__all__ = [
    "DivergenceConcentrationReport",
    "divergence_concentration",
    "divergence_concentration_from_counts",
    "effective_sample_size",
    "ess_bulk",
    "ess_tail",
    "format_divergence_warning",
    "imq_kernel",
    "kernelized_stein_discrepancy",
    "pareto_khat",
    "potential_scale_reduction",
    "rhat",
]


def imq_kernel(x: Array, y: Array, c: float = 1.0, beta: float = -0.5) -> Array:
    """Inverse multiquadric kernel ``(c² + ||x-y||²)**beta``.

    Use ``c > 0`` and ``-1 < beta < 0``. Bind non-default parameters with
    ``functools.partial``. Convergence guarantees additionally require target
    conditions; see :cite:p:`gorham2017kernels`.

    Vendored from unreleased BlackJAX main; see the module docstring.
    """
    return (c**2 + jnp.sum((x - y) ** 2)) ** beta


def kernelized_stein_discrepancy(
    samples: ArrayLikeTree,
    grad_logdensity_fn: Callable,
    kernel: Callable,
    *,
    statistic: str = "v",
) -> Array:
    """Estimate squared kernelized Stein discrepancy with a V- or U-statistic.

    Parameters
    ----------
    samples
        PyTree (an array, or a dict/tuple/NamedTuple of per-field arrays) whose
        leaves share a leading ``num_samples`` axis. Chain dimensions must be
        pooled explicitly before calling this function.
    grad_logdensity_fn
        Target score function, mapping one sample (one leading-axis slice of
        ``samples``, with the same PyTree structure) to the gradient of its log
        density. For example, ``jax.grad(logdensity_fn)``. The normalizing
        constant is not required.
    kernel
        Scalar-valued, twice differentiable positive semidefinite kernel taking
        two flattened sample vectors. Bind kernel parameters such as bandwidth
        with ``functools.partial``. The kernel must satisfy the target's Stein
        boundary conditions; choosing an appropriate kernel is the caller's
        responsibility.

    statistic
        ``"v"`` includes diagonal pairs; ``"u"`` excludes them and requires at
        least two draws. The U-statistic can be negative, even in exact arithmetic.

    Returns
    -------
    Estimate of squared KSD, not a calibrated p-value or convergence certificate.
    The V-statistic has a sample-dependent diagonal bias of order ``1/n``.

    Notes
    -----
    Complements rather than replaces R-hat. Approximately independent draws are
    assumed; autocorrelation and an unvisited mode can make the result misleading.
    A positive semidefinite kernel alone does not identify every distribution:
    this requires additional conditions, including integral strict positive
    definiteness :cite:p:`liu2016kernelized`. Gaussian RBF kernels do not in
    general detect non-convergence in dimension three or higher. IMQ kernels
    have convergence guarantees for suitable targets, but targets with bounded
    scores, including some heavy-tailed targets, remain a gap
    :cite:p:`gorham2017kernels`.

    Uses the score/kernel derivative expression in :cite:p:`liu2016kernelized`.
    Rows are processed sequentially and columns are vectorized. The computation
    has quadratic cost in sample count and mixed Hessians with ``d²`` entries;
    temporary pairwise storage scales as ``O(n*d²)`` rather than ``O(n²*d²)``.
    The supplied functions must support JAX differentiation and transformations.
    No clipping is applied to either statistic.

    Vendored from unreleased BlackJAX main; see the module docstring.
    """
    if statistic not in ("v", "u"):
        raise ValueError("statistic must be v or u")
    flat = jax.vmap(lambda s: ravel_pytree(s)[0])(samples)
    scores = jax.vmap(lambda s: ravel_pytree(grad_logdensity_fn(s))[0])(samples)
    kernel_value_and_grad = jax.value_and_grad(kernel, argnums=(0, 1))
    mixed_derivative = jax.jacfwd(jax.grad(kernel, argnums=0), argnums=1)

    def stein_kernel(x, y, score_x, score_y):
        value, (grad_x, grad_y) = kernel_value_and_grad(x, y)
        return (
            value * jnp.dot(score_x, score_y)
            + jnp.dot(score_x, grad_y)
            + jnp.dot(grad_x, score_y)
            + jnp.trace(mixed_derivative(x, y))
        )

    def row_sum(args):
        x, score_x = args
        return jax.vmap(stein_kernel, in_axes=(None, 0, None, 0))(
            x, flat, score_x, scores
        ).sum()

    n = flat.shape[0]
    total = jax.lax.map(row_sum, (flat, scores)).sum()
    if statistic == "v":
        return total / n**2
    diagonal = jax.vmap(stein_kernel)(flat, flat, scores, scores).sum()
    return (total - diagonal) / (n * (n - 1))

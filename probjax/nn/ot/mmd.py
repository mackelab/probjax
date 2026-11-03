"""Kernel two-sample statistics and KDE on top of streaming log-sum-exp.

Every kernel sum ``sum_j k(x_i, y_j)`` with ``k = exp(-c / eps)`` is computed as
``exp(kernel_logsumexp(...))``, i.e. block-wise with O((N + M) * D) memory and a
custom VJP that recomputes blocks (Pallas on GPU, blocked XLA on CPU).
Working in the log domain keeps the Gaussian KDE log-density finite where the
density itself underflows (high dimension / small bandwidth).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal, Sequence

import jax.numpy as jnp
from jax import lax

from probjax.nn.pallas_kernels.kernels.kernel_lse import (
    Impl,
    kernel_logsumexp,
    sqeuclidean_cost,
)
from probjax.utils.typing import Array

__all__ = [
    "kernel_sum_log",
    "mmd",
    "mmd_naive",
    "gaussian_kde_logpdf",
    "gaussian_kde_logpdf_naive",
]

KernelName = Literal["rbf", "imq"]


@dataclasses.dataclass(frozen=True)
class _IMQLogCost:
    """``-log k_IMQ = beta * log(c^2 + |x - y|^2)`` as a block cost."""

    c2: float
    beta: float

    def __call__(self, xb, yb):
        r2 = jnp.maximum(sqeuclidean_cost(xb, yb, lax.Precision.HIGHEST), 0.0)
        return self.beta * jnp.log(self.c2 + r2)


def kernel_sum_log(
    x: Array,
    y: Array,
    *,
    kernel: KernelName = "rbf",
    bandwidth: float | Array = 1.0,
    imq_beta: float = 0.5,
    impl: Impl | None = None,
) -> Array:
    """``log sum_j k(x_i, y_j)`` per row, without the N x M kernel matrix.

    ``rbf``: ``k = exp(-|x - y|^2 / (2 h^2))``; ``imq``: ``k = (h^2 + |x-y|^2)^-beta``
    (``h = bandwidth``; IMQ uses a static Python-float bandwidth).
    """
    if kernel == "rbf":
        h = jnp.asarray(bandwidth, jnp.float32)
        return kernel_logsumexp(x, y, epsilon=2.0 * h * h, impl=impl)
    if kernel == "imq":
        cost = _IMQLogCost(float(bandwidth) ** 2, float(imq_beta))
        return kernel_logsumexp(x, y, cost=cost, epsilon=1.0, impl=impl)
    raise ValueError(f"Unknown kernel {kernel!r}")


def _kernel_at_zero(kernel: str, bandwidth, imq_beta: float):
    if kernel == "rbf":
        return 1.0
    return float(bandwidth) ** (-2.0 * imq_beta)


def mmd(
    x: Array,
    y: Array,
    *,
    kernel: KernelName = "rbf",
    bandwidth: float | Array | Sequence[float] = 1.0,
    unbiased: bool = True,
    imq_beta: float = 0.5,
    impl: Impl | None = None,
) -> Array:
    """Squared maximum mean discrepancy ``MMD^2(x, y)``.

    Args:
        x: Samples ``[N, D]``; y: samples ``[M, D]``.
        kernel: ``"rbf"`` or ``"imq"``.
        bandwidth: Scalar or a sequence of bandwidths; a sequence gives the
            multi-bandwidth (sum of kernels) statistic.
        unbiased: U-statistic (drops the ``i == j`` terms) or the biased
            V-statistic.
        imq_beta: Exponent of the IMQ kernel.
        impl: :func:`kernel_logsumexp` backend (``None`` = auto).

    Returns:
        Scalar ``MMD^2`` (the unbiased estimate can be slightly negative).
    """
    bws = list(bandwidth) if isinstance(bandwidth, (list, tuple)) else [bandwidth]
    n, m = x.shape[0], y.shape[0]
    total = jnp.zeros((), jnp.float32)
    for h in bws:
        sxx = jnp.sum(
            jnp.exp(
                kernel_sum_log(
                    x, x, kernel=kernel, bandwidth=h, imq_beta=imq_beta, impl=impl
                )
            )
        )
        syy = jnp.sum(
            jnp.exp(
                kernel_sum_log(
                    y, y, kernel=kernel, bandwidth=h, imq_beta=imq_beta, impl=impl
                )
            )
        )
        sxy = jnp.sum(
            jnp.exp(
                kernel_sum_log(
                    x, y, kernel=kernel, bandwidth=h, imq_beta=imq_beta, impl=impl
                )
            )
        )
        if unbiased:
            k0 = _kernel_at_zero(kernel, h, imq_beta)
            total = (
                total + (sxx - n * k0) / (n * (n - 1)) + (syy - m * k0) / (m * (m - 1))
            )
        else:
            total = total + sxx / (n * n) + syy / (m * m)
        total = total - 2.0 * sxy / (n * m)
    return total


def mmd_naive(
    x: Array,
    y: Array,
    *,
    kernel: KernelName = "rbf",
    bandwidth: float | Sequence[float] = 1.0,
    unbiased: bool = True,
    imq_beta: float = 0.5,
) -> Array:
    """Dense reference for :func:`mmd` (materializes all kernel matrices)."""
    bws = list(bandwidth) if isinstance(bandwidth, (list, tuple)) else [bandwidth]

    def k(a, b, h):
        r2 = jnp.maximum(sqeuclidean_cost(a, b, lax.Precision.HIGHEST), 0.0)
        if kernel == "rbf":
            return jnp.exp(-r2 / (2.0 * h * h))
        return (h * h + r2) ** (-imq_beta)

    n, m = x.shape[0], y.shape[0]
    total = 0.0
    for h in bws:
        kxx, kyy, kxy = k(x, x, h), k(y, y, h), k(x, y, h)
        if unbiased:
            kxx = kxx * (1 - jnp.eye(n))
            kyy = kyy * (1 - jnp.eye(m))
            total = total + kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1))
        else:
            total = total + kxx.sum() / (n * n) + kyy.sum() / (m * m)
        total = total - 2.0 * kxy.sum() / (n * m)
    return total


def _bandwidth(data: Array, weights: Array | None, bandwidth) -> Array:
    n, d = data.shape
    if not isinstance(bandwidth, str):
        return jnp.asarray(bandwidth, jnp.float32)
    if weights is None:
        n_eff = n
        std = jnp.std(data.astype(jnp.float32), axis=0, ddof=1)
    else:
        w = weights / jnp.sum(weights)
        n_eff = 1.0 / jnp.sum(w * w)
        mu = w @ data
        std = jnp.sqrt(w @ (data - mu) ** 2)
    sigma = jnp.mean(std)
    if bandwidth == "scott":
        factor = n_eff ** (-1.0 / (d + 4))
    elif bandwidth == "silverman":
        factor = (n_eff * (d + 2) / 4.0) ** (-1.0 / (d + 4))
    else:
        raise ValueError(f"Unknown bandwidth rule {bandwidth!r}")
    return sigma * factor


def gaussian_kde_logpdf(
    x: Array,
    data: Array,
    *,
    bandwidth: float | Array | Literal["scott", "silverman"] = "scott",
    weights: Array | None = None,
    impl: Impl | None = None,
) -> Array:
    """Log-density of an isotropic Gaussian KDE at ``x`` [N, D].

    ``log p(x_i) = logsumexp_j(log w_j - |x_i - data_j|^2 / (2 h^2))
    - D/2 log(2 pi h^2)`` with normalized weights ``w`` (uniform if ``None``).
    Finite even where ``p`` underflows in fp32.
    """
    data = jnp.asarray(data)
    n, d = data.shape
    h = _bandwidth(data, weights, bandwidth)
    if weights is None:
        lw = jnp.full((n,), -math.log(n), jnp.float32)
    else:
        w = jnp.asarray(weights, jnp.float32)
        lw = jnp.log(w / jnp.sum(w))
    lse = kernel_logsumexp(x, data, log_weights=lw, epsilon=2.0 * h * h, impl=impl)
    return lse - 0.5 * d * jnp.log(2.0 * jnp.pi * h * h)


def gaussian_kde_logpdf_naive(x, data, *, bandwidth=1.0, weights=None):
    """Dense reference for :func:`gaussian_kde_logpdf` (numeric bandwidth)."""
    import jax

    n, d = data.shape
    h = jnp.asarray(bandwidth, jnp.float32)
    lw = (
        jnp.full((n,), -math.log(n))
        if weights is None
        else jnp.log(weights / jnp.sum(weights))
    )
    r2 = sqeuclidean_cost(x, data, lax.Precision.HIGHEST)
    lse = jax.nn.logsumexp(lw[None, :] - r2 / (2 * h * h), axis=1)
    return lse - 0.5 * d * jnp.log(2.0 * jnp.pi * h * h)

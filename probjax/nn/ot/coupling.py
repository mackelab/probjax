"""Minibatch optimal-transport couplings for flow matching (OT-CFM).

Tong et al. (2023, "Improving and generalizing flow-based generative models
with minibatch optimal transport") and Pooladian et al. (2023, "Multisample
flow matching") re-pair the independent noise/data minibatch by an OT plan
before regressing the conditional velocity. This straightens the learned
probability paths and reduces the number of ODE steps needed at sampling.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from probjax.nn.ot.sinkhorn import sample_conditional, sinkhorn
from probjax.nn.pallas_kernels.kernels.kernel_lse import sqeuclidean_cost
from probjax.utils.typing import Array

__all__ = ["ot_pairing", "apply_ot_pairing", "exact_assignment"]

Method = Literal["exact", "sinkhorn"]


def _hungarian_host(cost: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(cost, dtype=np.float64)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(cost.shape[0], dtype=np.int32)
    perm[rows] = cols
    return perm


def exact_assignment(cost: Array) -> Array:
    """Exact min-cost perfect matching of a square cost matrix (Hungarian).

    Returns ``perm`` with row ``i`` assigned to column ``perm[i]``. Runs SciPy's
    ``linear_sum_assignment`` on the host via ``jax.pure_callback`` (jit- and
    vmap-compatible; O(B^3), intended for minibatches of up to a few thousand).
    """
    n = cost.shape[0]
    if cost.shape != (n, n):
        raise ValueError(f"exact_assignment needs a square cost, got {cost.shape}")
    return jax.pure_callback(
        _hungarian_host,
        jax.ShapeDtypeStruct((n,), jnp.int32),
        cost,
        vmap_method="sequential",
    )


def ot_pairing(
    x0: Array,
    x1: Array,
    rng: Array | None = None,
    *,
    method: Method = "exact",
    epsilon: float | None = None,
    num_iters: int = 100,
    impl=None,
) -> Array:
    """Indices ``idx`` such that ``(x0[idx[k]], x1[k])`` are OT-coupled pairs.

    The data side ``x1`` keeps its order (so labels/conditioning stay aligned);
    only the source batch is re-indexed.

    Args:
        x0: Source batch ``[B, ...]`` (e.g. noise).
        x1: Target batch ``[B, ...]`` (e.g. data). Trailing dims are flattened.
        rng: Required for ``method="sinkhorn"`` (pairs are sampled).
        method: ``"exact"`` - Hungarian assignment of the squared-Euclidean
            cost, a permutation (Tong et al. use the exact plan too).
            ``"sinkhorn"`` - for each target ``k`` sample ``i ~ P(i | k)`` from
            the entropic plan (with replacement; scales to large batches and
            runs on device without a host round trip).
        epsilon: Entropic regularization for ``"sinkhorn"`` (``None`` =
            ``0.05 * mean cost``).
        num_iters: Sinkhorn iterations.
        impl: :func:`kernel_logsumexp` backend.

    Returns:
        ``idx`` ``[B]`` (int32).
    """
    b = x0.shape[0]
    if x1.shape[0] != b:
        raise ValueError(f"Batch sizes differ: {x0.shape[0]} vs {x1.shape[0]}")
    a = lax.stop_gradient(x0.reshape(b, -1))
    c = lax.stop_gradient(x1.reshape(b, -1))
    if method == "exact":
        cost = sqeuclidean_cost(c, a, lax.Precision.HIGHEST)
        return exact_assignment(cost)
    if method == "sinkhorn":
        if rng is None:
            raise ValueError("method='sinkhorn' needs an rng to sample pairs.")
        out = sinkhorn(c, a, epsilon=epsilon, num_iters=num_iters, impl=impl)
        return sample_conditional(out, rng)
    raise ValueError(f"Unknown OT pairing method {method!r}")


def apply_ot_pairing(x0: Array, x1: Array, rng: Array | None = None, **kwargs) -> Array:
    """Return ``x0`` re-indexed so that ``(x0_new[k], x1[k])`` are OT pairs."""
    idx = ot_pairing(x0, x1, rng, **kwargs)
    return jnp.take(x0, idx, axis=0)

"""Memory-scalable log-domain entropic optimal transport (Sinkhorn).

All N x M interactions go through :func:`~probjax.nn.pallas_kernels.kernels.
kernel_lse.kernel_logsumexp`, so the solver, the cost and its gradient, the
entropic (barycentric) map and plan sampling never materialize the cost or
transport matrix: memory is O((N + M) * D), time O(N * M * D) per iteration.

Conventions follow ott-jax (``reg_ot_cost``, potentials ``f``/``g``)::

    OT_eps(a, b) = min_P <P, C> + eps * KL(P | a b^T),
    P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax

from probjax.nn.pallas_kernels.kernels.kernel_lse import (
    Cost,
    Impl,
    kernel_logsumexp,
    sqeuclidean_cost,
)
from probjax.utils.typing import Array

__all__ = [
    "SinkhornOutput",
    "sinkhorn",
    "sinkhorn_divergence",
    "entropic_map",
    "sample_conditional",
    "transport_plan_dense",
    "mean_sqeuclidean_cost",
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SinkhornOutput:
    """Result of :func:`sinkhorn` (a pytree; ``cost``/``impl`` are static)."""

    f: Array
    g: Array
    reg_ot_cost: Array
    error: Array
    num_iters: Array
    converged: Array
    epsilon: Array
    x: Array
    y: Array
    log_a: Array
    log_b: Array
    cost: Cost = dataclasses.field(default="sqeuclidean", metadata=dict(static=True))
    impl: Impl | None = dataclasses.field(default=None, metadata=dict(static=True))


def _transpose_cost(cost: Cost) -> Cost:
    if cost == "sqeuclidean":
        return cost
    return _TransposedCost(cost)


@dataclasses.dataclass(frozen=True)
class _TransposedCost:
    """``c^T(y, x) = c(x, y)^T`` (hashable, so it can be a static argument)."""

    cost: object

    def __call__(self, yb, xb):
        return self.cost(xb, yb).T


def mean_sqeuclidean_cost(x: Array, y: Array, a: Array, b: Array) -> Array:
    """Exact ``sum_ij a_i b_j |x_i - y_j|^2`` in O((N + M) * D)."""
    xf = x.astype(jnp.float32)
    yf = y.astype(jnp.float32)
    mx = a @ xf
    my = b @ yf
    return a @ jnp.sum(xf * xf, -1) + b @ jnp.sum(yf * yf, -1) - 2.0 * mx @ my


def _log_weights(w: Array | None, n: int) -> Array:
    if w is None:
        return jnp.full((n,), -jnp.log(n), jnp.float32)
    w = jnp.asarray(w, jnp.float32)
    return jnp.log(w / jnp.sum(w))


def _f_update(x, y, log_b, g, eps, cost, impl):
    lse = kernel_logsumexp(
        x, y, log_weights=log_b + g / eps, cost=cost, epsilon=eps, impl=impl
    )
    return -eps * lse


def _g_update(x, y, log_a, f, eps, cost, impl):
    lse = kernel_logsumexp(
        y,
        x,
        log_weights=log_a + f / eps,
        cost=_transpose_cost(cost),
        epsilon=eps,
        impl=impl,
    )
    return -eps * lse


def _dual_objective(x, y, log_a, log_b, f, g, eps, cost, impl):
    """Entropic dual at (f, g); its gradient in (x, y, a, b) at the optimum
    equals that of ``OT_eps`` (envelope theorem), with O((N + M) D) memory."""
    lse = kernel_logsumexp(
        x, y, log_weights=log_b + g / eps, cost=cost, epsilon=eps, impl=impl
    )
    a = jnp.exp(log_a)
    b = jnp.exp(log_b)
    mass = jnp.sum(jnp.exp(log_a + f / eps + lse))
    return a @ f + b @ g - eps * mass + eps * jnp.sum(a) * jnp.sum(b)


def sinkhorn(
    x: Array,
    y: Array,
    a: Array | None = None,
    b: Array | None = None,
    *,
    epsilon: float | Array | None = None,
    cost: Cost = "sqeuclidean",
    num_iters: int = 100,
    tol: float | None = None,
    check_every: int = 10,
    f_init: Array | None = None,
    g_init: Array | None = None,
    differentiate: Literal["implicit", "unroll"] = "implicit",
    impl: Impl | None = None,
) -> SinkhornOutput:
    """Balanced entropic OT between weighted point clouds, in the log domain.

    Args:
        x: Source points ``[N, D]``.
        y: Target points ``[M, D]``.
        a, b: Optional nonnegative weights (normalized internally); uniform
            if ``None``.
        epsilon: Entropic regularization. ``None`` uses ``0.05 * mean cost``
            (sqeuclidean only; computed exactly in O((N + M) D)).
        cost: ``"sqeuclidean"`` or a block callable ``c(x[bq, D], y[bk, D])``.
        num_iters: Maximum number of (g, f) update pairs.
        tol: If given, stop once the L1 error of the source marginal is below
            ``tol`` (checked every ``check_every`` iterations). Requires
            ``differentiate="implicit"``.
        f_init, g_init: Warm-start potentials (only ``f_init`` is used by the
            first ``g`` update; ``g_init`` is accepted for symmetry).
        differentiate: ``"implicit"`` (default) runs the solver under
            ``stop_gradient`` and differentiates ``reg_ot_cost`` through the
            dual objective at the optimum (envelope theorem; exact at
            convergence, O((N + M) D) memory). ``"unroll"`` backpropagates
            through every iteration (memory grows with ``num_iters``).
        impl: Backend for :func:`kernel_logsumexp` (``None`` = auto).

    Returns:
        :class:`SinkhornOutput` with potentials ``f`` [N], ``g`` [M] and the
        entropic cost ``reg_ot_cost = <a, f> + <b, g>`` (ott-jax convention).
    """
    del g_init
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n, m = x.shape[0], y.shape[0]
    log_a = _log_weights(a, n)
    log_b = _log_weights(b, m)
    if epsilon is None:
        if cost != "sqeuclidean":
            raise ValueError("epsilon=None (relative) requires cost='sqeuclidean'.")
        epsilon = 0.05 * lax.stop_gradient(
            mean_sqeuclidean_cost(x, y, jnp.exp(log_a), jnp.exp(log_b))
        )
    eps = jnp.asarray(epsilon, jnp.float32)
    if differentiate not in ("implicit", "unroll"):
        raise ValueError(f"Unknown differentiate={differentiate!r}")
    if tol is not None and differentiate == "unroll":
        raise ValueError("tol (early stopping) is not reverse-differentiable.")

    if differentiate == "implicit":
        xs, ys, las, lbs, es = map(lax.stop_gradient, (x, y, log_a, log_b, eps))
    else:
        xs, ys, las, lbs, es = x, y, log_a, log_b, eps
    f0 = jnp.zeros((n,), jnp.float32) if f_init is None else jnp.asarray(f_init)

    def step(f):
        g = _g_update(xs, ys, las, f, es, cost, impl)
        f_new = _f_update(xs, ys, lbs, g, es, cost, impl)
        # Source marginal of the plan at (f, g): a_i exp((f_i - f_new_i) / eps).
        err = jnp.sum(jnp.abs(jnp.exp(las) * jnp.expm1((f - f_new) / es)))
        return f_new, g, err

    if tol is None:

        def body(f, _):
            f, g, err = step(f)
            return f, (g, err)

        f, (gs, errs) = lax.scan(body, f0, None, length=num_iters)
        g, err = gs[-1], errs[-1]
        iters = jnp.asarray(num_iters)
    else:
        check_every = max(1, min(int(check_every), int(num_iters)))

        def chunk(state):
            f, _, _, it = state

            def inner(_, carry):
                f, g, err = carry
                return step(f)

            g0 = jnp.zeros((m,), jnp.float32)
            f, g, err = lax.fori_loop(
                0, check_every, inner, (f, g0, jnp.asarray(jnp.inf, jnp.float32))
            )
            return f, g, err, it + check_every

        def cond(state):
            _, _, err, it = state
            return (it < num_iters) & (err > tol)

        init = (
            f0,
            jnp.zeros((m,), jnp.float32),
            jnp.asarray(jnp.inf, jnp.float32),
            jnp.asarray(0),
        )
        f, g, err, iters = lax.while_loop(cond, chunk, init)

    if differentiate == "implicit":
        f = lax.stop_gradient(f)
        g = lax.stop_gradient(g)
        cost_value = _dual_objective(x, y, log_a, log_b, f, g, eps, cost, impl)
    else:
        cost_value = jnp.exp(log_a) @ f + jnp.exp(log_b) @ g
    converged = err <= (tol if tol is not None else jnp.inf)
    return SinkhornOutput(
        f=f,
        g=g,
        reg_ot_cost=cost_value,
        error=err,
        num_iters=iters,
        converged=converged,
        epsilon=eps,
        x=x,
        y=y,
        log_a=log_a,
        log_b=log_b,
        cost=cost,
        impl=impl,
    )


def sinkhorn_divergence(
    x: Array,
    y: Array,
    a: Array | None = None,
    b: Array | None = None,
    *,
    epsilon: float | Array | None = None,
    **kwargs,
) -> Array:
    """Debiased Sinkhorn divergence ``OT(a,b) - (OT(a,a) + OT(b,b)) / 2``.

    ``epsilon=None`` resolves to ``0.05 * mean cost`` of the cross problem and
    the same value is used for all three terms.
    """
    if epsilon is None:
        n, m = x.shape[0], y.shape[0]
        wa = jnp.exp(_log_weights(a, n))
        wb = jnp.exp(_log_weights(b, m))
        epsilon = 0.05 * lax.stop_gradient(mean_sqeuclidean_cost(x, y, wa, wb))
    xy = sinkhorn(x, y, a, b, epsilon=epsilon, **kwargs).reg_ot_cost
    xx = sinkhorn(x, x, a, a, epsilon=epsilon, **kwargs).reg_ot_cost
    yy = sinkhorn(y, y, b, b, epsilon=epsilon, **kwargs).reg_ot_cost
    return xy - 0.5 * (xx + yy)


def entropic_map(out: SinkhornOutput, x_new: Array | None = None) -> Array:
    """Entropic (barycentric) map ``T(x) = E_P[y | x]`` without the N x M plan.

    ``T(x) = sum_j softmax_j(log b_j + (g_j - c(x, y_j)) / eps) y_j``; for
    ``x_new=None`` it is evaluated on the source points (the barycentric
    projection of the plan), otherwise out of sample (Pooladian & Niles-Weed).
    """
    x = out.x if x_new is None else jnp.asarray(x_new)
    _, t = kernel_logsumexp(
        x,
        out.y,
        values=out.y,
        log_weights=out.log_b + out.g / out.epsilon,
        cost=out.cost,
        epsilon=out.epsilon,
        impl=out.impl,
    )
    return t


def transport_plan_dense(out: SinkhornOutput) -> Array:
    """Materialized plan ``P`` [N, M] (testing / small problems only)."""
    from probjax.nn.pallas_kernels.kernels.kernel_lse import _cost_block

    c = _cost_block(out.cost, out.x, out.y, lax.Precision.HIGHEST)
    return jnp.exp(
        out.log_a[:, None]
        + out.log_b[None, :]
        + (out.f[:, None] + out.g[None, :] - c) / out.epsilon
    )


def sample_conditional(
    out: SinkhornOutput,
    rng: Array,
    *,
    block_q: int = 512,
    block_k: int = 2048,
) -> Array:
    """Draw ``j_i ~ P(j | i)`` for every source point (one target index per row).

    Uses the Gumbel-max trick over key blocks with a running argmax, so memory
    is O(block_q * block_k) instead of O(N * M). Since ``P(i)`` is the source
    marginal, pairing ``(x_i, y_{j_i})`` samples the plan with the exact source
    marginal (the usual minibatch-OT pairing).
    """
    x, y = out.x, out.y
    n, m = x.shape[0], y.shape[0]
    bq = min(block_q, n)
    bk = min(block_k, m)
    n_pad = -(-n // bq) * bq
    m_pad = -(-m // bk) * bk
    xp = jnp.pad(x, [(0, n_pad - n), (0, 0)])
    yp = jnp.pad(y, [(0, m_pad - m), (0, 0)])
    lw = jnp.pad(
        out.log_b + out.g / out.epsilon, (0, m_pad - m), constant_values=-jnp.inf
    )
    inv_eps = 1.0 / out.epsilon
    cost = out.cost

    def cost_block(xb, yb):
        if cost == "sqeuclidean":
            return sqeuclidean_cost(xb, yb, lax.Precision.HIGHEST)
        return cost(xb, yb).astype(jnp.float32)

    def q_block(args):
        qb, xb = args
        key_q = jax.random.fold_in(rng, qb)

        def body(kb, carry):
            best, arg = carry
            start = kb * bk
            yb = lax.dynamic_slice_in_dim(yp, start, bk)
            lwb = lax.dynamic_slice_in_dim(lw, start, bk)
            s = lwb[None, :] - cost_block(xb, yb) * inv_eps
            s = s + jax.random.gumbel(jax.random.fold_in(key_q, kb), s.shape)
            bmax = jnp.max(s, axis=1)
            barg = (jnp.argmax(s, axis=1) + start).astype(jnp.int32)
            take = bmax > best
            return jnp.where(take, bmax, best), jnp.where(take, barg, arg)

        init = (jnp.full((bq,), -jnp.inf), jnp.zeros((bq,), jnp.int32))
        _, arg = lax.fori_loop(0, m_pad // bk, body, init)
        return arg

    idx = lax.map(q_block, (jnp.arange(n_pad // bq), xp.reshape(-1, bq, x.shape[1])))
    return idx.reshape(n_pad)[:n]

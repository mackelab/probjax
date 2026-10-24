"""Gather / scatter dropless dispatch (MoELayer(dropless_dispatch=...)).

"gather" and "fused" must compute exactly what the sorted top-k-copy path
computes -- outputs and every gradient -- for any routing.  On CPU the Pallas
kernels run in interpret mode; tests/test_moe_dispatch_gpu.py repeats the
comparison at production size on a GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.nn.layers.grouped_matmul import moe_swiglu_dispatch
from probjax.nn.layers.moe import MoELayer

# On GPU the Triton dots run fp32 inputs as TF32 (~1e-3 relative), so paths
# with different reduction orders agree only to that; on CPU (interpret mode)
# they agree to fp32 rounding.
TOL = 2e-2 if jax.default_backend() == "gpu" else 1e-4
# fused vs gather: same kernels, but the atomic adds sum fp32 contributions in
# another order (measured 4.5e-5 on an H100).
TIGHT = 2e-4 if jax.default_backend() == "gpu" else 1e-5


def _close(a, b, tol=None):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
    assert err < (TOL if tol is None else tol), err


def _layer(dispatch, d=32, h=16, e=8, k=2, shared=None, seed=0):
    return MoELayer(d, h, e, top_k=k, routing="dropless", dropless_impl="triton",
                    dropless_dispatch=dispatch, shared_hidden=shared, rngs=nnx.Rngs(seed))


def _fwd_grads(m, x):
    g, p = nnx.split(m, nnx.Param)

    def loss(p, x):
        y, aux = nnx.merge(g, p)(x, return_aux=True)
        return jnp.sum(jnp.sin(y)) + aux.load_balance_loss, y

    (_, y), grads = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(p, x)
    return y, grads


@pytest.mark.parametrize("dispatch", ["gather", "fused", "fused_gsum"])
@pytest.mark.parametrize("cfg", [dict(e=8, k=2), dict(e=16, k=4, h=8, shared=24), dict(e=4, k=4)])
def test_dispatch_matches_sorted(dispatch, cfg):
    x = jax.random.normal(jax.random.key(1), (37, 32))  # 37 tokens: odd, needs padding
    y_ref, g_ref = _fwd_grads(_layer("sorted", **cfg), x)
    y, g = _fwd_grads(_layer(dispatch, **cfg), x)
    _close(y, y_ref)
    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(g_ref)):
        _close(a, b)


@pytest.mark.parametrize("cfg", [dict(e=8, k=2), dict(e=16, k=4, h=8, shared=24)])
def test_fused_matches_gather_tightly(cfg):
    """Same math, same kernels for the matmuls: only the gather / scatter-add
    moves into the kernel, so the two agree far below TF32 noise."""
    x = jax.random.normal(jax.random.key(2), (45, 32))
    y_g, g_g = _fwd_grads(_layer("gather", **cfg), x)
    y_f, g_f = _fwd_grads(_layer("fused", **cfg), x)
    _close(y_f, y_g, TIGHT)
    for a, b in zip(jax.tree.leaves(g_f), jax.tree.leaves(g_g)):
        _close(a, b, TIGHT)


@pytest.mark.parametrize("impl", ["gather", "fused", "fused_gsum"])
def test_dispatch_function_against_dense_reference(impl):
    """Direct check against the definition, with empty experts and one expert
    taking most rows."""
    n, d, h, e, k = 50, 16, 16, 6, 3
    ks = jax.random.split(jax.random.key(0), 6)
    x = jax.random.normal(ks[0], (n, d))
    wg, wu, wd = (jax.random.normal(kk, s) * 0.3 for kk, s in
                  zip(ks[1:4], [(e, d, h), (e, d, h), (e, h, d)]))
    # experts 1 and 4 never chosen, expert 0 chosen by every token
    ids = jnp.stack([jnp.zeros(n, jnp.int32),
                     jax.random.choice(ks[4], jnp.array([2, 3]), (n,)),
                     jax.random.choice(ks[5], jnp.array([5, 2]), (n,))], axis=1)
    ids = jnp.where(ids[:, 2:3] == ids[:, 1:2], ids.at[:, 2].set(5), ids)
    gates = jax.nn.softmax(jax.random.normal(ks[5], (n, k)), -1)
    counts = jnp.bincount(ids.reshape(-1), length=e).astype(jnp.int32)

    def dense(x, gates, wg, wu, wd):
        def one(xn, idn, gn):
            hid = jax.nn.silu(xn @ wg[idn]) * (xn @ wu[idn])  # [k, h]
            return jnp.einsum("k,kh,khd->d", gn, hid, wd[idn])
        return jax.vmap(one)(x, ids, gates)

    def f(x, gates, wg, wu, wd):
        return moe_swiglu_dispatch(x, gates, wg, wu, wd, ids, counts, 16, impl, None)

    _close(f(x, gates, wg, wu, wd), dense(x, gates, wg, wu, wd))
    loss = lambda fn: (lambda *a: jnp.sum(jnp.cos(fn(*a))))
    ga = jax.grad(loss(f), argnums=(0, 1, 2, 3, 4))(x, gates, wg, wu, wd)
    gb = jax.grad(loss(dense), argnums=(0, 1, 2, 3, 4))(x, gates, wg, wu, wd)
    for a, b in zip(ga, gb):
        _close(a, b)


def test_invalid_dispatch_rejected():
    with pytest.raises(ValueError, match="dropless_dispatch"):
        _layer("scatter")


@pytest.mark.parametrize("impl", ["mosaic", "ragged_dot", "blocked"])
def test_dispatch_with_incompatible_backend_is_rejected(impl):
    """No silent fallback to the sorted path."""
    with pytest.raises(ValueError, match="Triton kernels"):
        MoELayer(32, 16, 8, top_k=2, routing="dropless", dropless_impl=impl,
                 dropless_dispatch="fused", rngs=nnx.Rngs(0))


@pytest.mark.parametrize("impl", ["auto", "pallas"])
def test_auto_and_pallas_use_the_dispatch_path(impl, monkeypatch):
    """On Hopper "auto"/"pallas" resolve to Mosaic for the sorted path; with a
    dispatch mode set they must still run moe_swiglu_dispatch."""
    import probjax.nn.layers.grouped_matmul as gm
    calls = []
    real = gm.moe_swiglu_dispatch
    monkeypatch.setattr(gm, "moe_swiglu_dispatch", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(gm, "grouped_backend", lambda: "mosaic")
    m = MoELayer(32, 16, 8, top_k=2, routing="dropless", dropless_impl=impl,
                 dropless_dispatch="fused", rngs=nnx.Rngs(0))
    m(jax.random.normal(jax.random.key(0), (20, 32)))
    assert calls

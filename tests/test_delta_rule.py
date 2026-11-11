"""Tests for the gated delta rule kernels and the Gated DeltaNet layer."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn import GatedDeltaNetCell
from probjax.nn.nets.ssm import SSMModel
from probjax.nn.pallas_kernels.kernels import delta_rule as dr

HIGHEST = jax.lax.Precision.HIGHEST


def _inputs(
    b=2, h=2, length=100, dk=16, dv=32, *, gated=True, dtype=jnp.float32, seed=0
):
    ks = jax.random.split(jax.random.key(seed), 6)
    q = dr.l2_normalize(jax.random.normal(ks[0], (b, h, length, dk))).astype(dtype)
    k = dr.l2_normalize(jax.random.normal(ks[1], (b, h, length, dk))).astype(dtype)
    v = jax.random.normal(ks[2], (b, h, length, dv)).astype(dtype)
    log_alpha = (
        jax.nn.log_sigmoid(jax.random.normal(ks[3], (b, h, length)) + 3.0)
        if gated
        else None
    )
    beta = jax.nn.sigmoid(jax.random.normal(ks[4], (b, h, length)))
    h0 = 0.3 * jax.random.normal(ks[5], (b, h, dk, dv))
    return q, k, v, log_alpha, beta, h0


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _loss(fn):
    def loss(q, k, v, log_alpha, beta, h0):
        o, s = fn(q, k, v, log_alpha, beta, h0)
        o = o.astype(jnp.float32)
        return jnp.sum(o * jnp.sin(o)) + jnp.sum(s**2)

    return loss


def _chunked(chunk_size, **kw):
    return lambda *a: dr.gated_delta_rule_chunked(
        *a, chunk_size=chunk_size, precision=HIGHEST, **kw
    )


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("length,chunk", [(64, 16), (100, 32), (37, 64)])
def test_chunked_matches_recurrent(gated, length, chunk):
    args = _inputs(length=length, gated=gated)
    o_ref, s_ref = dr.gated_delta_rule_recurrent(*args)
    o, s = jax.jit(_chunked(chunk))(*args)
    assert o.shape == o_ref.shape and s.shape == s_ref.shape
    assert s.dtype == jnp.float32
    assert _rel(o, o_ref) < 1e-4
    assert _rel(s, s_ref) < 1e-4


@pytest.mark.parametrize("gated", [True, False])
def test_chunked_grads_match_recurrent(gated):
    args = _inputs(length=100, gated=gated)
    argnums = (0, 1, 2, 4, 5) + ((3,) if gated else ())
    g_ref = jax.grad(_loss(dr.gated_delta_rule_recurrent), argnums=argnums)(*args)
    g = jax.jit(jax.grad(_loss(_chunked(32)), argnums=argnums))(*args)
    for a, b in zip(g, g_ref, strict=True):
        assert _rel(a, b) < 1e-4


def test_alpha_none_equals_zero_log_gate():
    q, k, v, _, beta, h0 = _inputs(length=48)
    zeros = jnp.zeros(beta.shape)
    o1, s1 = _chunked(16)(q, k, v, None, beta, h0)
    o2, s2 = _chunked(16)(q, k, v, zeros, beta, h0)
    np.testing.assert_allclose(o1, o2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(s1, s2, rtol=1e-6, atol=1e-6)


def test_delta_rule_semantics_single_step():
    """One step from h0 equals the closed-form generalized Householder update."""
    q, k, v, la, beta, h0 = _inputs(b=1, h=1, length=1)
    with jax.default_matmul_precision("highest"):
        o, s = dr.gated_delta_rule_chunked(q, k, v, la, beta, h0, chunk_size=16)
        expected = _single_step_expected(q, k, v, la, beta, h0)
    np.testing.assert_allclose(s[0, 0], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        o[0, 0, 0],
        np.asarray(expected).T @ np.asarray(q[0, 0, 0]),
        rtol=1e-5,
        atol=1e-5,
    )


def _single_step_expected(q, k, v, la, beta, h0):
    a, bt = jnp.exp(la[0, 0, 0]), beta[0, 0, 0]
    kk, vv, qq, s0 = k[0, 0, 0], v[0, 0, 0], q[0, 0, 0], h0[0, 0]
    del qq
    return a * (jnp.eye(16) - bt * jnp.outer(kk, kk)) @ s0 + bt * jnp.outer(kk, vv)


def test_qk_l2norm_flag():
    q, k, v, la, beta, h0 = _inputs(length=40)
    q2, k2 = 3.0 * q, 2.0 * k
    o1, s1 = _chunked(16)(q, k, v, la, beta, h0)
    o2, s2 = _chunked(16, qk_l2norm=True)(q2, k2, v, la, beta, h0)
    assert _rel(o2, o1) < 1e-5 and _rel(s2, s1) < 1e-5


def test_bf16_inputs_fp32_state():
    args32 = _inputs(length=96)
    q, k, v = (x.astype(jnp.bfloat16) for x in args32[:3])
    o, s = dr.gated_delta_rule(q, k, v, *args32[3:], chunk_size=32)
    assert o.dtype == jnp.bfloat16 and s.dtype == jnp.float32
    o_ref, s_ref = dr.gated_delta_rule_recurrent(
        *(x.astype(jnp.float32) for x in (q, k, v)), *args32[3:]
    )
    assert _rel(o, o_ref) < 2e-2 and _rel(s, s_ref) < 2e-2
    g = jax.grad(
        _loss(lambda *a: dr.gated_delta_rule(*a, chunk_size=32)), argnums=(0, 1, 2)
    )(q, k, v, *args32[3:])
    assert all(x.dtype == jnp.bfloat16 and bool(jnp.all(jnp.isfinite(x))) for x in g)


def test_public_api_dispatch_and_validation():
    args = _inputs(length=50)
    o_rec, _ = dr.gated_delta_rule(*args, implementation="recurrent")
    o_auto, _ = jax.jit(lambda *a: dr.gated_delta_rule(*a, precision=HIGHEST))(*args)
    assert _rel(o_auto, o_rec) < 1e-4
    with pytest.raises(ValueError):
        dr.gated_delta_rule(*args, implementation="nope")
    with pytest.raises(ValueError):
        dr.gated_delta_rule(args[0], args[1][:, :, :10], *args[2:])
    with pytest.raises(ValueError):
        dr.gated_delta_rule(*args[:5], args[5][..., :3])


def test_strong_decay_is_finite():
    """Cumulative gates far below exp's range must not produce NaN/inf."""
    q, k, v, _, beta, h0 = _inputs(length=128)
    la = jnp.full(beta.shape, -50.0)
    o, s = _chunked(64)(q, k, v, la, beta, h0)
    o_ref, s_ref = dr.gated_delta_rule_recurrent(q, k, v, la, beta, h0)
    assert bool(jnp.all(jnp.isfinite(o))) and _rel(o, o_ref) < 1e-4
    g = jax.grad(_loss(_chunked(64)), argnums=(3,))(q, k, v, la, beta, h0)
    assert bool(jnp.all(jnp.isfinite(g[0])))


def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("requires a GPU backend")


@pytest.mark.parametrize("mode", ["fused", "pallas"])
@pytest.mark.parametrize("dk,dv", [(16, 32), (64, 64), (128, 128)])
@pytest.mark.parametrize("gated", [True, False])
def test_pallas_matches_chunked_gpu(dk, dv, gated, mode):
    _requires_gpu()
    args = _inputs(length=200, dk=dk, dv=dv, gated=gated)
    argnums = (0, 1, 2, 4, 5) + ((3,) if gated else ())
    ref = _chunked(64)
    pal = lambda *a: dr.gated_delta_rule_pallas(  # noqa: E731
        *a, chunk_size=64, precision=HIGHEST, mode=mode
    )
    o_ref, s_ref = jax.jit(ref)(*args)
    o, s = jax.jit(pal)(*args)
    assert _rel(o, o_ref) < 1e-4 and _rel(s, s_ref) < 1e-4
    g_ref = jax.jit(jax.grad(_loss(ref), argnums=argnums))(*args)
    g = jax.jit(jax.grad(_loss(pal), argnums=argnums))(*args)
    for a, b in zip(g, g_ref, strict=True):
        assert _rel(a, b) < 1e-4


def test_pallas_default_precision_gpu():
    """Default (TF32) GPU path stays close to the fp32 recurrence."""
    _requires_gpu()
    args = _inputs(length=256, dk=64, dv=64)
    o_ref, s_ref = dr.gated_delta_rule_recurrent(*args)
    o, s = jax.jit(dr.gated_delta_rule)(*args)
    assert _rel(o, o_ref) < 1e-2 and _rel(s, s_ref) < 1e-2


@pytest.mark.parametrize("use_gate", [True, False])
def test_gated_deltanet_cell(use_gate):
    with jax.default_matmul_precision("highest"):
        _check_cell(use_gate)


def _check_cell(use_gate):
    cell = GatedDeltaNetCell(32, nnx.Rngs(0), num_heads=2, use_gate=use_gate)
    x = jax.random.normal(jax.random.key(1), (2, 37, 32))
    y = cell(x)
    assert y.shape == x.shape and bool(jnp.all(jnp.isfinite(y)))
    assert cell(x[0]).shape == (37, 32)
    np.testing.assert_allclose(cell(x[0]), y[0], rtol=1e-5, atol=1e-5)
    # Causality: perturbing the future leaves past outputs unchanged.
    x2 = x.at[:, 20:].add(1.0)
    np.testing.assert_allclose(cell(x2)[:, :20], y[:, :20], rtol=1e-5, atol=1e-5)
    # Recurrent and chunked kernels agree inside the layer.
    cell.implementation = "recurrent"
    np.testing.assert_allclose(cell(x), y, rtol=2e-3, atol=2e-3)


def test_ssm_model_with_gated_deltanet_jit_grad():
    model = SSMModel(
        3,
        32,
        2,
        num_layers=2,
        recurrent_cls=GatedDeltaNetCell,
        recurrent_kwargs=dict(num_heads=2, chunk_size=16),
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (2, 40, 3))

    @nnx.jit
    def loss(model, x):
        return jnp.mean(model(x) ** 2)

    val, grads = nnx.jit(nnx.value_and_grad(loss))(model, x)
    assert np.isfinite(float(val))
    leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


@pytest.mark.parametrize("chunk", [16, 64])
def test_repeated_keys_ill_conditioned_wy(chunk):
    """Identical keys with beta ~ 1 make (I + A) far from identity; the block
    forward substitution must stay exact (a Neumann series would not)."""
    q, k, v, _, beta, h0 = _inputs(length=128, dk=16)
    k = jnp.broadcast_to(k[:, :, :1], k.shape)
    beta = jnp.full(beta.shape, 0.99)
    o_ref, s_ref = dr.gated_delta_rule_recurrent(q, k, v, None, beta, h0)
    o, s = _chunked(chunk)(q, k, v, None, beta, h0)
    assert _rel(o, o_ref) < 1e-4 and _rel(s, s_ref) < 1e-4
    g_ref = jax.grad(_loss(dr.gated_delta_rule_recurrent), argnums=(1, 4))(
        q, k, v, None, beta, h0
    )
    g = jax.grad(_loss(_chunked(chunk)), argnums=(1, 4))(q, k, v, None, beta, h0)
    for a, b in zip(g, g_ref, strict=True):
        assert _rel(a, b) < 1e-4


def test_store_prep_variant_matches():
    args = _inputs(length=96)

    def stored(*a):
        return dr._chunked_impl(
            *a,
            chunk_size=32,
            qk_l2norm=False,
            precision=HIGHEST,
            mode="xla",
            store_prep=True,
        )

    g1 = jax.grad(_loss(stored), argnums=range(6))(*args)
    g2 = jax.grad(_loss(_chunked(32)), argnums=range(6))(*args)
    for a, b in zip(g1, g2, strict=True):
        assert _rel(a, b) < 1e-5


@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_pallas_unit_lower_inverse_gpu(chunk):
    _requires_gpu()
    from probjax.nn.pallas_kernels.kernels import delta_rule_pallas as dp

    a = jnp.tril(0.5 * jax.random.normal(jax.random.key(0), (3, 5, chunk, chunk)), -1)
    eye = jnp.eye(chunk)
    ref = jax.lax.linalg.triangular_solve(
        a + eye, jnp.broadcast_to(eye, a.shape), left_side=True, lower=True,
        unit_diagonal=True,
    )  # fmt: skip
    assert _rel(dp.unit_lower_inverse(a, precision=HIGHEST), ref) < 1e-5


@pytest.mark.parametrize("mode", ["fused", "pallas"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_pallas_modes_default_precision_gpu(mode, dtype):
    """TF32/bf16 tensor-core paths (incl. the fused kernels at Dk = 128)."""
    _requires_gpu()
    args = _inputs(length=256, dk=128, dv=128, dtype=dtype)
    ref = _loss(dr.gated_delta_rule_recurrent)
    f = _loss(lambda *a: dr.gated_delta_rule_pallas(*a, chunk_size=64, mode=mode))
    val_ref, g_ref = jax.value_and_grad(ref, argnums=range(6))(
        *(x.astype(jnp.float32) if x is not None else x for x in args)
    )
    val, g = jax.jit(jax.value_and_grad(f, argnums=range(6)))(*args)
    tol = 2e-2 if dtype == jnp.bfloat16 else 5e-3
    assert abs(float(val) - float(val_ref)) / abs(float(val_ref)) < tol
    for a, b in zip(g, g_ref, strict=True):
        assert _rel(a, b) < 5 * tol

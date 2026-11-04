"""Single-pass (FlashAttention-2 style) attention backward.

``backward_pass_impl="triton_onepass"`` computes dK/dV per KV block and adds
dQ atomically into a float32 buffer. It must agree with the two-pass
(``triton_split``) backward for every feature the kernels support: masks
(incl. block-sparse iteration), dense and functional biases (score-mod
gradients), counter-based dropout, GQA and cross-attention lengths. CPU runs
use Pallas interpret mode; GPU-only tests check the compiled kernels.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.layers.attention import flex_attention
from probjax.nn.pallas_kernels.attention_mask_bias import (
    CausalMask,
    DenseBias,
    DistanceDecayBias,
    LocalWindowMask,
    SameSegmentMask,
)
from probjax.nn.pallas_kernels.kernels.attention import BlockSizes, mha

BLOCKS = BlockSizes(16, 16, 16, 16, 16, 16)


def _inputs(b, lq, lkv, hq, hkv, d, dtype=jnp.float32, seed=0):
    ks = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(ks[0], (b, lq, hq, d), jnp.float32).astype(dtype)
    k = jax.random.normal(ks[1], (b, lkv, hkv, d), jnp.float32).astype(dtype)
    v = jax.random.normal(ks[2], (b, lkv, hkv, d), jnp.float32).astype(dtype)
    g = jax.random.normal(ks[3], (b, lq, hq, d), jnp.float32)
    return q, k, v, g


def _grads(fn, q, k, v, g):
    def loss(q, k, v):
        return jnp.sum(fn(q, k, v).astype(jnp.float32) * g)

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def _segments(b, length):
    ids = (jnp.arange(length) * 3) // length
    return jnp.broadcast_to(ids, (b, length)).astype(jnp.int32)


CASES = {
    "plain": dict(),
    "causal": dict(mask=CausalMask()),
    "window": dict(mask=LocalWindowMask(12)),
    "segments": dict(mask=SameSegmentMask(_segments(1, 48), _segments(1, 48))),
    "dense_bias": dict(
        bias=DenseBias(jax.random.normal(jax.random.key(7), (1, 1, 48, 48)))
    ),
    "decay_bias_causal": dict(mask=CausalMask(), bias=DistanceDecayBias(0.3)),
    "dropout": dict(dropout_rate=0.2, rng=jax.random.key(11)),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("hkv", [2, 1])
def test_onepass_matches_split_interpret(case, hkv):
    q, k, v, g = _inputs(1, 48, 48, 2, hkv, 16, seed=hkv)
    kw = CASES[case]

    def fn(impl):
        return lambda q, k, v: mha(
            q,
            k,
            v,
            sm_scale=0.25,
            block_sizes=BLOCKS,
            backward_pass_impl=impl,
            interpret=True,
            **kw,
        )

    got = _grads(fn("triton_onepass"), q, k, v, g)
    want = _grads(fn("triton_split"), q, k, v, g)
    for name, a, b in zip("qkv", got, want, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert _rel(a, b) < 1e-5, (case, name, _rel(a, b))


@pytest.mark.parametrize("lq,lkv", [(16, 48), (48, 16)])
def test_onepass_cross_attention_interpret(lq, lkv):
    q, k, v, g = _inputs(2, lq, lkv, 2, 2, 16, seed=lq)

    def fn(impl):
        return lambda q, k, v: mha(
            q,
            k,
            v,
            sm_scale=0.25,
            block_sizes=BLOCKS,
            backward_pass_impl=impl,
            interpret=True,
        )

    got = _grads(fn("triton_onepass"), q, k, v, g)
    want = _grads(fn("triton_split"), q, k, v, g)
    for a, b in zip(got, want, strict=True):
        assert _rel(a, b) < 1e-5


# ---------------------------------------------------------------------------
# GPU-only
# ---------------------------------------------------------------------------


def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("requires the GPU (Triton) kernels")


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_gpu_onepass_matches_split(dtype, d, causal):
    _requires_gpu()
    q, k, v, g = _inputs(2, 1000, 1000, 4, 2, d, dtype, seed=d)
    mask = CausalMask() if causal else None

    def fn(impl):
        return lambda q, k, v: flex_attention(
            q, k, v, mask=mask, enable_gqa=True, backward_pass_impl=impl
        )

    got = _grads(fn("triton_onepass"), q, k, v, g)
    want = _grads(fn("triton_split"), q, k, v, g)
    # Same math; only the dQ summation order (atomics), the tile shapes and,
    # for 16-bit inputs, the rounding of the dS tile differ (fp32 dots run in
    # TF32 on GPU).
    tol = 5e-4 if dtype == jnp.float32 else 1e-2
    for name, a, b in zip("qkv", got, want, strict=True):
        assert bool(jnp.all(jnp.isfinite(a.astype(jnp.float32)))), name
        assert _rel(a, b) < tol, (name, _rel(a, b))


@pytest.mark.parametrize(
    "case", ["window", "dense_bias", "decay_bias_causal", "dropout"]
)
def test_gpu_onepass_features_match_split(case):
    _requires_gpu()
    # Raw mha needs block-aligned lengths with biases: 128 tokens.
    q, k, v, g = _inputs(1, 128, 128, 2, 2, 64, seed=3)
    kw = dict(CASES[case])
    if case == "dense_bias":
        kw["bias"] = DenseBias(jax.random.normal(jax.random.key(7), (1, 1, 128, 128)))

    def fn(impl):
        return lambda q, k, v: mha(
            q, k, v, sm_scale=0.125, backward_pass_impl=impl, **kw
        )

    got = _grads(fn("triton_onepass"), q, k, v, g)
    want = _grads(fn("triton_split"), q, k, v, g)
    for name, a, b in zip("qkv", got, want, strict=True):
        assert _rel(a, b) < 5e-4, (case, name, _rel(a, b))

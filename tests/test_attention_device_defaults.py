"""Device-specific attention defaults (probjax.nn.layers.attention_device).

The routing rules are checked on any backend by pretending to be an sm_90
device and stubbing the cuDNN call; the numerics of the routed paths (cuDNN,
the Triton backward choice, the explicit-fused fallback and the Mosaic
``mha_flash`` wrapper) are checked on an H100 only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from probjax.nn.layers import attention_device as ad
from probjax.nn.pallas_kernels import CausalMask, KVLenMask, SeqLenMask

SENTINEL = object()


@pytest.fixture
def sm90(monkeypatch):
    """Pretend the default device is an H100 and record cuDNN calls."""
    calls = []
    monkeypatch.setattr(ad, "_compute_capability_major", lambda: 9)
    monkeypatch.delenv("PROBJAX_ATTENTION_DEVICE_DEFAULTS", raising=False)
    monkeypatch.delenv("PROBJAX_ATTENTION_CUDNN", raising=False)

    def fake_cudnn(q, k, v, **kw):
        calls.append(kw)
        return SENTINEL

    monkeypatch.setattr(ad.jax.nn, "dot_product_attention", fake_cudnn)
    return calls


class _Spec:
    """Shape/dtype stand-in: the routing rules only look at these."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype, self.ndim = tuple(shape), jnp.dtype(dtype), len(shape)


def _qkv(b, lq, lkv, hq, hkv, d, dtype=jnp.bfloat16):
    q = _Spec((b, lq, hq, d), dtype)
    k = _Spec((b, lkv, hkv, d), dtype)
    return q, k, k


def _route(q, k, v, mask=None, bias=None, dropout_rate=0.0):
    return ad.cudnn_attention_or_none(
        q, k, v, mask=mask, bias=bias, dropout_rate=dropout_rate, sm_scale=0.1
    )


# ---------------------------------------------------------------------------
# Rules (any backend)
# ---------------------------------------------------------------------------


def test_no_table_without_known_device(monkeypatch):
    monkeypatch.setattr(ad, "_compute_capability_major", lambda: 8)
    assert ad.device_attention_defaults() is None
    assert ad.resolve_auto_backward(4096, 4096) is None
    assert _route(*_qkv(4, 4096, 4096, 8, 8, 64)) is None


def test_env_disables_table(sm90, monkeypatch):
    monkeypatch.setenv("PROBJAX_ATTENTION_DEVICE_DEFAULTS", "0")
    assert ad.device_attention_defaults() is None
    assert _route(*_qkv(4, 4096, 4096, 8, 8, 64)) is None


def test_env_disables_cudnn_only(sm90, monkeypatch):
    monkeypatch.setenv("PROBJAX_ATTENTION_CUDNN", "0")
    assert _route(*_qkv(4, 4096, 4096, 8, 8, 64)) is None
    assert ad.resolve_auto_backward(4096, 4096) == "triton_split"


def test_auto_backward_choice(sm90):
    # Split everywhere (fused: up to 1.9x slower on asymmetric lengths on H100)...
    assert ad.resolve_auto_backward(131072, 32768) == "triton_split"
    assert ad.resolve_auto_backward(1024, 256) == "triton_split"
    assert ad.resolve_auto_backward(4096, 4096) == "triton_split"
    # ...except few 16-bit queries over long KV, where the single pass wins.
    assert ad.resolve_auto_backward(256, 16384) == "triton_onepass"
    assert ad.resolve_auto_backward(256, 16384, dtype_bytes=4) == "triton_split"
    assert ad.resolve_auto_backward(512, 131072) == "triton_split"


@pytest.mark.parametrize(
    "shape, mask, expect",
    [
        ((2, 131072, 32768, 1, 1, 128), "kvlen", True),  # MLA test-row cross-attention
        ((4, 4096, 4096, 8, 8, 64), None, True),
        ((2, 4096, 4096, 8, 8, 128), "causal", True),
        ((2, 4096, 4096, 8, 2, 128), None, True),  # GQA
        ((2, 3000, 5000, 8, 8, 64), "kvlen", True),  # unaligned lengths
        ((56, 256, 32768, 8, 8, 32), "kvlen", True),  # ISAB inducing points over rows
        ((4, 512, 512, 8, 8, 64), None, False),  # too little work
        ((4, 2048, 2048, 8, 8, 32), None, False),  # d32 below its threshold
        ((8, 32768, 256, 8, 8, 32), None, False),  # query-heavy over short KV
        ((8, 64, 131072, 8, 8, 64), "kvlen", False),  # few queries: split-KV + single pass
        ((2, 2048, 4096, 8, 8, 64), "causal", False),  # causal, unequal lengths
        ((2, 4096, 4096, 8, 8, 64), "seqlen", False),  # not expressible in cuDNN
        ((2, 4096, 4096, 8, 8, 72), None, True),  # head_dim 72 -> padded 128 entry
        ((2, 4096, 4096, 8, 8, 100), None, False),  # head_dim not a multiple of 8
    ],
)
def test_cudnn_routing(sm90, shape, mask, expect):
    b, lq, lkv, hq, hkv, d = shape
    lens = jnp.full((b,), lkv, jnp.int32)
    m = {
        None: None,
        "kvlen": KVLenMask(lens),
        "causal": CausalMask(),
        "seqlen": SeqLenMask(lens),
    }[mask]
    out = _route(*_qkv(b, lq, lkv, hq, hkv, d), mask=m)
    assert (out is SENTINEL) == expect
    if expect:
        kw = sm90[-1]
        assert kw["implementation"] == "cudnn"
        assert kw["is_causal"] == (mask == "causal")
        assert (kw["key_value_seq_lengths"] is not None) == (mask == "kvlen")


def test_cudnn_not_for_fp32_bias_or_dropout(sm90):
    q, k, v = _qkv(4, 4096, 4096, 8, 8, 64, jnp.float32)
    assert _route(q, k, v) is None
    q, k, v = _qkv(4, 4096, 4096, 8, 8, 64)
    assert _route(q, k, v, bias=object()) is None
    assert _route(q, k, v, dropout_rate=0.1) is None
    assert _route(q, k, v) is SENTINEL


# ---------------------------------------------------------------------------
# Numerics on an H100
# ---------------------------------------------------------------------------


def _requires_sm90():
    if jax.default_backend() != "gpu":
        pytest.skip("requires a GPU")
    if ad._compute_capability_major() != 9:
        pytest.skip("requires an sm_90 (Hopper) GPU")


def _inputs(b, lq, lkv, hq, hkv, d, dtype=jnp.bfloat16, seed=0):
    ks = jax.random.split(jax.random.key(seed), 4)
    q = jax.random.normal(ks[0], (b, lq, hq, d), dtype)
    k = jax.random.normal(ks[1], (b, lkv, hkv, d), dtype)
    v = jax.random.normal(ks[2], (b, lkv, hkv, d), dtype)
    g = jax.random.normal(ks[3], (b, lq, hq, d), jnp.float32)
    return q, k, v, g


def _out_and_grads(fn, q, k, v, g):
    out = jax.jit(fn)(q, k, v)
    grads = jax.jit(
        jax.grad(lambda q, k, v: jnp.sum(fn(q, k, v).astype(jnp.float32) * g), (0, 1, 2))
    )(q, k, v)
    return out, grads


def _max_err(a, b):
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def _assert_close(fn, ref, inputs, out_tol=0.02, grad_tol=0.1):
    o, g = _out_and_grads(fn, *inputs)
    ro, rg = _out_and_grads(ref, *inputs)
    assert _max_err(o, ro) < out_tol
    for a, b in zip(g, rg, strict=True):
        assert _max_err(a, b) < grad_tol


@pytest.mark.parametrize(
    "shape, mask",
    [
        ((2, 8192, 4096, 1, 1, 128), "kvlen"),
        ((2, 4096, 4096, 8, 8, 128), "causal"),
        ((2, 4096, 4096, 8, 2, 128), None),
        ((2, 3000, 5000, 8, 8, 64), "kvlen"),
    ],
)
def test_gpu_routed_default_matches_triton(shape, mask):
    _requires_sm90()
    from probjax.nn.layers.attention import flex_attention

    b, lq, lkv, hq, hkv, d = shape
    m = {None: None, "kvlen": KVLenMask(jnp.linspace(lkv // 2, lkv, b).astype(jnp.int32)),
         "causal": CausalMask()}[mask]

    def default(q, k, v):
        return flex_attention(q, k, v, mask=m, enable_gqa=True)

    def triton(q, k, v):
        return flex_attention(q, k, v, mask=m, enable_gqa=True, implementation="triton")

    assert "cudnn" in jax.jit(default).lower(*_inputs(*shape)[:3]).as_text().lower()
    _assert_close(default, triton, _inputs(*shape))


@pytest.mark.parametrize("shape", [(4, 256, 16384, 8, 8, 64), (8, 32768, 256, 8, 8, 32)])
def test_gpu_explicit_fused_asymmetric_matches_split(shape):
    """Tile matching would force < 16-row tiles (formerly interpret mode, wrong dQ)."""
    _requires_sm90()
    from probjax.nn.layers.attention import flex_attention

    def impl(name):
        return lambda q, k, v: flex_attention(
            q, k, v, backward_pass_impl=name, implementation="triton"
        )

    _assert_close(impl("triton_fused"), impl("triton_split"), _inputs(*shape),
                  out_tol=1e-6, grad_tol=1e-6)


@pytest.mark.parametrize(
    "shape, gqa",
    [
        ((2, 4096, 8192, 8, 8, 128), False),  # cross-attention
        ((2, 1000, 2048, 8, 8, 64), False),  # padded query length
        ((2, 2048, 2048, 8, 8, 32), False),  # padded head_dim
        ((2, 2048, 2048, 8, 2, 128), True),  # GQA
    ],
)
def test_gpu_mha_flash_shapes_match_triton(shape, gqa):
    _requires_sm90()
    from probjax.nn.layers.attention import flex_attention
    from probjax.nn.pallas_kernels.kernels.flash_attention3 import mha_flash

    _assert_close(
        lambda q, k, v: mha_flash(q, k, v, enable_gqa=gqa),
        lambda q, k, v: flex_attention(q, k, v, enable_gqa=True, implementation="triton"),
        _inputs(*shape),
    )


def test_mha_flash_rejects_unaligned_kv():
    _requires_sm90()
    from probjax.nn.pallas_kernels.kernels.flash_attention3 import mha_flash

    q, k, v, _ = _inputs(1, 256, 1000, 2, 2, 64)
    with pytest.raises(NotImplementedError, match="KV-length mask"):
        mha_flash(q, k, v)

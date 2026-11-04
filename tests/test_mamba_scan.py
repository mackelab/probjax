"""Mamba (S6) selective scan: CPU chunked path, GPU kernels, gradients, memory."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.pallas_kernels import compute_mamba_scan
from probjax.nn.pallas_kernels.kernels import mamba as mamba_mod
from probjax.nn.pallas_kernels.kernels.mamba_chunked import mamba_scan_chunked

_requires_gpu = pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="needs the GPU (Triton) kernels"
)


def _inputs(batch, seq, dim, state, seed=0, dtype=jnp.float32):
    ks = jax.random.split(jax.random.key(seed), 6)
    x = jax.random.normal(ks[0], (batch, seq, dim))
    a = -jnp.abs(jax.random.normal(ks[1], (state, dim)))
    b = jax.random.normal(ks[2], (batch, seq, state))
    c = jax.random.normal(ks[3], (batch, seq, state))
    delta = jax.nn.softplus(jax.random.normal(ks[4], (batch, seq, dim)))
    d = jax.random.normal(ks[5], (1, dim))
    return tuple(v.astype(dtype) for v in (x, a, b, c, delta, d))


def _grads(fn, args):
    def loss(*xs):
        return 0.5 * jnp.sum(fn(*xs).astype(jnp.float32) ** 2)

    return jax.jit(jax.grad(loss, argnums=tuple(range(6))))(*args)


def _rel_err(out, ref):
    out = np.asarray(out, np.float64)
    ref = np.asarray(ref, np.float64)
    return float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))


def _assert_matches_reference(fn, args, fwd_tol=1e-5, grad_tol=1e-5):
    ref = mamba_mod._mamba_scan_reference
    out = jax.jit(fn)(*args)
    assert out.shape == args[0].shape and out.dtype == args[0].dtype
    assert _rel_err(out, jax.jit(ref)(*args)) < fwd_tol
    for name, g, g_ref in zip(
        ["x", "a", "b", "c", "delta", "d"],
        _grads(fn, args),
        _grads(ref, args),
        strict=True,
    ):
        assert g.shape == g_ref.shape, name
        assert _rel_err(g, g_ref) < grad_tol, (name, _rel_err(g, g_ref))


@pytest.mark.parametrize(
    "seq,chunk,inner",
    [
        (64, 16, "assoc"),
        (100, 32, "assoc"),
        (7, 64, "assoc"),
        (100, 16, "fused"),
        (64, 64, "fused"),
        (7, 32, "fused"),
    ],
)
def test_chunked_matches_sequential_reference(seq, chunk, inner):
    args = _inputs(2, seq, 24, 4, seed=seq)
    fn = functools.partial(mamba_scan_chunked, chunk_size=chunk, inner=inner)
    _assert_matches_reference(fn, args)


def test_chunked_bf16_inputs_keep_dtypes():
    args = _inputs(1, 40, 16, 4, dtype=jnp.bfloat16)
    fn = functools.partial(mamba_scan_chunked, chunk_size=16)
    out = fn(*args)
    assert out.dtype == jnp.bfloat16
    for g, v in zip(_grads(fn, args), args, strict=True):
        assert g.dtype == v.dtype


def test_chunked_vjp_residuals_are_bounded():
    """Regression (memory): the backward must not keep per-step states.

    The ``lax.scan`` reference saves ``[L, B, S, D]`` hidden states for its
    VJP; the chunked path keeps only the inputs and the ``[L/T, B, S, D]``
    chunk-entry states.
    """
    batch, seq, dim, state, chunk = 2, 256, 32, 8, 64
    args = _inputs(batch, seq, dim, state)
    fn = functools.partial(mamba_scan_chunked, chunk_size=chunk)
    _, vjp_fn = jax.vjp(fn, *args)
    sizes = [leaf.size for leaf in jax.tree.leaves(vjp_fn) if hasattr(leaf, "size")]
    full_history = batch * seq * state * dim
    assert max(sizes) <= max(batch * seq * dim, full_history // chunk)
    assert sum(sizes) < full_history // 2


def test_compute_mamba_scan_dispatch_matches_reference():
    """Public API on the current backend (CPU: chunked XLA; GPU: Pallas)."""
    args = _inputs(2, 96, 64, 16, seed=3)
    fn = functools.partial(compute_mamba_scan, seq_tile_size=64, dim_tile_size=128)
    _assert_matches_reference(fn, args)


def test_compute_mamba_scan_cpu_uses_chunked_path(monkeypatch):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU dispatch")

    def _fail(*_args, **_kwargs):
        raise AssertionError("CPU must not use the Pallas op or the full scan")

    monkeypatch.setattr(mamba_mod, "_mamba_scan_op", _fail)
    monkeypatch.setattr(mamba_mod, "_mamba_scan_associative", _fail)
    args = _inputs(1, 16, 8, 4)
    out = compute_mamba_scan(*args, seq_tile_size=8, dim_tile_size=128)
    assert out.shape == args[0].shape


@_requires_gpu
@pytest.mark.parametrize(
    "batch,seq,dim,state",
    [(2, 96, 64, 16), (1, 37, 100, 5), (3, 33, 130, 16), (1, 1, 32, 1)],
)
def test_gpu_kernels_unaligned_shapes(batch, seq, dim, state):
    args = _inputs(batch, seq, dim, state, seed=seq + dim)
    fn = functools.partial(compute_mamba_scan, seq_tile_size=64, dim_tile_size=128)
    _assert_matches_reference(fn, args, fwd_tol=2e-5, grad_tol=2e-5)


@_requires_gpu
def test_gpu_long_sequence_gradients_relative_error():
    """Regression for the reported O(100) absolute grad error at L=8192.

    The absolute error tracks the gradient magnitude (|da| ~ 1e9); relative
    to it, every fp32 implementation (sequential scan included) agrees with
    a float64 reference to ~1e-4. Checked here against the fp32 scan.
    """
    args = _inputs(1, 4096, 128, 16, seed=11)
    fn = functools.partial(compute_mamba_scan, seq_tile_size=64, dim_tile_size=128)
    _assert_matches_reference(fn, args, fwd_tol=1e-4, grad_tol=3e-4)


@_requires_gpu
def test_gpu_dispatch_uses_pallas_kernels(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("GPU must not fall back to the XLA scans")

    monkeypatch.setattr(mamba_mod, "mamba_scan_chunked", _fail)
    monkeypatch.setattr(mamba_mod, "_mamba_scan_associative", _fail)
    args = _inputs(1, 32, 32, 16)
    out = compute_mamba_scan(*args, seq_tile_size=8, dim_tile_size=128)
    assert out.shape == args[0].shape


@_requires_gpu
@pytest.mark.parametrize("segment", [8, 16, 48, 4096])
def test_gpu_segment_parallel_matches_reference(monkeypatch, segment):
    """Segments (incl. a partial last one) combine to the sequential scan."""
    for direction in ("fwd", "bwd"):
        monkeypatch.setitem(mamba_mod._GPU_SCAN_CONFIG[direction], "segment", segment)
    jax.clear_caches()
    args = _inputs(2, 100, 40, 16, seed=segment)
    fn = functools.partial(compute_mamba_scan, seq_tile_size=64, dim_tile_size=128)
    _assert_matches_reference(fn, args, fwd_tol=2e-5, grad_tol=2e-5)
    jax.clear_caches()

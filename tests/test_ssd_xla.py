"""Chunked pure-JAX SSD (``ssd_chunked``) and ``ssd`` dispatch.

CPU tests compare against the per-step reference ``ssd_reference`` in float64.
GPU-only tests (skipped on CPU) cover the XLA fallback for shapes the Triton
kernel rejects and the Pallas kernel at sequence lengths spanning several
chunks (regression: cross-chunk state was carried through an output ref, which
races on GPU because Triton runs grid programs concurrently).
"""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.pallas_kernels.kernels.ssd_xla import ssd_chunked
from tests.test_ssd_ad import ssd_reference

ssd_mod = importlib.import_module("probjax.nn.pallas_kernels.kernels.ssd")


def _requires_accelerator():
    if jax.default_backend() == "cpu":
        pytest.skip("requires an accelerator backend")


def _inputs(batch, groups, heads, seq, dk, dv, *, seed=0, dtype=jnp.float32):
    ks = jax.random.split(jax.random.key(seed), 6)
    q = jax.random.normal(ks[0], (batch, groups, seq, dk), dtype) / np.sqrt(dk)
    k = jax.random.normal(ks[1], (batch, groups, seq, dk), dtype)
    v = jax.random.normal(ks[2], (batch, heads, seq, dv), dtype)
    log_alpha = -jax.nn.sigmoid(jax.random.normal(ks[3], (batch, heads, seq), dtype))
    h0 = jax.random.normal(ks[4], (batch, heads, dk, dv), dtype) * 0.3
    ct = jax.random.normal(ks[5], (batch, heads, seq, dv), dtype)
    return (q, k, v, log_alpha, h0), ct


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _f64(xs):
    return tuple(np.asarray(x, np.float64) for x in xs)


def _ref64(args, ct, *, chunked=False):
    """float64 reference output and gradients (numpy-backed, x64-local).

    ``chunked=True`` uses ``ssd_chunked`` in float64 (validated against the
    per-step reference by the CPU tests) to keep large GPU cases in memory.
    """
    ref = (lambda *a: ssd_chunked(*a, remat="full")) if chunked else ssd_reference
    with jax.enable_x64(True):
        args64 = tuple(jnp.asarray(x, jnp.float64) for x in _f64(args))
        ct64 = jnp.asarray(np.asarray(ct, np.float64))
        out = ref(*args64)
        grads = jax.grad(lambda *a: jnp.vdot(ref(*a), ct64), argnums=tuple(range(5)))(
            *args64
        )
        return np.asarray(out), tuple(np.asarray(g) for g in grads)


SHAPES = [
    # batch, groups, heads, seq, dk, dv
    (2, 2, 4, 37, 3, 5),  # odd L, tiny dims
    (1, 1, 1, 200, 48, 40),  # SSDCell-like single head; dk % 128 != 0
    (2, 1, 3, 130, 16, 8),  # L just above two chunks of 64
    (1, 2, 2, 64, 8, 8),  # exactly one chunk
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("inter_chunk", ["scan", "matmul"])
def test_chunked_matches_reference_fwd_and_grads(shape, chunk_size, inter_chunk):
    args, ct = _inputs(*shape)
    ref_out, ref_grads = _ref64(args, ct)
    fn = lambda *a: ssd_chunked(*a, chunk_size=chunk_size, inter_chunk=inter_chunk)
    assert _rel(fn(*args), ref_out) < 1e-5
    grads = jax.grad(lambda *a: jnp.vdot(fn(*a), ct), argnums=tuple(range(5)))(*args)
    for g, r in zip(grads, ref_grads, strict=True):
        assert _rel(g, r) < 1e-4


@pytest.mark.parametrize("remat", ["states", "none", "full"])
def test_chunked_remat_modes_agree(remat):
    args, ct = _inputs(2, 1, 2, 150, 16, 12)
    ref_out, ref_grads = _ref64(args, ct)
    fn = lambda *a: ssd_chunked(*a, remat=remat)
    grads = jax.grad(lambda *a: jnp.vdot(fn(*a), ct), argnums=tuple(range(5)))(*args)
    assert _rel(fn(*args), ref_out) < 1e-5
    for g, r in zip(grads, ref_grads, strict=True):
        assert _rel(g, r) < 1e-4


def test_chunked_states_remat_saves_only_chunk_states():
    # Reverse mode keeps O(L / C * dk * dv) per head, not the C x C blocks.
    args, _ = _inputs(1, 1, 2, 512, 32, 32)
    ids = {id(a) for a in args}

    def saved_bytes(remat):
        _, vjp = jax.vjp(lambda *a: ssd_chunked(*a, remat=remat, chunk_size=64), *args)
        leaves = [x for x in jax.tree.leaves(vjp) if hasattr(x, "nbytes")]
        return sum(x.nbytes for x in leaves if id(x) not in ids)

    states = 1 * 2 * (512 // 64) * 32 * 32 * 4
    assert saved_bytes("states") <= states + 4096
    assert saved_bytes("full") <= 4096
    assert saved_bytes("none") > 2 * states


def test_chunked_final_state_and_no_h0():
    args, _ = _inputs(2, 2, 4, 99, 8, 6)
    q, k, v, la, h0 = args
    out, final = ssd_chunked(q, k, v, la, h0, return_final_state=True, chunk_size=32)
    # Final state from the per-step recurrence.
    rep = v.shape[1] // q.shape[1]
    kh = np.repeat(np.asarray(k, np.float64), rep, axis=1)
    h = np.asarray(h0, np.float64)
    for t in range(q.shape[2]):
        h = np.exp(np.asarray(la, np.float64)[:, :, t, None, None]) * h + np.einsum(
            "bhd,bhv->bhdv", kh[:, :, t], np.asarray(v, np.float64)[:, :, t]
        )
    assert _rel(final, h) < 1e-5
    out0 = ssd_chunked(q, k, v, la)
    ref0 = ssd_reference(q, k, v, la, jnp.zeros_like(h0))
    assert _rel(out0, ref0) < 1e-5


def test_chunked_jvp_matches_reference():
    args, _ = _inputs(1, 1, 2, 70, 8, 8)
    tangents = tuple(
        jax.random.normal(jax.random.key(i + 10), a.shape) for i, a in enumerate(args)
    )
    _, t_new = jax.jvp(ssd_chunked, args, tangents)
    _, t_ref = jax.jvp(ssd_reference, args, tangents)
    assert _rel(t_new, t_ref) < 1e-4


def test_chunked_bf16_qk_inputs():
    args, _ = _inputs(1, 1, 2, 64, 16, 16)
    q, k, v, la, h0 = args
    out = ssd_chunked(q.astype(jnp.bfloat16), k.astype(jnp.bfloat16), v, la, h0)
    assert out.dtype == jnp.float32
    ref = ssd_reference(
        q.astype(jnp.bfloat16).astype(jnp.float32),
        k.astype(jnp.bfloat16).astype(jnp.float32),
        v,
        la,
        h0,
    )
    assert _rel(out, ref) < 1e-4


def test_ssd_dispatches_to_chunked_on_cpu():
    if jax.default_backend() != "cpu":
        pytest.skip("CPU dispatch")
    args, _ = _inputs(1, 1, 1, 100, 24, 24)
    assert _rel(ssd_mod.ssd(*args), ssd_reference(*args)) < 1e-5
    assert _rel(ssd_mod.ssd(*args[:4]), ssd_chunked(*args[:4])) == 0.0
    with pytest.raises(RuntimeError, match="accelerator"):
        ssd_mod.ssd(*args, implementation="pallas")


def test_ssd_rejects_unknown_implementation():
    args, _ = _inputs(1, 1, 1, 8, 4, 4)
    with pytest.raises(ValueError, match="implementation"):
        ssd_mod.ssd(*args, implementation="triton")


def test_ssd_gpu_unsupported_shape_falls_back_to_chunked(monkeypatch):
    # Dispatch logic only (CPU-safe): pretend to be a GPU whose Triton kernel
    # rejects the shape.
    monkeypatch.setattr(ssd_mod.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        ssd_mod, "_pallas_backend", lambda prefer_mosaic_gpu=None: "triton"
    )
    monkeypatch.setattr(
        ssd_mod, "_gpu_supports_ssd_pallas_for_shape", lambda **_: False
    )

    def _unexpected(*args):
        raise AssertionError("Pallas kernel must not be called")

    monkeypatch.setattr(ssd_mod, "_ssd_op", _unexpected)
    args, _ = _inputs(1, 1, 1, 50, 8, 8)
    out = ssd_mod.ssd(*args)
    assert _rel(out, ssd_reference(*args)) < 1e-5
    with pytest.raises(RuntimeError, match="Reference fallback on GPU is disabled"):
        ssd_mod.ssd(*args, implementation="pallas")


# ------------------------------------------------------------------ GPU only


@pytest.mark.parametrize(
    "shape",
    [
        (2, 1, 1, 1000, 128, 128),  # SSDCell default heads=1, odd-ish L
        (2, 2, 4, 300, 48, 40),  # dims not multiples of 64
        (1, 1, 2, 77, 64, 64),  # L not a multiple of 64
    ],
)
def test_ssd_gpu_unsupported_shapes_use_xla_fallback(shape):
    _requires_accelerator()
    args, ct = _inputs(*shape)
    ref_out, ref_grads = _ref64(args, ct, chunked=True)
    assert _rel(ssd_mod.ssd(*args), ref_out) < 1e-4
    grads = jax.grad(lambda *a: jnp.vdot(ssd_mod.ssd(*a), ct), argnums=tuple(range(5)))(
        *args
    )
    for g, r in zip(grads, ref_grads, strict=True):
        assert _rel(g, r) < 1e-3


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 2, 512, 64, 64),
        (1, 2, 4, 768, 64, 64),
        (2, 2, 8, 2048, 128, 128),
    ],
)
def test_ssd_pallas_multi_chunk_matches_reference_gpu(shape):
    # Regression: seq_len > 256 used to give wrong (even NaN) outputs and
    # gradients on GPU because chunk states were passed between concurrently
    # running Triton programs through an output ref.
    _requires_accelerator()
    args, ct = _inputs(*shape)
    ref_out, ref_grads = _ref64(args, ct, chunked=True)
    fn = lambda *a: ssd_mod.ssd(*a, implementation="pallas")
    # The Triton kernel uses TF32 matmuls: ~5e-4 relative error.
    assert _rel(fn(*args), ref_out) < 5e-3
    grads = jax.grad(lambda *a: jnp.vdot(fn(*a), ct), argnums=tuple(range(5)))(*args)
    for name, g, r in zip(
        ("q", "k", "v", "log_alpha", "h0"), grads, ref_grads, strict=True
    ):
        # dlog_alpha = cumsum(q.dq - k.dk) cancels -> TF32 error amplified.
        assert _rel(g, r) < (2e-2 if name == "log_alpha" else 5e-3), name


def test_ssdcell_single_head_runs_on_gpu():
    _requires_accelerator()
    from flax import nnx

    from probjax.nn.layers.ssm import SSDCell

    cell = SSDCell(32, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 100, 32)) * 0.1
    y = cell(x)
    assert y.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(y)))

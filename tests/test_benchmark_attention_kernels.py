import jax
import jax.numpy as jnp
import pytest

from probjax.nn.layers.attention import dot_product_attention, flex_attention
from probjax.nn.pallas_kernels.flash_attention3 import mha_flash
from probjax.nn.pallas_kernels.mambda import _mamba_scan_reference, compute_mamba_scan
from probjax.nn.pallas_kernels.ssd import ssd, ssd_linear_scan


BATCH_SIZE = 32
SEQ_LEN = 1024
NUM_HEADS = 8
HEAD_DIM = 64

# Cross-attention parameters (Q << KV scenario)
CROSS_Q_LEN = 128
CROSS_KV_LEN = 2048

BLOCK_Q = 128
BLOCK_K = 128
BLOCK_KV = 128
MAX_CONCURRENT_STEPS = 2

MAMBA_BATCH_SIZE = 8
MAMBA_SEQ_LEN = 1024
MAMBA_INNER_DIM = 256
MAMBA_STATE_DIM = 16
MAMBA_SEQ_TILE = 64
MAMBA_DIM_TILE = 128

SSD_BATCH_SIZE = 8
SSD_NUM_GROUPS = 2
SSD_NUM_HEADS = 8
SSD_SEQ_LEN = 512
SSD_DK = 128
SSD_DV = 128


_FLASH3_INCOMPAT_SUBSTRINGS = (
    "flash_attention3 is not available in this jax build",
    "flash_attention3 requires a gpu backend",
    "flash_attention3 requires a mosaic-compatible gpu",
    "causal flash_attention3 is unsupported for cuda runtime versions",
    "causal attention is not supported with the pipeline emitter",
)


def _is_expected_flash3_incompatibility(exc: Exception) -> bool:
    message = str(exc).lower()
    return isinstance(exc, (RuntimeError, NotImplementedError)) and any(
        token in message for token in _FLASH3_INCOMPAT_SUBSTRINGS
    )


def _build_inputs():
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)
    shape = (BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM)
    q = jax.random.normal(kq, shape, dtype=jnp.float16)
    k = jax.random.normal(kk, shape, dtype=jnp.float16)
    v = jax.random.normal(kv, shape, dtype=jnp.float16)
    return q, k, v


def _build_cross_attention_inputs():
    """Build inputs for cross-attention benchmarks (Q << KV scenario)."""
    kq, kk, kv = jax.random.split(jax.random.PRNGKey(42), 3)
    q_shape = (BATCH_SIZE, CROSS_Q_LEN, NUM_HEADS, HEAD_DIM)
    kv_shape = (BATCH_SIZE, CROSS_KV_LEN, NUM_HEADS, HEAD_DIM)
    q = jax.random.normal(kq, q_shape, dtype=jnp.float16)
    k = jax.random.normal(kk, kv_shape, dtype=jnp.float16)
    v = jax.random.normal(kv, kv_shape, dtype=jnp.float16)
    return q, k, v


def _build_mamba_inputs():
    k = jax.random.PRNGKey(11)
    kx, ka, kb, kc, kdlt, kd = jax.random.split(k, 6)
    x = jax.random.normal(
        kx, (MAMBA_BATCH_SIZE, MAMBA_SEQ_LEN, MAMBA_INNER_DIM), dtype=jnp.float32
    )
    a = jax.random.normal(ka, (MAMBA_STATE_DIM, MAMBA_INNER_DIM), dtype=jnp.float32)
    b = jax.random.normal(
        kb, (MAMBA_BATCH_SIZE, MAMBA_SEQ_LEN, MAMBA_STATE_DIM), dtype=jnp.float32
    )
    c = jax.random.normal(
        kc, (MAMBA_BATCH_SIZE, MAMBA_SEQ_LEN, MAMBA_STATE_DIM), dtype=jnp.float32
    )
    delta = jax.random.normal(
        kdlt, (MAMBA_BATCH_SIZE, MAMBA_SEQ_LEN, MAMBA_INNER_DIM), dtype=jnp.float32
    )
    d = jax.random.normal(kd, (1, MAMBA_INNER_DIM), dtype=jnp.float32)
    return x, a, b, c, delta, d


def _build_ssd_inputs():
    k = jax.random.PRNGKey(17)
    kq, kk, kv, kla = jax.random.split(k, 4)
    q = jax.random.normal(
        kq,
        (SSD_BATCH_SIZE, SSD_NUM_GROUPS, SSD_SEQ_LEN, SSD_DK),
        dtype=jnp.float32,
    )
    k_t = jax.random.normal(
        kk,
        (SSD_BATCH_SIZE, SSD_NUM_GROUPS, SSD_SEQ_LEN, SSD_DK),
        dtype=jnp.float32,
    )
    v = jax.random.normal(
        kv,
        (SSD_BATCH_SIZE, SSD_NUM_HEADS, SSD_SEQ_LEN, SSD_DV),
        dtype=jnp.float32,
    )
    log_alpha = jax.random.normal(
        kla,
        (SSD_BATCH_SIZE, SSD_NUM_HEADS, SSD_SEQ_LEN),
        dtype=jnp.float32,
    )
    return q, k_t, v, log_alpha


def _forward_impl(name: str):
    if name == "naive":
        return lambda q, k, v: dot_product_attention(q, k, v)
    if name == "flex":
        return lambda q, k, v: flex_attention(
            q,
            k,
            v,
            deterministic=True,
            dropout_rate=0.0,
            block_q=BLOCK_Q,
            block_k=BLOCK_K,
        )
    if name == "flex_auto":
        return lambda q, k, v: flex_attention(
            q,
            k,
            v,
            deterministic=True,
            dropout_rate=0.0,
            block_q=BLOCK_Q,
            block_k=BLOCK_K,
            backward_pass_impl="auto",  # Test auto-selection
        )
    if name == "flex_split":
        return lambda q, k, v: flex_attention(
            q,
            k,
            v,
            deterministic=True,
            dropout_rate=0.0,
            block_q=BLOCK_Q,
            block_k=BLOCK_K,
            backward_pass_impl="triton_split",  # Force split for comparison
        )
    if name == "flex_fused":
        return lambda q, k, v: flex_attention(
            q,
            k,
            v,
            deterministic=True,
            dropout_rate=0.0,
            block_q=BLOCK_Q,
            block_k=BLOCK_K,
            backward_pass_impl="triton_fused",  # Force fused for comparison
        )
    if name == "flash":
        return lambda q, k, v: mha_flash(
            q,
            k,
            v,
            deterministic=True,
            dropout_rate=0.0,
            block_q=BLOCK_Q,
            block_k=BLOCK_K,
            block_kv=BLOCK_KV,
            max_concurrent_steps=MAX_CONCURRENT_STEPS,
            causal=False,
        )
    raise ValueError(f"Unknown implementation: {name}")


def _maybe_skip_flash(name: str, exc: Exception):
    if name == "flash" and _is_expected_flash3_incompatibility(exc):
        pytest.skip(f"Expected FlashAttention3 incompatibility: {exc}")
    raise exc


def _mamba_impl(name: str):
    if name == "naive":
        return lambda x, a, b, c, delta, d: _mamba_scan_reference(x, a, b, c, delta, d)
    if name == "pallas":
        return lambda x, a, b, c, delta, d: compute_mamba_scan(
            x,
            a,
            b,
            c,
            delta,
            d,
            seq_tile_size=MAMBA_SEQ_TILE,
            dim_tile_size=MAMBA_DIM_TILE,
        )
    raise ValueError(f"Unknown implementation: {name}")


def _ssd_impl(name: str):
    if name == "naive":
        return lambda q, k, v, a: ssd_linear_scan(q, k, v, a)[0]
    if name == "pallas":
        return lambda q, k, v, a: ssd(q, k, v, a)
    raise ValueError(f"Unknown implementation: {name}")


@pytest.mark.gpu
@pytest.mark.benchmark(group="attention_forward")
@pytest.mark.parametrize("impl", ["naive", "flex", "flash"])
def test_benchmark_attention_forward(benchmark, impl):
    q, k, v = _build_inputs()
    fn = jax.jit(_forward_impl(impl))

    try:
        warm = fn(q, k, v)
        warm = jax.block_until_ready(warm)
    except Exception as exc:
        _maybe_skip_flash(impl, exc)

    def run_once():
        out = fn(q, k, v)
        return jax.block_until_ready(out)

    out = benchmark(run_once)
    assert out.shape == q.shape


@pytest.mark.gpu
@pytest.mark.benchmark(group="attention_backward")
@pytest.mark.parametrize("impl", ["naive", "flex", "flash"])
def test_benchmark_attention_backward(benchmark, impl):
    q, k, v = _build_inputs()
    fwd = _forward_impl(impl)
    grad_fn = jax.jit(
        jax.grad(lambda q, k, v: jnp.sum(fwd(q, k, v)), argnums=(0, 1, 2))
    )

    try:
        warm = grad_fn(q, k, v)
        warm = jax.tree_util.tree_map(jax.block_until_ready, warm)
    except Exception as exc:
        _maybe_skip_flash(impl, exc)

    def run_once():
        grads = grad_fn(q, k, v)
        grads = jax.tree_util.tree_map(jax.block_until_ready, grads)
        return grads[0]

    dq = benchmark(run_once)
    assert dq.shape == q.shape


@pytest.mark.gpu
@pytest.mark.benchmark(group="cross_attention_forward")
@pytest.mark.parametrize("impl", ["naive", "flex_auto", "flex_split", "flex_fused"])
def test_benchmark_cross_attention_forward(benchmark, impl):
    """Benchmark cross-attention forward pass (Q << KV scenario)."""
    q, k, v = _build_cross_attention_inputs()
    fn = jax.jit(_forward_impl(impl))

    # Warm up
    warm = fn(q, k, v)
    warm = jax.block_until_ready(warm)

    def run_once():
        out = fn(q, k, v)
        return jax.block_until_ready(out)

    out = benchmark(run_once)
    assert out.shape == q.shape


@pytest.mark.gpu
@pytest.mark.benchmark(group="cross_attention_backward")
@pytest.mark.parametrize("impl", ["naive", "flex_auto", "flex_split", "flex_fused"])
def test_benchmark_cross_attention_backward(benchmark, impl):
    """Benchmark cross-attention backward pass (Q << KV scenario).

    This tests the performance impact of the auto-selection logic and
    demonstrates the efficiency difference between fused vs split backward
    when Q sequence length << KV sequence length.
    """
    q, k, v = _build_cross_attention_inputs()
    fwd = _forward_impl(impl)
    grad_fn = jax.jit(
        jax.grad(lambda q, k, v: jnp.sum(fwd(q, k, v)), argnums=(0, 1, 2))
    )

    # Warm up
    warm = grad_fn(q, k, v)
    warm = jax.tree_util.tree_map(jax.block_until_ready, warm)

    def run_once():
        grads = grad_fn(q, k, v)
        grads = jax.tree_util.tree_map(jax.block_until_ready, grads)
        return grads[0]

    dq = benchmark(run_once)
    assert dq.shape == q.shape


@pytest.mark.gpu
@pytest.mark.benchmark(group="ssm_forward")
@pytest.mark.parametrize("impl", ["naive", "pallas"])
def test_benchmark_mamba_forward(benchmark, impl):
    x, a, b, c, delta, d = _build_mamba_inputs()
    fn = jax.jit(_mamba_impl(impl))
    warm = fn(x, a, b, c, delta, d)
    _ = jax.block_until_ready(warm)

    def run_once():
        out = fn(x, a, b, c, delta, d)
        return jax.block_until_ready(out)

    out = benchmark(run_once)
    assert out.shape == x.shape


@pytest.mark.gpu
@pytest.mark.benchmark(group="ssm_backward")
@pytest.mark.parametrize("impl", ["naive", "pallas"])
def test_benchmark_mamba_backward(benchmark, impl):
    x, a, b, c, delta, d = _build_mamba_inputs()
    fwd = _mamba_impl(impl)
    grad_fn = jax.jit(
        jax.grad(
            lambda x, a, b, c, delta, d: jnp.sum(fwd(x, a, b, c, delta, d)),
            argnums=(0, 1, 2, 3, 4, 5),
        )
    )
    warm = grad_fn(x, a, b, c, delta, d)
    _ = jax.tree_util.tree_map(jax.block_until_ready, warm)

    def run_once():
        grads = grad_fn(x, a, b, c, delta, d)
        grads = jax.tree_util.tree_map(jax.block_until_ready, grads)
        return grads[0]

    dx = benchmark(run_once)
    assert dx.shape == x.shape


@pytest.mark.gpu
@pytest.mark.benchmark(group="ssm_forward")
@pytest.mark.parametrize("impl", ["naive", "pallas"])
def test_benchmark_ssd_forward(benchmark, impl):
    q, k_t, v, log_alpha = _build_ssd_inputs()
    fn = jax.jit(_ssd_impl(impl))
    warm = fn(q, k_t, v, log_alpha)
    _ = jax.block_until_ready(warm)

    def run_once():
        out = fn(q, k_t, v, log_alpha)
        return jax.block_until_ready(out)

    out = benchmark(run_once)
    assert out.shape == (SSD_BATCH_SIZE, SSD_NUM_HEADS, SSD_SEQ_LEN, SSD_DV)


@pytest.mark.gpu
@pytest.mark.benchmark(group="ssm_backward")
@pytest.mark.parametrize("impl", ["naive", "pallas"])
def test_benchmark_ssd_backward(benchmark, impl):
    q, k_t, v, log_alpha = _build_ssd_inputs()
    fwd = _ssd_impl(impl)
    grad_fn = jax.jit(
        jax.grad(
            lambda q, k, v, a: jnp.sum(fwd(q, k, v, a)),
            argnums=(0, 1, 2, 3),
        )
    )
    warm = grad_fn(q, k_t, v, log_alpha)
    _ = jax.tree_util.tree_map(jax.block_until_ready, warm)

    def run_once():
        grads = grad_fn(q, k_t, v, log_alpha)
        grads = jax.tree_util.tree_map(jax.block_until_ready, grads)
        return grads[0]

    dq = benchmark(run_once)
    assert dq.shape == q.shape

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.pallas_kernels import (
    FeatureMap,
    residual_linear_attention,
    residual_linear_attention_pallas,
    residual_linear_attention_scan,
    right_shift_and_zero_pad,
)
from probjax.nn.pallas_kernels.kernels import linear_attention as la
from probjax.nn.pallas_kernels.kernels.linear_attention import (
    residual_linear_attention_chunked,
)

_on_gpu = jax.default_backend() == "gpu"
_requires_gpu = pytest.mark.skipif(not _on_gpu, reason="requires a GPU backend")

# On GPU, fp32 matmuls default to TF32 (~1e-3 relative). Tests that check the
# algorithm (not TF32 rounding) run under "highest" precision there; on CPU this
# is a no-op.
_exact = jax.default_matmul_precision("highest")


def _inputs(*, num_heads=2, num_kv_heads=2, seq_len=8, head_dim=4, dtype=None):
    keys = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(keys[0], (1, num_heads, seq_len, head_dim))
    k = jax.random.normal(keys[1], (1, num_kv_heads, seq_len, head_dim))
    v = jax.random.normal(keys[2], (1, num_kv_heads, seq_len, head_dim))
    h0 = 0.1 * jax.random.normal(keys[3], (1, num_heads, 2 * head_dim, head_dim))
    if dtype is not None:
        q, k, v = (x.astype(dtype) for x in (q, k, v))
    return q, k, v, h0


def _rel(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)


def _loss(fn, feature_map, **kwargs):
    def loss(q, k, v, h0):
        output, final_state = fn(
            q, k, v, h0, window_size=2, feature_map=feature_map, **kwargs
        )
        weights = jnp.cos(jnp.arange(output.size, dtype=jnp.float32))
        return jnp.sum(output.astype(jnp.float32).ravel() * weights) + 0.01 * jnp.sum(
            final_state**2
        )

    return loss


def _grads(fn, feature_map, inputs, **kwargs):
    return jax.grad(_loss(fn, feature_map, **kwargs), argnums=(0, 1, 2, 3))(*inputs)


def test_right_shift_and_zero_pad():
    x = jnp.arange(5)
    np.testing.assert_array_equal(
        right_shift_and_zero_pad(x, 2, axis=0), jnp.array([0, 0, 0, 1, 2])
    )
    np.testing.assert_array_equal(right_shift_and_zero_pad(x, 0, axis=0), x)


@pytest.mark.parametrize("feature_map", list(FeatureMap))
@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_pallas_interpreter_matches_scan(feature_map, num_kv_heads):
    inputs = _inputs(num_kv_heads=num_kv_heads)
    with _exact:
        expected = residual_linear_attention_scan(
            *inputs, window_size=2, feature_map=feature_map
        )
        actual = residual_linear_attention_pallas(
            *inputs, window_size=2, feature_map=feature_map, chunk_size=4
        )
    np.testing.assert_allclose(actual[0], expected[0], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual[1], expected[1], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("feature_map", list(FeatureMap))
def test_pallas_interpreter_gradient_matches_scan(feature_map):
    inputs = _inputs(num_kv_heads=1)
    with _exact:
        expected = _grads(residual_linear_attention_scan, feature_map, inputs)
        actual = _grads(
            residual_linear_attention_pallas, feature_map, inputs, chunk_size=4
        )
    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_grad, expected_grad, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("feature_map", list(FeatureMap))
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "seq_len", "chunk_size"),
    [
        (2, 2, 32, 8),  # divisible
        (4, 2, 37, 8),  # GQA + ragged tail
        (4, 1, 5, 16),  # shorter than one chunk
        (2, 2, 20, 1),  # degenerate chunk == step scan
    ],
)
def test_chunked_matches_scan_forward_and_gradients(
    feature_map, num_heads, num_kv_heads, seq_len, chunk_size
):
    inputs = _inputs(num_heads=num_heads, num_kv_heads=num_kv_heads, seq_len=seq_len)
    with _exact:
        expected = residual_linear_attention_scan(
            *inputs, window_size=2, feature_map=feature_map
        )
        actual = residual_linear_attention_chunked(
            *inputs, window_size=2, feature_map=feature_map, chunk_size=chunk_size
        )
        expected_grads = _grads(residual_linear_attention_scan, feature_map, inputs)
        actual_grads = _grads(
            residual_linear_attention_chunked,
            feature_map,
            inputs,
            chunk_size=chunk_size,
        )
    assert actual[0].shape == expected[0].shape
    assert _rel(actual[0], expected[0]) < 1e-5
    assert _rel(actual[1], expected[1]) < 1e-5
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert actual_grad.dtype == expected_grad.dtype
        assert _rel(actual_grad, expected_grad) < 1e-5


@pytest.mark.parametrize("feature_map", list(FeatureMap))
def test_chunked_bfloat16_close_to_float32_reference(feature_map):
    inputs = _inputs(num_heads=4, num_kv_heads=2, seq_len=48, head_dim=8)
    low = tuple(x.astype(jnp.bfloat16) for x in inputs[:3]) + (inputs[3],)
    expected = residual_linear_attention_scan(
        *inputs, window_size=2, feature_map=feature_map
    )
    actual = residual_linear_attention_chunked(
        *low, window_size=2, feature_map=feature_map, chunk_size=16
    )
    assert actual[0].dtype == jnp.bfloat16
    assert actual[1].dtype == jnp.float32
    assert _rel(actual[0], expected[0]) < 2e-2
    assert _rel(actual[1], expected[1]) < 2e-2
    expected_grads = _grads(residual_linear_attention_scan, feature_map, inputs)
    actual_grads = _grads(
        residual_linear_attention_chunked, feature_map, low, chunk_size=16
    )
    for actual_grad, expected_grad, x in zip(
        actual_grads, expected_grads, low, strict=True
    ):
        assert actual_grad.dtype == x.dtype
        assert _rel(actual_grad, expected_grad) < 3e-2


def test_auto_uses_chunked_off_gpu_and_rejects_unknown_implementation():
    inputs = _inputs(seq_len=24)
    auto = residual_linear_attention(*inputs, window_size=2, chunk_size=8)
    chunked = residual_linear_attention_chunked(*inputs, window_size=2, chunk_size=8)
    if not _on_gpu:
        np.testing.assert_array_equal(auto[0], chunked[0])
    with pytest.raises(ValueError, match="implementation"):
        residual_linear_attention(*inputs, window_size=2, implementation="nope")


def test_gpu_chunk_size_requires_power_of_two_tiles():
    # Regression: Pallas-Triton cannot lower head_dim 48; "auto" must not pick it.
    q = jnp.zeros((1, 1, 256, 48))
    assert la._gpu_chunk_size(q, q, 128) is None
    q = jnp.zeros((1, 1, 256, 32))
    assert la._gpu_chunk_size(q, q, 96) is None


# Shared memory Triton requested for configurations that did NOT fit an RTX 3090
# (101376 bytes): (head_dim, dtype, chunk, precision) -> bytes.
_MEASURED_SMEM = {
    (32, "float32", 128, None): 139264,
    (64, "float32", 64, None): 163840,
    (64, "float32", 128, None): 294912,
    (128, "float32", 16, None): 196608,
    (32, "bfloat16", 128, None): 122880,
    (64, "bfloat16", 64, None): 147456,
    (64, "bfloat16", 128, None): 262144,
    (16, "bfloat16", 128, "highest"): 131072,
    (32, "bfloat16", 128, "highest"): 196608,
    (64, "bfloat16", 128, "highest"): 327680,
    (16, "float32", 128, "highest"): 147456,
    (32, "float32", 128, "highest"): 229376,
    (64, "float32", 32, "highest"): 114688,
    (64, "float32", 64, "highest"): 196608,
    (64, "float32", 128, "highest"): 393216,
}


@pytest.mark.parametrize(("config", "required"), list(_MEASURED_SMEM.items()))
def test_smem_model_never_underestimates_measured_requests(config, required):
    head_dim, dtype, chunk, precision = config
    with jax.default_matmul_precision(precision):
        model = la._kernel_smem_bytes(chunk, 2 * head_dim, head_dim, dtype)
    assert model >= required


@_requires_gpu
@pytest.mark.parametrize("precision", [None, "highest"])
@pytest.mark.parametrize("head_dim", [16, 32, 64])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_gpu_fitted_chunk_compiles(precision, head_dim, dtype):
    inputs = _inputs(seq_len=256, head_dim=head_dim, dtype=dtype)
    with jax.default_matmul_precision(precision):
        output, state = residual_linear_attention_pallas(
            *inputs, window_size=2, chunk_size=128
        )
    assert output.dtype == dtype
    reference = _inputs(seq_len=256, head_dim=head_dim)
    with _exact:
        expected, expected_state = residual_linear_attention_scan(
            *reference, window_size=2
        )
    tol = 2e-3 if dtype == jnp.float32 else 2e-2
    assert _rel(output, expected) < tol
    assert _rel(state, expected_state) < tol


@_requires_gpu
@pytest.mark.parametrize("feature_map", list(FeatureMap))
@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_gpu_pallas_forward_and_gradients_match_scan(feature_map, num_kv_heads):
    inputs = _inputs(num_heads=4, num_kv_heads=num_kv_heads, seq_len=512, head_dim=32)
    with _exact:
        expected = residual_linear_attention_scan(
            *inputs, window_size=2, feature_map=feature_map
        )
        expected_grads = _grads(residual_linear_attention_scan, feature_map, inputs)
    # Default (TF32) precision: the documented GPU accuracy.
    actual = residual_linear_attention_pallas(
        *inputs, window_size=2, feature_map=feature_map
    )
    actual_grads = _grads(residual_linear_attention_pallas, feature_map, inputs)
    assert _rel(actual[0], expected[0]) < 2e-3
    assert _rel(actual[1], expected[1]) < 2e-3
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert _rel(actual_grad, expected_grad) < 2e-2
    with _exact:
        actual_grads = _grads(residual_linear_attention_pallas, feature_map, inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert _rel(actual_grad, expected_grad) < 1e-4


@_requires_gpu
def test_gpu_auto_falls_back_to_chunked_for_unsupported_head_dims():
    for head_dim in (48, 128):
        inputs = _inputs(seq_len=256, head_dim=head_dim)
        output, _ = residual_linear_attention(*inputs, window_size=2)
        with _exact:
            expected, _ = residual_linear_attention_scan(*inputs, window_size=2)
        assert _rel(output, expected) < 2e-3

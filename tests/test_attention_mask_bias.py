import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.pallas_kernels.attention_mask_bias import (
    BlockDiagonalCausalMask,
    BlockDiagonalMask,
    CausalAlibiBias,
    CausalFromBottomRightMask,
    CausalLocalWindowMask,
    CausalMask,
    ComposeMask,
    ConstantBias,
    DenseBias,
    DistanceDecayBias,
    IdentityBias,
    KVLenMask,
    KeyPaddingMask,
    LearnedRelativePositionBias,
    LocalWindowMask,
    MarginalizationMask,
    NoMask,
    NotMask,
    PerHeadScaleBias,
    PrefixLMMask,
    QKVLengthMask,
    SameSegmentMask,
    SeqLenMask,
    SoftCappingBias,
    SumBias,
    SymmetricAlibiBias,
    T5RelativePositionBias,
)


def _idx(n: int) -> jax.Array:
    return jnp.arange(n, dtype=jnp.int32)


def _block_mask_from_dense(mask: jax.Array, block_q: int, block_k: int) -> np.ndarray:
    q_len, k_len = int(mask.shape[0]), int(mask.shape[1])
    nqb = (q_len + block_q - 1) // block_q
    nkb = (k_len + block_k - 1) // block_k
    out = np.zeros((nqb, nkb), dtype=np.bool_)
    for i in range(nqb):
        for j in range(nkb):
            q0, q1 = i * block_q, min((i + 1) * block_q, q_len)
            k0, k1 = j * block_k, min((j + 1) * block_k, k_len)
            out[i, j] = bool(np.asarray(mask[q0:q1, k0:k1]).any())
    return out


def test_no_mask_all_true():
    q_idx = _idx(5)
    k_idx = _idx(7)
    mask = NoMask()(q_idx, k_idx)
    assert mask.shape == (5, 7)
    assert jnp.all(mask)


def test_causal_mask_lower_triangular():
    q_idx = _idx(6)
    k_idx = _idx(6)
    got = CausalMask()(q_idx, k_idx)
    expected = q_idx[:, None] >= k_idx[None, :]
    assert jnp.array_equal(got, expected)


def test_local_window_mask_band():
    q_idx = _idx(8)
    k_idx = _idx(8)
    mask = LocalWindowMask(left_window=2, right_window=1)
    got = mask(q_idx, k_idx)
    dqk = q_idx[:, None] - k_idx[None, :]
    expected = (dqk >= -1) & (dqk <= 2)
    assert jnp.array_equal(got, expected)


def test_local_window_mask_right_none_is_zero():
    q_idx = _idx(8)
    k_idx = _idx(8)
    got_none = LocalWindowMask(left_window=3, right_window=None)(q_idx, k_idx)
    got_zero = LocalWindowMask(left_window=3, right_window=0)(q_idx, k_idx)
    assert jnp.array_equal(got_none, got_zero)


def test_causal_local_window_mask_band():
    q_idx = _idx(8)
    k_idx = _idx(8)
    got = CausalLocalWindowMask(left_window=3)(q_idx, k_idx)
    dqk = q_idx[:, None] - k_idx[None, :]
    expected = (dqk >= 0) & (dqk <= 3)
    assert jnp.array_equal(got, expected)


def test_causal_from_bottom_right_mask_alignment():
    q_len, kv_len = 3, 5
    q_idx = _idx(q_len)
    k_idx = _idx(kv_len)
    m = CausalFromBottomRightMask(q_length=q_len, kv_length=kv_len)
    got = m(q_idx, k_idx)
    expected = (q_idx[:, None] + (kv_len - q_len)) >= k_idx[None, :]
    assert jnp.array_equal(got, expected)


def test_prefix_lm_mask_scalar_prefix():
    q_idx = _idx(6)
    k_idx = _idx(6)
    got = PrefixLMMask(jnp.array(2, dtype=jnp.int32))(q_idx, k_idx)
    expected = (k_idx[None, :] < 2) | (q_idx[:, None] >= k_idx[None, :])
    assert jnp.array_equal(got, expected)


def test_block_diagonal_masks():
    q_ids = jnp.array([0, 0, 1, 1, 2], dtype=jnp.int32)
    k_ids = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
    q_idx = _idx(q_ids.shape[0])
    k_idx = _idx(k_ids.shape[0])

    diag = BlockDiagonalMask(q_ids, k_ids)(q_idx, k_idx)
    expected_diag = q_ids[:, None] == k_ids[None, :]
    assert jnp.array_equal(diag, expected_diag)

    causal = BlockDiagonalCausalMask(q_ids, k_ids)(q_idx, k_idx)
    expected_causal = expected_diag & (q_idx[:, None] >= k_idx[None, :])
    assert jnp.array_equal(causal, expected_causal)


def test_qkv_length_mask_clips_both_axes():
    q_len, kv_len = 7, 9
    mask = QKVLengthMask(q_length=4, kv_length=6)
    got = mask(_idx(q_len), _idx(kv_len))
    expected = (_idx(q_len)[:, None] < 4) & (_idx(kv_len)[None, :] < 6)
    assert jnp.array_equal(got, expected)


def test_seq_len_mask_dense_matches_reference():
    lengths = jnp.array([2, 4, 6], dtype=jnp.int32)
    q_len = 6
    dense = SeqLenMask(lengths).dense(q_len, q_len, batch_size=3, num_heads=2)

    q_idx = _idx(q_len)
    valid = q_idx[None, :] < lengths[:, None]
    rect = valid[:, :, None] & valid[:, None, :]
    diag = jnp.eye(q_len, dtype=bool)[None, :, :]
    expected = (rect | diag)[:, None, :, :]
    expected = jnp.broadcast_to(expected, dense.shape)
    assert jnp.array_equal(dense, expected)


def test_kv_len_mask_dense_matches_reference():
    lengths = jnp.array([2, 4, 6], dtype=jnp.int32)
    q_len = 5
    kv_len = 6
    dense = KVLenMask(lengths).dense(q_len, kv_len, batch_size=3, num_heads=2)

    k_idx = _idx(kv_len)
    valid_k = k_idx[None, :] < lengths[:, None]  # [B, K]
    expected = valid_k[:, None, None, :]
    expected = jnp.broadcast_to(expected, dense.shape)
    assert jnp.array_equal(dense, expected)


def test_same_segment_mask_same_id_only():
    q_ids = jnp.array([0, 1, 1, 2], dtype=jnp.int32)
    k_ids = jnp.array([1, 0, 2, 1, 2], dtype=jnp.int32)
    got = SameSegmentMask(q_ids, k_ids)(_idx(4), _idx(5))
    expected = q_ids[:, None] == k_ids[None, :]
    assert jnp.array_equal(got, expected)


def test_marginalization_mask_rect_plus_diag():
    m = jnp.array([True, False, True, False], dtype=bool)
    q_idx = _idx(4)
    k_idx = _idx(4)
    got = MarginalizationMask(m)(q_idx, k_idx)
    expected = (m[:, None] & m[None, :]) | (q_idx[:, None] == k_idx[None, :])
    assert jnp.array_equal(got, expected)


def test_key_padding_mask_call_with_block_lengths_vector():
    q_idx = _idx(4)
    k_idx = _idx(6)
    seg_q = jnp.array([1, 3, 5, 6], dtype=jnp.int32)
    got = KeyPaddingMask(jnp.array([6], dtype=jnp.int32))(q_idx, k_idx, seg_q=seg_q)
    expected = k_idx[None, :] < seg_q[:, None]
    assert jnp.array_equal(got, expected)


def test_compose_mask_and_or_xor_not():
    q_len = 8
    kv_len = 8
    a = CausalMask()
    b = LocalWindowMask(left_window=2, right_window=1)
    q_idx, k_idx = _idx(q_len), _idx(kv_len)

    and_mask = (a & b)(q_idx, k_idx)
    or_mask = (a | b)(q_idx, k_idx)
    xor_mask = (a ^ b)(q_idx, k_idx)
    not_mask = (~a)(q_idx, k_idx)

    va = a(q_idx, k_idx)
    vb = b(q_idx, k_idx)
    assert jnp.array_equal(and_mask, va & vb)
    assert jnp.array_equal(or_mask, va | vb)
    assert jnp.array_equal(xor_mask, va ^ vb)
    assert jnp.array_equal(not_mask, ~va)


def test_compose_mask_get_data_two_stateful_raises():
    a = SeqLenMask(jnp.array([3, 4], dtype=jnp.int32))
    b = KeyPaddingMask(jnp.array([2, 5], dtype=jnp.int32))
    composed = ComposeMask("and", a, b)
    with pytest.raises(ValueError, match="Cannot compose two stateful masks"):
        composed.get_data(q_seq_len=6, kv_seq_len=6)


def test_sum_bias_adds_biases_without_double_counting_scores():
    q_idx = _idx(3)
    k_idx = _idx(4)
    scores = jnp.ones((3, 4), dtype=jnp.float32)
    bias = SumBias(ConstantBias(0.25), ConstantBias(-0.1))
    got = bias(scores, jnp.array(0), q_idx, k_idx)
    expected = scores + 0.25 - 0.1
    assert jnp.allclose(got, expected)


def test_sum_bias_with_dense_and_constant():
    q_len, kv_len = 3, 5
    scores = jnp.zeros((q_len, kv_len), dtype=jnp.float32)
    q_idx = _idx(q_len)
    k_idx = _idx(kv_len)
    dense = jnp.arange(q_len * kv_len, dtype=jnp.float32).reshape(1, 1, q_len, kv_len)
    bias = DenseBias(dense) + ConstantBias(2.0)
    got = bias(scores, jnp.array(0), q_idx, k_idx)
    assert jnp.allclose(got, dense[0, 0] + 2.0)


def test_dense_bias_call_with_data_adds_tile():
    scores = jnp.ones((2, 3), dtype=jnp.float32)
    tile = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    b = DenseBias(jnp.zeros((1, 1, 2, 3), dtype=jnp.float32))
    got = b(scores, jnp.array(0), _idx(2), _idx(3), data=tile)
    assert jnp.allclose(got, scores + tile)


def test_dense_bias_call_fallback_b1():
    q_len, kv_len = 3, 4
    dense = jnp.arange(q_len * kv_len, dtype=jnp.float32).reshape(1, 1, q_len, kv_len)
    b = DenseBias(dense)
    scores = jnp.zeros((q_len, kv_len), dtype=jnp.float32)
    got = b(scores, jnp.array(0), _idx(q_len), _idx(kv_len), data=None)
    assert jnp.allclose(got, dense[0, 0])


def test_dense_bias_call_fallback_b_gt_1_raises():
    q_len, kv_len = 3, 4
    dense = jnp.zeros((2, 1, q_len, kv_len), dtype=jnp.float32)
    b = DenseBias(dense)
    with pytest.raises(ValueError, match="requires bias batch dimension to be 1"):
        _ = b(jnp.zeros((q_len, kv_len)), jnp.array(0), _idx(q_len), _idx(kv_len))


def test_dense_bias_dense_broadcast():
    q_len, kv_len = 3, 4
    dense = jnp.arange(q_len * kv_len, dtype=jnp.float32).reshape(1, 1, q_len, kv_len)
    got = DenseBias(dense).dense(q_len, kv_len, batch_size=2, num_heads=3)
    expected = jnp.broadcast_to(dense, (2, 3, q_len, kv_len))
    assert jnp.array_equal(got, expected)


def test_bias_value_formulas():
    q_idx = _idx(4)
    k_idx = _idx(4)
    h = jnp.array(1, dtype=jnp.int32)
    scores = jnp.zeros((4, 4), dtype=jnp.float32)

    slope = 2.0 ** (-(int(h) + 1))

    causal = CausalAlibiBias()(scores, h, q_idx, k_idx)
    expected_causal = -slope * jnp.maximum(q_idx[:, None] - k_idx[None, :], 0)
    assert jnp.allclose(causal, expected_causal)

    sym = SymmetricAlibiBias()(scores, h, q_idx, k_idx)
    expected_sym = -slope * jnp.abs(q_idx[:, None] - k_idx[None, :])
    assert jnp.allclose(sym, expected_sym)

    dist = DistanceDecayBias(alpha=0.7)(scores, h, q_idx, k_idx)
    expected_dist = -0.7 * jnp.abs(q_idx[:, None] - k_idx[None, :])
    assert jnp.allclose(dist, expected_dist)

    ident = IdentityBias()(scores, h, q_idx, k_idx)
    assert jnp.array_equal(ident, scores)

    cst = ConstantBias(0.3)(scores, h, q_idx, k_idx)
    assert jnp.allclose(cst, scores + 0.3)


def test_t5_relative_position_bias_values():
    q_idx = _idx(4)
    k_idx = _idx(4)
    scores = jnp.zeros((4, 4), dtype=jnp.float32)
    table = jnp.arange(32, dtype=jnp.float32).reshape(2, 16)
    b = T5RelativePositionBias(
        table, num_buckets=16, max_distance=8, bidirectional=True
    )
    out_h0 = b(scores, jnp.array(0), q_idx, k_idx)
    out_h1 = b(scores, jnp.array(1), q_idx, k_idx)
    assert out_h0.shape == scores.shape
    assert out_h1.shape == scores.shape
    assert not jnp.array_equal(out_h0, out_h1)


def test_learned_relative_position_bias_clipping():
    q_idx = _idx(3)
    k_idx = _idx(5)
    scores = jnp.zeros((3, 5), dtype=jnp.float32)
    max_d = 2
    table = jnp.arange(2 * max_d + 1, dtype=jnp.float32)
    b = LearnedRelativePositionBias(table, max_distance=max_d)
    got = b(scores, jnp.array(0), q_idx, k_idx)
    rel = jnp.clip(k_idx[None, :] - q_idx[:, None], -max_d, max_d) + max_d
    expected = table[rel]
    assert jnp.allclose(got, expected)


def test_soft_capping_bias_and_grad():
    scores = jnp.array([[-50.0, 0.0, 50.0]], dtype=jnp.float32)
    b = SoftCappingBias(softcap=10.0)
    out = b(scores, jnp.array(0), _idx(1), _idx(3))
    assert jnp.all(jnp.abs(out) <= 10.0 + 1e-5)
    g = b.grad(scores, jnp.array(0), _idx(1), _idx(3))
    assert g.shape == scores.shape
    assert jnp.all(g >= 0.0)
    assert jnp.all(g <= 1.0)


def test_per_head_scale_bias_scales_by_head():
    scores = jnp.ones((2, 3), dtype=jnp.float32)
    b = PerHeadScaleBias(jnp.array([2.0, 0.5], dtype=jnp.float32))
    out0 = b(scores, jnp.array(0), _idx(2), _idx(3))
    out1 = b(scores, jnp.array(1), _idx(2), _idx(3))
    assert jnp.allclose(out0, 2.0)
    assert jnp.allclose(out1, 0.5)


@pytest.mark.parametrize(
    "mask",
    [
        NoMask(),
        CausalMask(),
        LocalWindowMask(left_window=2, right_window=1),
        QKVLengthMask(q_length=3, kv_length=4),
        KVLenMask(jnp.array([3, 4], dtype=jnp.int32)),
        SameSegmentMask(
            jnp.array([0, 1, 1, 2], dtype=jnp.int32),
            jnp.array([1, 0, 2, 1, 2], dtype=jnp.int32),
        ),
        MarginalizationMask(jnp.array([True, False, True, False], dtype=bool)),
        NotMask(CausalMask()),
        CausalMask() & LocalWindowMask(left_window=2, right_window=1),
    ],
)
def test_mask_pytree_roundtrip(mask):
    q_len, kv_len = 4, 5
    flat, aux = mask.tree_flatten()
    rebuilt = mask.__class__.tree_unflatten(aux, flat)
    got = rebuilt(_idx(q_len), _idx(kv_len))
    exp = mask(_idx(q_len), _idx(kv_len))
    assert jnp.array_equal(got, exp)


@pytest.mark.parametrize(
    "bias",
    [
        IdentityBias(),
        ConstantBias(0.2),
        CausalAlibiBias(),
        SymmetricAlibiBias(),
        DistanceDecayBias(0.5),
        DenseBias(jnp.arange(12, dtype=jnp.float32).reshape(1, 1, 3, 4)),
        ConstantBias(0.1) + DistanceDecayBias(0.3),
    ],
)
def test_bias_pytree_roundtrip(bias):
    q_len, kv_len = 3, 4
    scores = jnp.zeros((q_len, kv_len), dtype=jnp.float32)
    h = jnp.array(0, dtype=jnp.int32)
    q_idx, k_idx = _idx(q_len), _idx(kv_len)
    flat, aux = bias.tree_flatten()
    rebuilt = bias.__class__.tree_unflatten(aux, flat)
    got = rebuilt(scores, h, q_idx, k_idx)
    exp = bias(scores, h, q_idx, k_idx)
    assert jnp.allclose(got, exp)


def test_no_mask_block_mask_none():
    assert NoMask().block_mask(16, 12, 4, 3) is None


def test_causal_block_mask_matches_dense_nonempty_blocks():
    q_len, kv_len = 11, 10
    block_q, block_k = 4, 3
    m = CausalMask()
    dense = np.asarray(m(_idx(q_len), _idx(kv_len)))
    expected = _block_mask_from_dense(dense, block_q, block_k)
    got = m.block_mask(q_len, kv_len, block_q, block_k)
    assert np.array_equal(got, expected)


def test_local_window_block_mask_matches_dense_nonempty_blocks():
    q_len, kv_len = 11, 12
    block_q, block_k = 4, 5
    m = LocalWindowMask(left_window=3, right_window=2)
    dense = np.asarray(m(_idx(q_len), _idx(kv_len)))
    expected = _block_mask_from_dense(dense, block_q, block_k)
    got = m.block_mask(q_len, kv_len, block_q, block_k)
    assert np.array_equal(got, expected)


def test_qkv_length_block_mask_matches_dense_nonempty_blocks():
    q_len, kv_len = 9, 11
    block_q, block_k = 4, 3
    m = QKVLengthMask(q_length=6, kv_length=7)
    dense = np.asarray(m(_idx(q_len), _idx(kv_len)))
    expected = _block_mask_from_dense(dense, block_q, block_k)
    got = m.block_mask(q_len, kv_len, block_q, block_k)
    assert np.array_equal(got, expected)


def test_same_segment_get_data_block_spec_returns_tuple():
    m = SameSegmentMask(jnp.array([0, 1, 1], dtype=jnp.int32), None)
    spec = m.get_data_block_spec(q_len=3, kv_len=3, block_q=2, block_k=2)
    assert isinstance(spec, tuple)
    assert len(spec) == 2
    assert spec[1] is None

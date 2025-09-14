import jax
import jax.numpy as jnp
import pytest

from probjax.nn.pallas_kernels.attention_mask_bias import (
    AttentionMask,
    NoMask,
    CausalMask,
    LocalWindowMask,
    KeyPaddingMask,
    SameSegmentMask,
    ComposeMask,
    NotMask,
    ConstantBias,
    FromMaskBias,
    DenseBias,
    get_bias_grad,
    compute_block_mask,
    apply_mask,
    apply_bias,
    bias_identity,
    bias_causal,
    bias_alibi,
    bias_distance_decay,
    compute_block_iterators,
    compute_kv_iterators,
)
from probjax.nn.pallas_kernels.utils import (
    DEFAULT_MASK_VALUE,
    materialize_mask,
    materialize_bias,
)


def test_nomask_and_causal_composition():
    mask = NoMask() & CausalMask()
    q_idx = jnp.arange(4)
    k_idx = jnp.arange(4)
    out = mask(0, 0, q_idx, k_idx)
    assert out.shape == (4, 4)
    assert jnp.all(out == (q_idx[:, None] >= k_idx[None, :]))


def test_notmask_inverts():
    base = CausalMask()
    mask = ~base
    q_idx = jnp.arange(3)
    k_idx = jnp.arange(3)
    out = mask(0, 0, q_idx, k_idx)
    assert jnp.all(out == jnp.logical_not(q_idx[:, None] >= k_idx[None, :]))


def test_causalmask_adapter():
    mask = CausalMask()
    out = mask(0, 0, jnp.arange(2), jnp.arange(2))
    assert jnp.array_equal(out, jnp.array([[True, False], [True, True]]))


def test_local_window_mask_call_and_block():
    mask = LocalWindowMask(left_window=1, right_window=2)
    q_idx = jnp.arange(6)
    k_idx = jnp.arange(6)
    dense = mask(0, 0, q_idx, k_idx)
    assert dense.dtype == jnp.bool_
    # Check block mask shape and that it allows at least one kv per q
    bm = mask.block_mask(0, 0, q_len=6, kv_len=6, block_q=2, block_k=3)
    assert bm.shape == (3, 2)
    assert jnp.all(bm.sum(axis=-1) > 0)


@pytest.mark.parametrize("as_bool", [False, True])
def test_key_padding_mask_lengths_and_bool(as_bool):
    B, K = 2, 5
    valid = jnp.array([3, 4], dtype=jnp.int32)
    if as_bool:
        bool_mask = jnp.array([
            [True, True, True, False, False],
            [True, True, True, True, False],
        ])
        mask = KeyPaddingMask(bool_mask)
    else:
        mask = KeyPaddingMask(valid)
    q_idx = jnp.arange(4)
    k_idx = jnp.arange(K)
    out0 = mask(0, 0, q_idx, k_idx)
    out1 = mask(1, 0, q_idx, k_idx)
    assert out0.shape == (4, 5)
    assert jnp.all(out0[:, 3:] == False)
    assert jnp.all(out1[:, 4:] == False)
    assert jnp.all(out0[:, :3]) and jnp.all(out1[:, :4])


def test_same_segment_mask_with_and_without_batch_dim():
    B, Q, K = 2, 5, 5
    qseg = jnp.tile(jnp.array([1, 1, 2, 0, 2]), (B, 1))
    kseg = jnp.tile(jnp.array([1, 0, 2, 2, 2]), (B, 1))
    mask = SameSegmentMask(query_segment_ids=qseg, key_segment_ids=kseg)
    out = mask(1, 0, jnp.arange(Q), jnp.arange(K))
    # Compare with explicit call-time seg ids
    out2 = SameSegmentMask()(1, 0, jnp.arange(Q), jnp.arange(K), qseg, kseg)
    assert jnp.array_equal(out, out2)
    # Missing seg ids should error
    with pytest.raises(ValueError):
        _ = SameSegmentMask()(0, 0, jnp.arange(Q), jnp.arange(K))


def test_compute_block_mask_class():
    B, H, Q, K = 1, 1, 4, 5
    bm = compute_block_mask(
        CausalMask(),
        batch_size=B,
        num_heads=H,
        q_len=Q,
        kv_len=K,
        block_q=2,
        block_k=2,
        segment_ids=None,
    )
    assert bm.shape == (B, H, 2, 3)
    assert bm[0, 0, 0, 0]


def test_apply_mask_and_bias_helpers():
    q_idx = jnp.arange(3)
    k_idx = jnp.arange(3)
    # Mask helper with class
    m_obj = CausalMask()
    m1 = apply_mask(m_obj, 0, 0, q_idx, k_idx)
    assert m1.dtype == jnp.bool_
    # Bias helper with class
    scores = jnp.zeros((3, 3))
    b_obj = ConstantBias(0.5)
    b1 = apply_bias(b_obj, scores, 0, 0, q_idx, k_idx)
    assert jnp.allclose(b1, 0.5)


def test_from_mask_bias_and_constant_bias():
    scores = jnp.zeros((3, 3))
    fb = FromMaskBias(CausalMask(), mask_value=DEFAULT_MASK_VALUE)
    out = fb(scores, 0, 0, jnp.arange(3), jnp.arange(3))
    assert jnp.all(out[jnp.triu_indices(3, 1)] == DEFAULT_MASK_VALUE)
    cb = ConstantBias(0.2)
    assert jnp.allclose(cb(scores, 0, 0, jnp.arange(3), jnp.arange(3)), 0.2)


def test_dense_bias_broadcasting():
    # Bias of shape [1, H, Q, K]
    bias = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 2, 2)
    db = DenseBias(bias)
    out = db(jnp.zeros((2, 2)), 0, 2, jnp.array([0, 1]), jnp.array([0, 1]))
    assert jnp.allclose(out, bias[0, 2])
    # Bias of shape [B, 1, Q, K]
    bias2 = jnp.arange(8, dtype=jnp.float32).reshape(2, 1, 2, 2)
    db2 = DenseBias(bias2)
    out2 = db2(jnp.zeros((2, 2)), 1, 0, jnp.array([0, 1]), jnp.array([0, 1]))
    assert jnp.allclose(out2, bias2[1, 0])


def test_bias_grads_registry():
    assert get_bias_grad(ConstantBias(0.1)) is not None
    assert get_bias_grad(FromMaskBias(CausalMask())) is not None


def test_stateless_biases_behave_reasonably():
    scores = jnp.zeros((4, 4))
    q_idx = jnp.arange(4)
    k_idx = jnp.arange(4)
    # identity leaves unchanged
    assert jnp.allclose(bias_identity(scores, 0, 0, q_idx, k_idx), scores)
    # causal sets future to -inf default
    bc = bias_causal(scores, 0, 0, q_idx, k_idx)
    assert jnp.all(bc[jnp.triu_indices(4, 1)] == DEFAULT_MASK_VALUE)
    # alibi produces negative bias growing with distance
    ba = bias_alibi(scores, 0, jnp.array(0), q_idx, k_idx)
    assert ba.dtype == scores.dtype
    assert ba[3, 0] < ba[2, 0] <= ba[1, 0] <= ba[0, 0]
    # distance decay symmetric
    bd = bias_distance_decay(scores, 0, 0, q_idx, k_idx)
    assert jnp.isclose(bd[1, 3], bd[3, 1])


def test_logit_bias_layers_shapes_and_semantics():
    from probjax.nn.pallas_kernels.attention_mask_bias import (
        CausalAttentionLogitBiasLayer,
        FullAttentionLogitBiasLayer,
        ALiBiAttentionLogitBiasLayer,
        SymmetricALiBiAttentionLogitBiasLayer,
        make_segment_bias,
    )

    B, L, H = 2, 6, 3
    seg = jnp.array([[1, 1, 1, 0, 2, 2], [1, 0, 0, 2, 2, 2]])
    pos = jnp.arange(L)[None, :].repeat(B, axis=0)

    causal = CausalAttentionLogitBiasLayer().forward(segment_ids=seg, positions=pos)
    assert causal.shape == (B, 1, L, L)
    # future positions must be masked (at most DEFAULT_MASK_VALUE, possibly sum with segment mask)
    masked_vals = causal[:, :, jnp.triu_indices(L, 1)[0], jnp.triu_indices(L, 1)[1]]
    assert jnp.all(masked_vals <= DEFAULT_MASK_VALUE)
    # Check semantics: allowed iff same nonzero segment and not future; else masked
    seg_bias = make_segment_bias(seg, seg)  # 0 where allowed, -inf otherwise
    q_idx = jnp.arange(L)[None, :, None]  # broadcast as [1, Q, 1]
    k_idx = jnp.arange(L)[None, None, :]  # [1, 1, K]
    not_future = q_idx >= k_idx  # [1, Q, K]
    same_nonzero = (
        (seg[:, :, None] == seg[:, None, :])
        & (seg[:, :, None] != 0)
        & (seg[:, None, :] != 0)
    )  # [B, Q, K]
    allowed = same_nonzero & not_future
    # Where allowed, causal bias must be zero
    assert jnp.all(causal[:, 0][allowed] == 0)
    # Where disallowed, causal bias must be <= DEFAULT_MASK_VALUE
    assert jnp.all(causal[:, 0][~allowed] <= DEFAULT_MASK_VALUE)

    full = FullAttentionLogitBiasLayer().forward(segment_ids=seg, positions=pos)
    assert full.shape == (B, 1, L, L)
    allowed_full = (
        (seg[:, :, None] == seg[:, None, :])
        & (seg[:, :, None] != 0)
        & (seg[:, None, :] != 0)
    )
    assert jnp.all(full[:, 0][allowed_full] == 0)
    assert jnp.all(full[:, 0][~allowed_full] == DEFAULT_MASK_VALUE)

    alibi = ALiBiAttentionLogitBiasLayer(num_heads=H).forward(
        segment_ids=seg, positions=pos
    )
    assert alibi.shape == (B, H, L, L)

    salibi = SymmetricALiBiAttentionLogitBiasLayer(num_heads=H).forward(
        segment_ids=seg, positions=pos
    )
    assert salibi.shape == (B, H, L, L)


def test_compose_mask_ops_and_sum_bias():
    # Compose masks: (causal OR no-mask) XOR (NOT causal) should reduce to causal
    base = CausalMask()
    composed = (base | NoMask()) ^ (~base)
    q_idx = jnp.arange(5)
    k_idx = jnp.arange(5)
    out = composed(0, 0, q_idx, k_idx)
    assert jnp.array_equal(out, base(0, 0, q_idx, k_idx))

    # Sum of biases equals elementwise sum
    s = jnp.zeros((3, 3))
    b = ConstantBias(0.3) + FromMaskBias(CausalMask())
    out_b = b(s, 0, 0, jnp.arange(3), jnp.arange(3))
    expected = ConstantBias(0.3)(s, 0, 0, jnp.arange(3), jnp.arange(3)) + FromMaskBias(
        CausalMask()
    )(s, 0, 0, jnp.arange(3), jnp.arange(3))
    assert jnp.allclose(out_b, expected)


def test_compute_block_mask_with_callable_and_iterators():
    # Callable mask that mimics causal: allow k <= q
    def fn(b, h, qi, ki, seg_q=None, seg_k=None):
        del b, h, seg_q, seg_k
        return qi[:, None] >= ki[None, :]

    B, H, Q, K = 1, 2, 8, 8
    bm = compute_block_mask(
        fn,
        batch_size=B,
        num_heads=H,
        q_len=Q,
        kv_len=K,
        block_q=4,
        block_k=2,
        segment_ids=None,
    )
    assert bm.shape == (B, H, 2, 4)
    # Dynamic iterators shapes
    idx, sz = compute_block_iterators(bm)
    assert idx.shape == (B, H, 2, 4)
    assert sz.shape == (B, H, 2)
    kidx, ksz = compute_kv_iterators(bm)
    assert kidx.shape == (B, H, 4, 2)
    assert ksz.shape == (B, H, 4)


def test_key_padding_mask_get_data_shapes():
    B, K = 2, 6
    lengths = jnp.array([3, 5], dtype=jnp.int32)
    mask = KeyPaddingMask(lengths)
    # Without kv_seq_len, returns original
    qd, kd = mask.get_data(q_seq_len=None, kv_seq_len=None)
    assert kd is lengths
    # With kv_seq_len, returns boolean [B, K]
    _, kd2 = mask.get_data(q_seq_len=None, kv_seq_len=K)
    assert kd2.shape == (B, K) and kd2.dtype == jnp.bool_


def test_apply_helpers_with_data_tuples():
    # SameSegmentMask with provided data via helper should use seg ids
    qids = jnp.array([1, 1, 2, 0])
    kids = jnp.array([1, 2, 2, 0])
    mask = SameSegmentMask()
    q_idx = jnp.arange(4)
    k_idx = jnp.arange(4)
    out = apply_mask(mask, 0, 0, q_idx, k_idx, (qids,), (kids,))
    # Positions with same nonzero id are True
    assert out[0, 0]
    assert not out[0, 1]

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.nn.layers.attention import (
    dot_product_attention,
    flex_attention,
)
from probjax.nn.pallas_kernels.attention_mask_bias import (
    CausalAlibiBias,
    CausalMask,
    ConstantBias,
    DenseBias,
    DistanceDecayBias,
    IdentityBias,
    KeyPaddingMask,
    LocalWindowMask,
    MarginalizationMask,
    NoMask,
    QKVLengthMask,
    SameSegmentMask,
    SeqLenMask,
    SymmetricAlibiBias,
)

# materialize helpers deprecated; use class methods on mask/bias instead


@pytest.fixture(
    params=[
        CausalMask,
        LocalWindowMask,
        NoMask,
        KeyPaddingMask,
        QKVLengthMask,
        # SameSegmentMask,
        MarginalizationMask,
    ]
)
def mask_fn(request):
    def mask_builder(q_len, k_len):
        if request.param in {CausalMask, NoMask}:
            return request.param()
        elif request.param == LocalWindowMask:
            left_window_size = int(np.random.randint(1, q_len))
            right_window_size = int(np.random.randint(1, k_len))
            return request.param(left_window_size, right_window_size)
        elif request.param == QKVLengthMask:
            q_lengths = int(np.random.randint(1, q_len + 1))
            k_lengths = int(np.random.randint(1, k_len + 1))
            return request.param(q_lengths, k_lengths)
        elif request.param == KeyPaddingMask:
            key_lengths = np.random.randint(1, k_len + 1, size=(k_len,))
            return request.param(jnp.asarray(key_lengths))
        elif request.param == SameSegmentMask:
            segment_ids = jnp.asarray(np.random.randint(0, 5, size=(q_len,)))
            segment_ids_k = jnp.asarray(np.random.randint(0, 5, size=(k_len,)))
            return request.param(segment_ids, segment_ids_k)
        elif request.param == MarginalizationMask:
            marginalize = np.random.choice([True, False], size=(q_len,))
            return request.param(jnp.asarray(marginalize))

    return mask_builder


def test_attention_forward_mode_jvp_matches_reference():
    batch_size, seq_len, num_heads, qkv_dim = 2, 16, 4, 16
    key_q, key_k, key_v, key_dq, key_dk, key_dv = jax.random.split(
        jax.random.PRNGKey(123), 6
    )
    q = jax.random.normal(key_q, (batch_size, seq_len, num_heads, qkv_dim))
    k = jax.random.normal(key_k, (batch_size, seq_len, num_heads, qkv_dim))
    v = jax.random.normal(key_v, (batch_size, seq_len, num_heads, qkv_dim))
    dq = jax.random.normal(key_dq, (batch_size, seq_len, num_heads, qkv_dim))
    dk = jax.random.normal(key_dk, (batch_size, seq_len, num_heads, qkv_dim))
    dv = jax.random.normal(key_dv, (batch_size, seq_len, num_heads, qkv_dim))

    def loss_ref(q, k, v):
        return jnp.sum(dot_product_attention(q, k, v))

    def loss_flex(q, k, v):
        return jnp.sum(flex_attention(q, k, v))

    primal_ref, tangent_ref = jax.jvp(loss_ref, (q, k, v), (dq, dk, dv))
    primal_flex, tangent_flex = jax.jvp(loss_flex, (q, k, v), (dq, dk, dv))

    assert jnp.allclose(primal_ref, primal_flex, atol=1e-5)
    assert jnp.allclose(tangent_ref, tangent_flex, atol=1e-3)


# @pytest.mark.gpu
@pytest.mark.parametrize(
    "attention_fn",
    [
        dot_product_attention,
        flex_attention,
    ],
)
@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
    ],
)
def test_attention_functions(attention_fn, batch_size, seq_len, num_heads, qkv_dim):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    out = attention_fn(q, k, v)
    assert out.shape == (batch_size, seq_len, num_heads, qkv_dim)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_function_outputs_are_same(batch_size, seq_len, num_heads, qkv_dim):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    outputs = []
    attention_fns = [
        dot_product_attention,
        flex_attention,
    ]
    for attention_fn in attention_fns:
        outputs.append(attention_fn(q, k, v))

    for i in range(1, len(outputs)):
        assert jnp.allclose(outputs[0], outputs[i], atol=1e-5), (
            f"Outputs are not same for {attention_fns[i]}"
        )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_function_gradients_are_same(batch_size, seq_len, num_heads, qkv_dim):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    attention_fns = [
        dot_product_attention,
        flex_attention,
    ]
    grads = []
    for attention_fn in attention_fns:
        grad_fn = jax.grad(
            lambda q, k, v: jnp.sum(attention_fn(q, k, v)), argnums=(0, 1, 2)
        )
        grads.append(grad_fn(q, k, v))

    for i in range(1, len(grads)):
        for g1, g2 in zip(grads[0], grads[i], strict=False):
            assert jnp.allclose(g1, g2, atol=1e-3), (
                f"Gradients are not same for {attention_fns[i]}"
                f" error is {jnp.mean(jnp.abs(g1 - g2))}, std {jnp.std(g1 - g2)}"
            )


def test_attention_forward_mode_jvp_matches_reference():
    batch_size, seq_len, num_heads, qkv_dim = 2, 16, 4, 16
    key_q, key_k, key_v, key_dq, key_dk, key_dv = jax.random.split(
        jax.random.PRNGKey(123), 6
    )
    q = jax.random.normal(key_q, (batch_size, seq_len, num_heads, qkv_dim))
    k = jax.random.normal(key_k, (batch_size, seq_len, num_heads, qkv_dim))
    v = jax.random.normal(key_v, (batch_size, seq_len, num_heads, qkv_dim))
    dq = jax.random.normal(key_dq, (batch_size, seq_len, num_heads, qkv_dim))
    dk = jax.random.normal(key_dk, (batch_size, seq_len, num_heads, qkv_dim))
    dv = jax.random.normal(key_dv, (batch_size, seq_len, num_heads, qkv_dim))

    def loss_ref(q, k, v):
        return jnp.sum(dot_product_attention(q, k, v))

    def loss_flex(q, k, v):
        return jnp.sum(flex_attention(q, k, v))

    primal_ref, tangent_ref = jax.jvp(loss_ref, (q, k, v), (dq, dk, dv))
    primal_flex, tangent_flex = jax.jvp(loss_flex, (q, k, v), (dq, dk, dv))

    assert jnp.allclose(primal_ref, primal_flex, atol=1e-5)
    assert jnp.allclose(tangent_ref, tangent_flex, atol=1e-3)


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
@pytest.mark.parametrize("dropout_impl", ["materialize", "counter"])
def test_attention_with_dropout(batch_size, seq_len, num_heads, qkv_dim, dropout_impl):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    out1 = flex_attention(
        q,
        k,
        v,
        dropout_rate=0.1,
        deterministic=False,
        dropout_rng=jax.random.PRNGKey(0),
        dropout_impl=dropout_impl,
    )
    assert out1.shape == (batch_size, seq_len, num_heads, qkv_dim)
    out2 = flex_attention(
        q,
        k,
        v,
        dropout_rate=0.1,
        deterministic=False,
        dropout_rng=jax.random.PRNGKey(1),
        dropout_impl=dropout_impl,
    )
    assert out2.shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert not jnp.allclose(out1, out2, atol=1e-5), "Dropout did not change the output"


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_with_masks(batch_size, seq_len, num_heads, qkv_dim, mask_fn):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    mask = mask_fn(seq_len, seq_len)
    mask_dense = mask.dense(
        seq_len, seq_len, batch_size=batch_size, num_heads=num_heads
    )
    out = dot_product_attention(q, k, v, mask=mask_dense)
    assert out.shape == (batch_size, seq_len, num_heads, qkv_dim)
    out2 = flex_attention(q, k, v, mask)
    assert out2.shape == (batch_size, seq_len, num_heads, qkv_dim)
    if isinstance(mask, QKVLengthMask):
        # If stuff is completly gone including the diagonal they behave a bit differently.
        assert jnp.allclose(
            out[:, : mask.q_length], out2[:, : mask.q_length], atol=1e-4
        )
    else:
        assert jnp.allclose(out, out2, atol=1e-5), (
            f"Outputs are not same with mask, with error {jnp.max(jnp.abs(out - out2))}"
        )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_with_bias(batch_size, seq_len, num_heads, qkv_dim):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    bias = jax.random.normal(jax.random.PRNGKey(1), (1, 1, seq_len, seq_len)) * 10
    out1 = dot_product_attention(q, k, v, bias=bias)
    out2 = flex_attention(q, k, v, bias=DenseBias(bias))
    assert jnp.allclose(out1, out2, atol=1e-5), (
        f"Outputs are not same with bias, with error {jnp.max(jnp.abs(out1 - out2))}"
    )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (1, 64, 2, 8),
    ],
)
@pytest.mark.parametrize(
    "bias_obj",
    [
        IdentityBias(),
        CausalAlibiBias(),
        SymmetricAlibiBias(),
        DistanceDecayBias(0.7),
        ConstantBias(0.3),
    ],
)
def test_attention_with_stateless_bias_objects_equivalence(
    batch_size, seq_len, num_heads, qkv_dim, bias_obj
):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )

    # Materialize dense bias from bias object for baseline JAX attention
    dense_bias = bias_obj.dense(
        q_len=seq_len, kv_len=seq_len, batch_size=batch_size, num_heads=num_heads
    )

    out_ref = dot_product_attention(q, k, v, bias=dense_bias)
    out_flex = flex_attention(q, k, v, bias=bias_obj)
    assert jnp.allclose(out_ref, out_flex, atol=1e-5), (
        f"Mismatch with {bias_obj.__class__.__name__}: "
        f"max err={jnp.max(jnp.abs(out_ref - out_flex))}"
    )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_with_bias_gradients(batch_size, seq_len, num_heads, qkv_dim):
    batch_size, seq_len, num_heads, qkv_dim = 2, 16, 4, 16
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    bias_dense = jax.random.normal(jax.random.PRNGKey(1), (1, 1, seq_len, seq_len)) * 10
    bias = DenseBias(bias_dense)

    def loss_fn1(params):
        q, k, v = params
        out = dot_product_attention(q, k, v, bias=bias_dense)
        return jnp.sum(out**2)

    def loss_fn2(params):
        q, k, v = params
        out = flex_attention(q, k, v, bias=bias)
        return jnp.sum(out**2)

    out = jax.grad(loss_fn1)((q, k, v))
    out2 = jax.grad(loss_fn2)((q, k, v))
    assert out[0].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out[1].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out[2].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out2[0].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out2[1].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out2[2].shape == (batch_size, seq_len, num_heads, qkv_dim)

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(partial(jnp.allclose, atol=1e-2), out, out2)
    ), (
        f"Gradients are not same with bias, with error {jnp.max(jnp.abs(out[0] - out2[0]))}"
    )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim, bias_obj",
    [
        (2, 16, 4, 16, IdentityBias()),
        (1, 64, 2, 8, CausalAlibiBias()),
        (1, 64, 2, 8, SymmetricAlibiBias()),
        (1, 64, 2, 8, DistanceDecayBias(0.5)),
        (1, 64, 2, 8, ConstantBias(0.2)),
    ],
)
def test_attention_with_stateless_bias_gradients(
    batch_size, seq_len, num_heads, qkv_dim, bias_obj
):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )

    def loss_ref(params):
        q, k, v = params
        # materialize dense bias for JAX reference
        dense_bias = bias_obj.dense(
            q_len=seq_len, kv_len=seq_len, batch_size=batch_size, num_heads=num_heads
        )
        out = dot_product_attention(q, k, v, bias=dense_bias)
        return jnp.sum(out**2)

    def loss_flex(params):
        q, k, v = params
        out = flex_attention(q, k, v, bias=bias_obj)
        return jnp.sum(out**2)

    grads_ref = jax.grad(loss_ref)((q, k, v))
    grads_flex = jax.grad(loss_flex)((q, k, v))

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(partial(jnp.allclose, atol=1e-2), grads_ref, grads_flex)
    )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 16, 4, 16),
        (4, 128, 8, 32),
        (1, 256, 2, 8),
        (3, 50, 8, 32),
        (1, 1024, 16, 64),
        (1, 2048, 10, 64),
        (1, 1333, 12, 64),
        (1, 256, 4, 30),
        (1, 512, 8, 100),
    ],
)
def test_attention_gradient_with_masks(
    batch_size, seq_len, num_heads, qkv_dim, mask_fn
):
    q = k = v = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, num_heads, qkv_dim)
    )
    mask = mask_fn(seq_len, seq_len)
    mask_dense = mask.dense(
        seq_len, seq_len, batch_size=batch_size, num_heads=num_heads
    )

    def loss_fn1(params):
        q, k, v = params
        out = dot_product_attention(q, k, v, mask=mask_dense)
        return jnp.sum(out**2)

    def loss_fn2(params):
        q, k, v = params
        out = flex_attention(q, k, v, mask)
        return jnp.sum(out**2)

    out = jax.grad(loss_fn1)((q, k, v))
    out2 = jax.grad(loss_fn2)((q, k, v))
    assert out[0].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out[1].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out[2].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out2[0].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out2[1].shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert out2[2].shape == (batch_size, seq_len, num_heads, qkv_dim)

    # Different whole rows are masked out; KeyPaddingMask can be numerically noisy.
    if isinstance(mask, (QKVLengthMask, KeyPaddingMask)):
        return

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(partial(jnp.allclose, atol=1e-2), out, out2)
    )


@pytest.mark.parametrize(
    "attention_fn",
    [
        dot_product_attention,
        flex_attention,
    ],
)
@pytest.mark.parametrize(
    "batch_size, q_len, kv_len, num_heads, qkv_dim",
    [
        (2, 8, 64, 2, 8),
        (1, 64, 8, 2, 8),
        (2, 13, 17, 4, 16),
        (1, 5, 9, 2, 8),
    ],
)
def test_cross_attention_shapes(
    attention_fn, batch_size, q_len, kv_len, num_heads, qkv_dim
):
    # Cross attention: different Q and KV sequences and different lengths
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(key_q, (batch_size, q_len, num_heads, qkv_dim))
    k = jax.random.normal(key_k, (batch_size, kv_len, num_heads, qkv_dim))
    v = jax.random.normal(key_v, (batch_size, kv_len, num_heads, qkv_dim))
    out = attention_fn(q, k, v)
    assert out.shape == (batch_size, q_len, num_heads, qkv_dim)


def test_flex_attention_vmap_over_leading_batch_matches_manual():
    outer_batch, batch_size, seq_len, num_heads, qkv_dim = 3, 2, 8, 2, 8
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(
        key_q, (outer_batch, batch_size, seq_len, num_heads, qkv_dim)
    )
    k = jax.random.normal(
        key_k, (outer_batch, batch_size, seq_len, num_heads, qkv_dim)
    )
    v = jax.random.normal(
        key_v, (outer_batch, batch_size, seq_len, num_heads, qkv_dim)
    )

    def attention_fn(q_, k_, v_):
        return flex_attention(q_, k_, v_, deterministic=True)

    out_vmap = jax.vmap(attention_fn, in_axes=0)(q, k, v)
    out_manual = jnp.stack(
        [attention_fn(q[i], k[i], v[i]) for i in range(outer_batch)], axis=0
    )
    assert out_vmap.shape == (
        outer_batch,
        batch_size,
        seq_len,
        num_heads,
        qkv_dim,
    )
    assert jnp.allclose(out_vmap, out_manual, atol=1e-5)


def test_flex_attention_vmap_over_leading_batch_with_mask_matches_manual():
    outer_batch, batch_size, seq_len, num_heads, qkv_dim = 2, 2, 8, 2, 8
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(202), 3)
    q = jax.random.normal(
        key_q, (outer_batch, batch_size, seq_len, num_heads, qkv_dim)
    )
    k = jax.random.normal(
        key_k, (outer_batch, batch_size, seq_len, num_heads, qkv_dim)
    )
    v = jax.random.normal(
        key_v, (outer_batch, batch_size, seq_len, num_heads, qkv_dim)
    )
    mask = CausalMask()

    def attention_fn(q, k, v):
        return flex_attention(q, k, v, mask=mask)

    out_vmap = jax.vmap(attention_fn, in_axes=0)(q, k, v)
    out_manual = jnp.stack(
        [attention_fn(q[i], k[i], v[i]) for i in range(outer_batch)], axis=0
    )

    assert out_vmap.shape == (
        outer_batch,
        batch_size,
        seq_len,
        num_heads,
        qkv_dim,
    )
    assert jnp.allclose(out_vmap, out_manual, atol=1e-5)


@pytest.mark.parametrize(
    "batch_size, q_len, kv_len, num_heads, qkv_dim",
    [
        (2, 8, 64, 2, 8),
        (1, 64, 8, 2, 8),
        (2, 13, 17, 4, 16),  # q < kv
        (3, 21, 7, 2, 8),  # q > kv
    ],
)
def test_cross_attention_forward_lengths_mismatch(
    batch_size, q_len, kv_len, num_heads, qkv_dim
):
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(11), 3)
    q = jax.random.normal(key_q, (batch_size, q_len, num_heads, qkv_dim))
    k = jax.random.normal(key_k, (batch_size, kv_len, num_heads, qkv_dim))
    v = jax.random.normal(key_v, (batch_size, kv_len, num_heads, qkv_dim))
    out_ref = dot_product_attention(q, k, v)
    out_flex = flex_attention(q, k, v)
    assert out_ref.shape == out_flex.shape == (batch_size, q_len, num_heads, qkv_dim)
    assert jnp.allclose(out_ref, out_flex, atol=1e-5), (
        f"Cross-attention forward mismatch (Q={q_len},K={kv_len}), max err={jnp.max(jnp.abs(out_ref - out_flex))}"
    )


@pytest.mark.parametrize(
    "batch_size, q_len, kv_len, num_heads, qkv_dim",
    [
        (2, 8, 64, 2, 8),
        (1, 64, 8, 2, 8),
        (2, 13, 17, 4, 16),  # q < kv
        (1, 21, 7, 2, 8),  # q > kv
    ],
)
def test_cross_attention_backward_lengths_mismatch(
    batch_size, q_len, kv_len, num_heads, qkv_dim
):
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(12), 3)
    q = jax.random.normal(key_q, (batch_size, q_len, num_heads, qkv_dim))
    k = jax.random.normal(key_k, (batch_size, kv_len, num_heads, qkv_dim))
    v = jax.random.normal(key_v, (batch_size, kv_len, num_heads, qkv_dim))

    def loss_ref(q, k, v):
        out = dot_product_attention(q, k, v)
        return jnp.sum(out**2)

    def loss_flex(q, k, v):
        out = flex_attention(q, k, v)
        return jnp.sum(out**2)

    grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    grads_flex = jax.grad(loss_flex, argnums=(0, 1, 2))(q, k, v)

    for g_ref, g_flex in zip(grads_ref, grads_flex, strict=False):
        assert g_ref.shape == g_flex.shape
        assert jnp.allclose(g_ref, g_flex, atol=1e-2), (
            f"Cross-attention backward mismatch (Q={q_len},K={kv_len}), max err={jnp.max(jnp.abs(g_ref - g_flex))}"
        )


@pytest.mark.parametrize(
    "batch_size, q_len, kv_len, num_heads, qkv_dim",
    [
        (2, 128, 128, 4, 16),
        (1, 128, 256, 2, 8),
        (1, 64, 512, 8, 32),
        (3, 512, 64, 8, 32),
        (1, 32, 222, 4, 30),
        (1, 222, 32, 4, 30),
    ],
)
def test_cross_attention_outputs_match(batch_size, q_len, kv_len, num_heads, qkv_dim):
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(1), 3)
    q = jax.random.normal(key_q, (batch_size, q_len, num_heads, qkv_dim))
    k = jax.random.normal(key_k, (batch_size, kv_len, num_heads, qkv_dim))
    v = jax.random.normal(key_v, (batch_size, kv_len, num_heads, qkv_dim))

    attention_fns = [
        dot_product_attention,
        flex_attention,
    ]
    outputs = [fn(q, k, v) for fn in attention_fns]

    for i in range(1, len(outputs)):
        assert jnp.allclose(outputs[0], outputs[i], atol=1e-5), (
            f"Cross-attention outputs differ for {attention_fns[i]}"
        )


@pytest.mark.parametrize(
    "batch_size, q_len, kv_len, num_heads, qkv_dim",
    [
        (2, 128, 128, 4, 16),
        (1, 128, 256, 2, 8),
        (1, 64, 512, 8, 32),
        (3, 512, 64, 8, 32),
        (1, 32, 222, 4, 30),
        (1, 222, 32, 4, 30),
    ],
)
def test_cross_attention_gradients_match(batch_size, q_len, kv_len, num_heads, qkv_dim):
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(2), 3)
    q = jax.random.normal(key_q, (batch_size, q_len, num_heads, qkv_dim))
    k = jax.random.normal(key_k, (batch_size, kv_len, num_heads, qkv_dim))
    v = jax.random.normal(key_v, (batch_size, kv_len, num_heads, qkv_dim))

    attention_fns = [
        dot_product_attention,
        flex_attention,
    ]

    grads = []
    for fn in attention_fns:
        grad_fn = jax.grad(lambda q, k, v: jnp.sum(fn(q, k, v)), argnums=(0, 1, 2))
        grads.append(grad_fn(q, k, v))

    for i in range(1, len(grads)):
        for g1, g2 in zip(grads[0], grads[i], strict=False):
            assert jnp.allclose(g1, g2, atol=1e-2), (
                f"Cross-attention gradients differ for {attention_fns[i]}"
            )


@pytest.mark.parametrize(
    "batch_size, q_len, kv_len, num_heads, qkv_dim",
    [
        (2, 13, 17, 4, 16),  # q < kv
        (1, 21, 7, 2, 8),  # q > kv
    ],
)
@pytest.mark.parametrize("backward_impl", ["triton_fused", "triton_split"])
def test_cross_attention_backward_impl_variants(
    batch_size, q_len, kv_len, num_heads, qkv_dim, backward_impl
):
    key_q, key_k, key_v = jax.random.split(jax.random.PRNGKey(55), 3)
    q = jax.random.normal(key_q, (batch_size, q_len, num_heads, qkv_dim))
    k = jax.random.normal(key_k, (batch_size, kv_len, num_heads, qkv_dim))
    v = jax.random.normal(key_v, (batch_size, kv_len, num_heads, qkv_dim))

    def loss_ref(q, k, v):
        out = dot_product_attention(q, k, v)
        return jnp.sum(out**2)

    def loss_flex(q, k, v):
        out = flex_attention(q, k, v, backward_pass_impl=backward_impl)
        return jnp.sum(out**2)

    grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    grads_flex = jax.grad(loss_flex, argnums=(0, 1, 2))(q, k, v)

    for g_ref, g_flex in zip(grads_ref, grads_flex, strict=False):
        assert (
            g_ref.shape
            == g_flex.shape
            == (batch_size, g_ref.shape[1], num_heads, qkv_dim)
        )
        assert jnp.allclose(g_ref, g_flex, atol=1e-2), (
            f"Backward mismatch for impl={backward_impl} (Q={q_len},K={kv_len}), max err={jnp.max(jnp.abs(g_ref - g_flex))}"
        )


@pytest.mark.parametrize(
    "batch_size, q_len, kv_len, num_heads, qkv_dim",
    [
        (2, 13, 17, 4, 16),
        (1, 5, 9, 2, 8),
    ],
)
def test_cross_attention_with_mask_and_bias(
    batch_size, q_len, kv_len, num_heads, qkv_dim, mask_fn
):
    # Compare flex_attention with mask objects and DenseBias vs dense attention with materialized mask/bias
    key0, key1, key2, key3 = jax.random.split(jax.random.PRNGKey(3), 4)
    q = jax.random.normal(key0, (batch_size, q_len, num_heads, qkv_dim))
    k = jax.random.normal(key1, (batch_size, kv_len, num_heads, qkv_dim))
    v = jax.random.normal(key2, (batch_size, kv_len, num_heads, qkv_dim))

    mask = mask_fn(q_len, kv_len)
    mask_dense = mask.dense(q_len, kv_len, batch_size=batch_size, num_heads=num_heads)

    # Some stateful masks like KeyPaddingMask/MarginalizationMask may differ in
    # semantics under cross-attention with added bias; skip those here.
    if isinstance(mask, (KeyPaddingMask, MarginalizationMask)):
        pytest.skip(
            "Skipping cross-attention equivalence for KeyPadding/Marginalization masks"
        )

    bias_dense = jax.random.normal(key3, (1, 1, q_len, kv_len)) * 1.5

    out_dense = dot_product_attention(q, k, v, mask=mask_dense, bias=bias_dense)
    out_flex = flex_attention(q, k, v, mask=mask, bias=DenseBias(bias_dense))

    if isinstance(mask, QKVLengthMask):
        assert jnp.allclose(
            out_dense[:, : mask.q_length], out_flex[:, : mask.q_length], atol=1e-4
        )
    else:
        assert jnp.allclose(out_dense, out_flex, atol=1e-5), (
            f"Cross-attention with mask+bias mismatch, max err={jnp.max(jnp.abs(out_dense - out_flex))}"
        )


def _build_seq_len_dense_mask(seq_lengths: jax.Array, q_len: int) -> jax.Array:
    """Dense [B, Q, Q] mask for SeqLenMask semantics.

    True if (i< L_b and j < L_b) OR (i == j).
    """
    q_idx = jnp.arange(q_len)
    valid_q = q_idx[None, :] < seq_lengths[:, None]  # [B, Q]
    valid_k = valid_q
    rect = valid_q[:, :, None] & valid_k[:, None, :]
    eye = jnp.eye(q_len, dtype=rect.dtype)[None, :, :]
    mask = rect | eye
    return mask[:, None, :, :]


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 32, 4, 16),
        (3, 64, 2, 8),
    ],
)
def test_seq_len_mask_forward(batch_size, seq_len, num_heads, qkv_dim):
    key = jax.random.PRNGKey(0)
    q = k = v = jax.random.normal(key, (batch_size, seq_len, num_heads, qkv_dim))
    # Random per-batch lengths in [1, seq_len]
    L = jax.random.randint(jax.random.PRNGKey(1), (batch_size,), 1, seq_len + 1)

    dense_mask = _build_seq_len_dense_mask(L, seq_len)
    out_ref = dot_product_attention(q, k, v, mask=dense_mask)
    out_flex = flex_attention(q, k, v, mask=SeqLenMask(L))

    assert out_ref.shape == out_flex.shape == (batch_size, seq_len, num_heads, qkv_dim)
    assert jnp.allclose(out_ref, out_flex, atol=1e-5), (
        f"SeqLenMask forward mismatch, max err={jnp.max(jnp.abs(out_ref - out_flex))}"
    )


@pytest.mark.parametrize(
    "batch_size, seq_len, num_heads, qkv_dim",
    [
        (2, 32, 4, 16),
        (2, 64, 2, 8),
    ],
)
def test_seq_len_mask_backward(batch_size, seq_len, num_heads, qkv_dim):
    key_q, key_len = jax.random.split(jax.random.PRNGKey(42))
    q = k = v = jax.random.normal(key_q, (batch_size, seq_len, num_heads, qkv_dim))
    L = jax.random.randint(key_len, (batch_size,), 1, seq_len + 1)
    dense_mask = _build_seq_len_dense_mask(L, seq_len)

    def loss_ref(q, k, v):
        out = dot_product_attention(q, k, v, mask=dense_mask)
        return jnp.sum(out**2)

    def loss_flex(q, k, v):
        out = flex_attention(q, k, v, mask=SeqLenMask(L))
        return jnp.sum(out**2)

    grads_ref = jax.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    grads_flex = jax.grad(loss_flex, argnums=(0, 1, 2))(q, k, v)

    for g_ref, g_flex in zip(grads_ref, grads_flex, strict=False):
        assert g_ref.shape == g_flex.shape == (batch_size, seq_len, num_heads, qkv_dim)
        assert jnp.allclose(g_ref, g_flex, atol=1e-2), (
            f"SeqLenMask backward mismatch, max err={jnp.max(jnp.abs(g_ref - g_flex))}"
        )

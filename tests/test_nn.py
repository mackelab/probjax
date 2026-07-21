import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.core import inverse, inverse_and_logabsdet
from probjax.nn.layers.attention import (
    PerHeadQueryScale,
    QASSMaxQueryScale,
    dot_product_attention,
)
from probjax.nn import (
    AdditiveCouplingFlow,
    AdditiveBinaryFuse,
    AffineAutoregressiveFlow,
    AutoregressiveTransformer,
    CouplingMLP,
    CouplingTransformer,
    DropPath,
    GatedFuse,
    InducedSelfAttention,
    MaskedLinear,
    MLP,
    Transformer,
    chunkify,
)

pytest_plugins = ["test_problems.nns"]


def test_mlp(mlp, batch_shape):
    in_dim, out_dim, model = mlp
    x = jnp.ones(batch_shape + (in_dim,))
    y = model(x)
    assert y.shape == batch_shape + (out_dim,)

    def loss_fn(model):
        return jnp.sum(model(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)


def test_resnet(resnet, batch_shape):
    in_dim, out_dim, model = resnet
    x = jnp.ones(batch_shape + (in_dim,))
    y = model(x)
    assert y.shape == batch_shape + (out_dim,)

    def loss_fn(model):
        return jnp.sum(model(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)


def test_deepset(deepset, seq_len, batch_shape):
    in_dim, out_dim, model = deepset
    x = jnp.ones(batch_shape + (seq_len, in_dim))
    y = model(x)
    assert y.shape == batch_shape + (out_dim,)

    def loss_fn(model):
        return jnp.sum(model(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)


def test_attention(multi_head_attention, seq_len, batch_shape):
    in_dim, out_dim, model = multi_head_attention
    x = jnp.ones(batch_shape + (seq_len, in_dim))
    y = model(x, x, x)
    assert y.shape == batch_shape + (seq_len, out_dim)

    def loss_fn(model):
        return jnp.sum(model(x, x, x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)


def test_induced_self_attention():
    model = InducedSelfAttention(
        16,
        num_inducing_points=4,
        num_heads=4,
        attn_size=4,
        q_scale_cls=PerHeadQueryScale,
        output_q_scale_cls=PerHeadQueryScale,
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 7, 16))

    y = model(x)
    y_kv = model(x, kv_len=5)
    y_kv_arr = model(x, kv_len=jnp.array([5, 3]))

    assert y.shape == x.shape
    assert y_kv.shape == x.shape
    assert y_kv_arr.shape == x.shape

    def loss_fn(m):
        return jnp.sum(m(x, kv_len=5))

    _ = jax.grad(loss_fn)
    _, _ = jax.tree_util.tree_flatten(model)


def test_qassmax_query_scale_checkpointing_matches_eager():
    rngs = nnx.Rngs(0)
    eager = QASSMaxQueryScale(
        num_heads=2,
        head_dim=4,
        hidden_dim=8,
        use_checkpointing=False,
        rngs=rngs,
    )
    remat = QASSMaxQueryScale(
        num_heads=2,
        head_dim=4,
        hidden_dim=8,
        use_checkpointing=True,
        rngs=nnx.Rngs(0),
    )
    query = jnp.arange(2 * 5 * 2 * 4, dtype=jnp.float32).reshape(2, 5, 2, 4)
    kv_len = jnp.array([5, 3], dtype=jnp.int32)

    eager_out = eager(query, kv_len=kv_len)
    remat_out = remat(query, kv_len=kv_len)
    np.testing.assert_allclose(np.asarray(remat_out), np.asarray(eager_out), rtol=1e-5)

    def eager_loss(q):
        return jnp.sum(eager(q, kv_len=kv_len))

    def remat_loss(q):
        return jnp.sum(remat(q, kv_len=kv_len))

    np.testing.assert_allclose(
        np.asarray(jax.grad(remat_loss)(query)),
        np.asarray(jax.grad(eager_loss)(query)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_coupling(coupling_mlp, batch_shape):
    in_dim, out_dim, model = coupling_mlp
    x = jnp.ones(batch_shape + (in_dim,))
    y = model(x)
    assert y.shape == batch_shape + (out_dim,)

    def loss_fn(model):
        return jnp.sum(model(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)

    # Test inverse
    model_inv = inverse(model)
    y_inv = model_inv(y)
    assert jnp.allclose(x, y_inv), "Inverse is not correct"


def test_coupling_mlp_custom_split_merge():
    def add_bijector(params, x):
        return x + params

    def split_even_odd(x):
        return x[..., ::2], x[..., 1::2]

    def merge_even_odd(y1, y2):
        y = jnp.stack([y1, y2], axis=-1)
        return y.reshape(y.shape[:-2] + (y.shape[-2] * y.shape[-1],))

    model = CouplingMLP(
        split_index=2,
        bij_params_dim=2,
        bijector=add_bijector,
        split_fn=split_even_odd,
        merge_fn=merge_even_odd,
        hidden_dims=[16, 16],
        rngs=nnx.Rngs(0),
    )

    x = jnp.arange(8.0).reshape(2, 4)
    y = model(x)
    assert y.shape == x.shape


def test_coupling_transformer_with_context():
    def scale_bijector(params, x):
        return x * jnp.exp(params)

    model = CouplingTransformer(
        split_index=2,
        bij_params_dim=2,
        bijector=scale_bijector,
        context_features=3,
        model_dim=16,
        num_heads=2,
        num_layers=1,
        attn_size=8,
        rngs=nnx.Rngs(0),
    )

    x = jnp.ones((5, 4))
    context = jnp.ones((5, 3))
    y = model(x, context=context)
    assert y.shape == x.shape

    def loss_fn(m):
        return jnp.sum(m(x, context=context))

    _ = jax.grad(loss_fn)(model)


def test_autoregressive(autoregressive_mlp, batch_shape):
    in_dim, out_dim, model = autoregressive_mlp
    x = jnp.ones(batch_shape + (in_dim,))
    y = model(x)
    assert y.shape == batch_shape + (out_dim,)

    def loss_fn(model):
        return jnp.sum(model(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)

    # Test inverse
    model_inv = inverse(model)
    y_inv = model_inv(y)
    assert jnp.allclose(x, y_inv), " Inverse is not correct"


def test_autoregressive_transformer_kv_cache_matches_naive():
    def additive_bijector(params, value):
        return value + params[..., None, :]

    transformer = Transformer(
        model_dim=16,
        num_heads=2,
        num_layers=2,
        attn_size=8,
        attention_fn=dot_product_attention,
        rngs=nnx.Rngs(0),
    )
    model = AutoregressiveTransformer(
        in_out_dim=1,
        bijector_dim=1,
        bijector=additive_bijector,
        transformer=transformer,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(1), (2, 8, 1))
    mask = jnp.tril(jnp.ones((x.shape[-2] + 1, x.shape[-2] + 1), dtype=bool))

    y_naive = x
    for i in range(x.shape[-2]):
        bij_params = model.predict_bij_params(y_naive, mask=mask)
        bij_params_i = bij_params[..., i, :]
        x_i = y_naive[..., i : i + 1, :]
        x_new_i = model.bijector(bij_params_i, x_i)
        y_naive = y_naive.at[..., i, :].set(x_new_i[..., 0, :])

    y_kv_cache = model.forward(x, inverse_impl="kv_cache")

    assert y_naive.shape == y_kv_cache.shape
    assert jnp.allclose(y_naive, y_kv_cache, atol=1e-6, rtol=1e-6)


def test_gaussian_fourier_embedding(gaussian_fourier_embedding, batch_shape):
    in_dim, out_dim, model = gaussian_fourier_embedding
    x = jnp.ones(batch_shape + (in_dim,))
    y = model(x)
    assert y.shape == batch_shape + (out_dim,)

    def loss_fn(model):
        return jnp.sum(model(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)


def test_transformer(transformer, seq_len, batch_shape):
    model_dim, model = transformer
    x = jnp.ones(batch_shape + (seq_len, model_dim))
    y = model(x)
    assert y.shape == batch_shape + (seq_len, model_dim)

    def loss_fn(model):
        return jnp.sum(model(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)


def test_transformer_with_context(transformer_with_context, seq_len, batch_shape):
    model_dim, context_dim, model = transformer_with_context
    x = jnp.ones(batch_shape + (seq_len, model_dim))
    context = jnp.ones(batch_shape + (context_dim,))
    y = model(x, context=context)
    assert y.shape == batch_shape + (seq_len, model_dim)

    def loss_fn(model):
        return jnp.sum(model(x, context=context))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)


def test_transformer_with_context_and_cross_attention(
    transformer_with_cross_attention_and_context, seq_len, batch_shape
):
    model_dim, context_dim, model = transformer_with_cross_attention_and_context
    x = jnp.ones(batch_shape + (seq_len, model_dim))
    context = jnp.ones(batch_shape + (context_dim,))
    y = model(x, x + 1, x + 1, context=context)
    assert y.shape == batch_shape + (seq_len, model_dim)

    def loss_fn(model):
        return jnp.sum(model(x, x + 1, x + 1, context=context))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)


def test_lru(lru, seq_len):
    in_dim, out_dim, model = lru
    batch_shape = ()  # Needs vmap
    x = jnp.ones(batch_shape + (seq_len, in_dim))
    y = model(x)
    print(in_dim, out_dim)
    assert y.shape == batch_shape + (seq_len, out_dim)

    def loss_fn(model):
        return jnp.sum(model(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)


def test_diffusion(denoising_diffusion):
    in_dim, model = denoising_diffusion
    batch_shape = (10,)  # Only support one batch dimension
    x = jnp.ones(batch_shape + (in_dim,))
    t = jnp.ones(batch_shape + (1,))
    y = model(t, x)
    assert y.shape == batch_shape + (in_dim,), "Denosing shape is not correct"

    s = model.score(t, x)

    assert s.shape == batch_shape + (in_dim,), "Score shape is not correct"

    loss = model.loss(jax.random.key(0), x)
    assert loss.shape == (), "Loss shape is not correct"


def test_flows(flow):
    input_dim, model = flow
    x = jnp.ones((input_dim,))
    y = model.transform(x)

    distribution = model.as_dist()
    assert distribution.batch_shape == ()
    assert distribution.event_shape == (input_dim,)

    def loss_fn(model):
        return jnp.sum(model.as_dist().logpdf(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)

    # Test inverse
    model_inv = inverse(model.transform)
    y_inv = model_inv(y)
    assert jnp.allclose(x, y_inv, atol=1e-2, rtol=1e-1), "Inverse is not correct"

    # Test inverse and logabsdet
    model_inv_logabsdet = inverse_and_logabsdet(model.transform)
    y_inv, logabsdet = model_inv_logabsdet(y)
    assert jnp.allclose(x, y_inv, atol=1e-2, rtol=1e-1), "Inverse is not correct"
    assert logabsdet.shape == ()

    # Sampling
    samples = distribution.sample(rng=jax.random.PRNGKey(0), shape=(10,))
    assert samples.shape == (10, input_dim)
    # Log probability
    logprob = distribution.logpdf(samples)
    assert logprob.shape == (10,)


@pytest.mark.parametrize(
    "flow_ctor",
    [
        lambda rngs: AdditiveCouplingFlow(4, 2, rngs=rngs, context_features=3),
        lambda rngs: AffineAutoregressiveFlow(4, 2, rngs=rngs, context_features=3),
    ],
)
def test_conditional_flows_with_context(flow_ctor):
    model = flow_ctor(nnx.Rngs(0))
    x = jnp.ones((4,))
    context = jnp.array([0.1, -0.2, 0.3])

    y = model.transform(x, context=context)
    assert y.shape == x.shape

    model_inv = inverse(lambda z: model.transform(z, context=context))
    y_inv = model_inv(y)
    assert jnp.allclose(x, y_inv, atol=1e-2, rtol=1e-1), "Inverse is not correct"

    distribution = model.as_dist(context_spec=(3,))
    samples = distribution.sample(
        rng=jax.random.PRNGKey(0), shape=(8,), context=context
    )
    assert samples.shape == (8, 4)

    logprob = distribution.logpdf(samples, context=context)
    assert logprob.shape == (8,)

    context_2 = jnp.array([-0.5, 0.7, -0.9])
    logprob_2 = distribution.logpdf(samples, context=context_2)
    assert logprob_2.shape == (8,)


def test_named_flow_aliases_construct():
    from probjax.nn import NeuralSplineFlow, maf, nice, nsf, realnvp

    aliases = [nice, realnvp, maf, nsf, NeuralSplineFlow]
    for ctor in aliases:
        model = ctor(2, 1, rngs=nnx.Rngs(0))
        x = jnp.ones((2,))
        y = model.transform(x)
        assert y.shape == x.shape


def test_public_import_surface():
    import importlib

    modules = [
        "probjax.nn",
        "probjax.nn.losses",
        "probjax.nn.generative",
        "probjax.nn.generative.flows",
        "probjax.nn.generative.diffusion",
        "probjax.nn.generative.flow_matching",
        "probjax.nn.generative.mean_flow",
        "probjax.nn.generative.discrete",
    ]
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", []):
            assert hasattr(module, name), f"{module_name}.{name} does not resolve"


def test_chunkify(chunkify_inputs):
    x, chunk_shape, channel_axis = chunkify_inputs
    metadata = _chunkify_metadata(x, chunk_shape, channel_axis)
    expected_shape = _chunkify_expected_shape(metadata)
    tokens = chunkify(x, chunk_shape, channel_axis=channel_axis)
    assert tokens.shape == expected_shape
    reconstructed = _unchunkify(tokens, x, metadata)
    assert jnp.array_equal(reconstructed, x)


def test_masked_linear_forward(masked_linear_case):
    mask, kernel, bias, x, expected = masked_linear_case
    layer = MaskedLinear(2, 2, mask, rngs=nnx.Rngs(0))
    layer.kernel[...] = kernel
    layer.bias[...] = bias
    y = layer(x)
    assert jnp.allclose(y, expected)


def test_drop_path(drop_path_case):
    drop_rate, deterministic, x, expected = drop_path_case
    drop = DropPath(drop_rate=drop_rate, rngs=nnx.Rngs(0))
    y = drop(x, deterministic=deterministic)
    assert jnp.allclose(y, expected)


def test_additive_binary_fuse(additive_binary_fuse_case):
    x, y = additive_binary_fuse_case
    fuse = AdditiveBinaryFuse(x.shape[-1], context_features=None, rngs=nnx.Rngs(0))
    out = fuse(x, y, None)
    assert jnp.allclose(out, x + y)


def test_gated_fuse_modes(gated_fuse_case):
    mode, x, y, context = gated_fuse_case
    fuse = GatedFuse(4, 3, mode=mode, rngs=nnx.Rngs(0))
    out = fuse(x, y, context)
    assert out.shape == x.shape


def test_gated_fuse_invalid_mode():
    with pytest.raises(ValueError):
        GatedFuse(4, 3, mode="invalid", rngs=nnx.Rngs(0))


def _chunkify_metadata(x, chunk_shape, channel_axis):
    x_arr = jnp.asarray(x)
    chunk_shape_tuple = (
        (chunk_shape,) if isinstance(chunk_shape, int) else tuple(chunk_shape)
    )
    x_work = x_arr
    inserted_channel = False
    channel_axis_mod = None

    if channel_axis is None:
        x_work = jnp.expand_dims(x_work, axis=-1)
        inserted_channel = True
    else:
        channel_axis_mod = channel_axis % x_work.ndim
        if channel_axis_mod != x_work.ndim - 1:
            x_work = jnp.moveaxis(x_work, channel_axis_mod, -1)

    spatial_ndim = len(chunk_shape_tuple)
    spatial_start = x_work.ndim - spatial_ndim - 1
    batch_shape = x_work.shape[:spatial_start]
    spatial_shape = x_work.shape[spatial_start:-1]
    channel_dim = x_work.shape[-1]
    chunk_counts = tuple(
        size // chunk
        for size, chunk in zip(spatial_shape, chunk_shape_tuple, strict=False)
    )
    chunk_volume = int(np.prod(chunk_shape_tuple)) if chunk_shape_tuple else 1
    total_chunks = int(np.prod(chunk_counts)) if chunk_counts else 1
    perm = _chunkify_perm(len(batch_shape), spatial_ndim)

    return {
        "chunk_shape": chunk_shape_tuple,
        "batch_shape": batch_shape,
        "spatial_shape": spatial_shape,
        "channel_dim": channel_dim,
        "chunk_counts": chunk_counts,
        "chunk_volume": chunk_volume,
        "total_chunks": total_chunks,
        "perm": perm,
        "channel_axis_mod": channel_axis_mod,
        "inserted_channel": inserted_channel,
    }


def _chunkify_expected_shape(metadata):
    return metadata["batch_shape"] + (
        metadata["total_chunks"],
        metadata["chunk_volume"] * metadata["channel_dim"],
    )


def _chunkify_perm(batch_ndim, spatial_ndim):
    count_axes = [batch_ndim + 2 * idx for idx in range(spatial_ndim)]
    chunk_axes = [axis + 1 for axis in count_axes]
    channel_axis = batch_ndim + 2 * spatial_ndim
    return list(range(batch_ndim)) + count_axes + chunk_axes + [channel_axis]


def _unchunkify(tokens, x, metadata):
    chunk_shape = metadata["chunk_shape"]
    chunk_counts = metadata["chunk_counts"]
    batch_shape = metadata["batch_shape"]
    channel_dim = metadata["channel_dim"]
    perm = metadata["perm"]

    reshaped = tokens.reshape(batch_shape + chunk_counts + chunk_shape + (channel_dim,))
    perm_inv = tuple(np.argsort(perm))
    transposed = reshaped.transpose(perm_inv)
    spatial_shape = tuple(
        count * size for count, size in zip(chunk_counts, chunk_shape, strict=False)
    )
    x_channel_last = transposed.reshape(batch_shape + spatial_shape + (channel_dim,))

    if metadata["inserted_channel"]:
        return jnp.squeeze(x_channel_last, axis=-1)

    channel_axis_mod = metadata["channel_axis_mod"]
    if channel_axis_mod is not None and channel_axis_mod != x.ndim - 1:
        x_channel_last = jnp.moveaxis(x_channel_last, -1, channel_axis_mod)
    return x_channel_last

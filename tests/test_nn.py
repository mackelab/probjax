import jax
import jax.numpy as jnp

from probjax.core import inverse, inverse_and_logabsdet

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
    y = model(x, context)
    assert y.shape == batch_shape + (seq_len, model_dim)

    def loss_fn(model):
        return jnp.sum(model(x, context))

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


def test_flows(flow):
    input_dim, model = flow
    x = jnp.ones((input_dim,))
    y = model.transform(x)

    def loss_fn(model):
        return jnp.sum(model.log_prob(x))

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
    samples = model.sample(jax.random.PRNGKey(0), (10,))
    assert samples.shape == (10, input_dim)
    # Log probability
    logprob = model.log_prob(samples)
    assert logprob.shape == (10,)

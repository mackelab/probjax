from flax import nnx
import jax.numpy as jnp

import jax
from probjax.core import inverse, inverse_and_logabsdet

pytest_plugins = ["test_problems.nn_params"]


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


def test_lru(lru, seq_len):
    in_dim, out_dim, model = lru
    batch_shape = ()  # Needs vmap
    x = jnp.ones(batch_shape + (seq_len, in_dim))
    y = model(x)
    assert y.shape == batch_shape + (seq_len, out_dim)

    def loss_fn(model):
        return jnp.sum(model(x))

    # Can be differentiated
    _ = jax.grad(loss_fn)

    # Can be flattened
    _, _ = jax.tree_util.tree_flatten(model)

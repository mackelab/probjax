"""The complete functional fit composes with JIT; host effects are opt-in."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from probjax.nn import FitResult, fit, fit_module


def loss(params, key, primary):
    del key
    return jnp.mean((params["w"] - primary) ** 2)


def test_complete_fit_jits_with_legacy_and_detailed_returns():
    args = ({"w": jnp.array(0.0)}, jax.random.key(0), jnp.arange(5.0))
    options = dict(num_steps=7, batch_size=3, ema_decay=0.8, use_ema=True)
    expected = fit(loss, *args, **options)
    actual = jax.jit(lambda p, k, x: fit(loss, p, k, x, **options))(*args)
    np.testing.assert_array_equal(actual.losses, expected.losses)
    np.testing.assert_array_equal(actual.params["w"], expected.params["w"])
    result = jax.jit(lambda p, k, x: fit(loss, p, k, x, **options))(*args)
    assert isinstance(result, FitResult)
    assert int(result.state.step) == 7
    assert result.losses.shape == (7,)


def test_complete_fit_resumes_under_jit():
    def objective(p, k, x):
        return loss(p, k, x)

    options = dict(num_steps=3, optimizer=optax.adam(0.01))
    first = fit(
        objective, {"w": jnp.array(0.0)}, jax.random.key(0), jnp.ones(4), **options
    )
    result = jax.jit(
        lambda state, x: fit(objective, None, None, x, state=state, **options)
    )(first.state, jnp.ones(4))
    assert int(result.state.step) == 6
    assert result.losses.shape == (3,)
    # The resumed run continued from the snapshot, so it moved further.
    assert not np.array_equal(result.state.params["w"], first.state.params["w"])


def test_callbacks_are_rejected_inside_jit():
    """Callbacks need the host, and a jitted scan has no way to reach it."""
    with pytest.raises(ValueError, match="callbacks run on the host"):
        jax.jit(
            lambda p, k, x: fit(
                loss,
                p,
                k,
                x,
                num_steps=1,
                callback=lambda s, i: None,
            )
        )(jnp.array(0.0), jax.random.key(0), jnp.ones(4))


def test_pure_fit_contains_no_host_callback_and_differentiates():
    def scalar_loss(params, key, primary):
        del key
        return jnp.mean((params - primary) ** 2)

    train = lambda p, x: (
        fit(
            scalar_loss,
            p,
            jax.random.key(0),
            x,
            num_steps=2,
            optimizer=optax.sgd(0.1),
        ).params
    )
    text = str(jax.make_jaxpr(train)(jnp.array(0.0), jnp.ones(4)))
    assert "io_callback" not in text
    assert "debug_callback" not in text

    # Two SGD steps on mean((p - 1)^2): each step scales the incoming
    # gradient by (1 - lr * 2), so the parameter's sensitivity to its initial
    # value is (1 - 0.2)^2. The loop must not hide the update rule from
    # autodiff.
    gradient = jax.jit(jax.grad(lambda p: train(p, jnp.ones(4))))(jnp.array(0.0))
    np.testing.assert_allclose(gradient, 0.8**2, rtol=1e-6)


def test_module_fit_under_nnx_jit_updates_parameters_and_mutable_state():
    from flax import nnx

    class Model(nnx.Module):
        def __init__(self):
            self.w = nnx.Param(jnp.array(0.0))
            self.count = nnx.Variable(jnp.array(0))

        def loss(self, key, primary):
            self.count[...] += 1
            return jnp.mean((self.w[...] - primary) ** 2)

    model = Model()

    @nnx.jit
    def train(model, key, batch):
        return fit_module(model, key, batch, num_steps=3)

    result = train(model, jax.random.key(0), jnp.ones(4))
    assert model.w[...] > 0
    assert model.count[...] == 3
    assert int(result.state.step) == 3


def test_flow_fit_under_nnx_jit_standardizes_only_once():
    from flax import nnx

    from probjax.nn import maf

    model = maf(2, 1, rngs=nnx.Rngs(0))

    @nnx.jit
    def train(model, key, batch):
        return model.fit(key, batch, num_steps=2)

    data = jnp.arange(16.0, dtype=jnp.float32).reshape(8, 2)
    result = train(model, jax.random.key(0), data)
    assert jnp.all(jnp.isfinite(result.losses))
    shift, scale = model.standardization
    np.testing.assert_allclose(shift, jnp.mean(data, axis=0))
    train(model, jax.random.key(1), data + 10.0)
    np.testing.assert_array_equal(model.standardization[0], shift)
    np.testing.assert_array_equal(model.standardization[1], scale)


def test_jitted_standardization_preserves_state_dtype():
    from flax import nnx

    from probjax.nn.generative.standardize import StandardizingMixin

    class Model(nnx.Module, StandardizingMixin):
        def __init__(self):
            self._init_standardization(2)

        def _clear_distribution_cache(self):
            pass

    with jax.enable_x64():
        model = Model()
        batch = jnp.arange(8.0, dtype=jnp.float32).reshape(4, 2)
        nnx.jit(lambda m, x: m.fit_standardization(x))(model, batch)
        assert model.standardization[0].dtype == jnp.float64
        np.testing.assert_allclose(model.standardization[0], jnp.mean(batch, axis=0))

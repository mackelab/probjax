"""Callbacks, averaging, option checks and module state in the training loop."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from probjax.nn import fit, fit_module


def loss(params, rng, batch):
    del rng
    return jnp.mean((params["w"] - batch) ** 2)


def test_ema_matches_manual_updates_and_keeps_raw_parameters():
    initial = {"w": jnp.array(0.0)}
    raw, ema = 0.0, 0.0
    for _ in range(5):
        raw -= 0.1 * 2 * (raw - 2.0)
        ema = 0.8 * ema + 0.2 * raw
    result = fit(
        loss,
        initial,
        jax.random.key(0),
        jnp.full((4,), 2.0),
        num_steps=5,
        optimizer=optax.sgd(0.1),
        ema_decay=0.8,
        use_ema=True,
    )
    np.testing.assert_allclose(result.state.params["w"], raw, rtol=1e-6)
    np.testing.assert_allclose(result.params["w"], ema, rtol=1e-6)
    assert int(result.state.step) == 5
    assert initial["w"] == 0


def test_callback_sees_device_snapshots_and_can_stop_early():
    events = []

    def monitor(state, info):
        assert isinstance(state.params["w"], jax.Array)
        assert jnp.isfinite(info.loss)
        events.append(int(state.step))
        return bool(state.step < 6)

    result = fit(
        loss,
        {"w": jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(8),
        num_steps=20,
        every=3,
        ema_decay=0.9,
        callback=monitor,
    )
    assert events == [3, 6]
    assert result.losses.shape == (6,)


def test_callback_chunks_preserve_randomness_and_final_partial_chunk():
    seen = []
    options = dict(num_steps=13, batch_size=3, ema_decay=0.8)
    args = (loss, {"w": jnp.array(0.0)}, jax.random.key(5), jnp.arange(9.0))
    single = fit(*args, **options)
    chunked = fit(
        *args,
        every=5,
        callback=lambda state, info: seen.append(int(state.step)),
        **options,
    )
    assert seen == [5, 10, 13]
    np.testing.assert_array_equal(single.losses, chunked.losses)
    np.testing.assert_array_equal(single.params["w"], chunked.params["w"])
    np.testing.assert_array_equal(
        single.state.ema_params["w"], chunked.state.ema_params["w"]
    )
    np.testing.assert_array_equal(
        jax.random.key_data(single.state.rng), jax.random.key_data(chunked.state.rng)
    )


def test_callback_can_validate_with_jit_and_exception_propagates():
    evaluate = jax.jit(lambda params: params["w"] ** 2)
    values = []

    def callback(state, info):
        values.append(float(evaluate(state.params)))
        raise ValueError("validation failed")

    with pytest.raises(ValueError, match="validation failed"):
        fit(
            loss,
            {"w": jnp.array(0.0)},
            jax.random.key(0),
            jnp.ones(4),
            num_steps=3,
            callback=callback,
        )
    assert len(values) == 1


def test_a_non_finite_loss_still_counts_as_an_update():
    def nonfinite(params, rng, batch):
        del rng, batch
        return params["w"] * jnp.nan

    with pytest.warns(RuntimeWarning, match="non-finite"):
        result = fit(
            nonfinite,
            {"w": jnp.array(0.0)},
            jax.random.key(0),
            jnp.ones(4),
            num_steps=5,
        )
    assert result.losses.shape == (5,)
    assert jnp.isnan(result.losses[0])
    assert int(result.state.step) == 5


def test_zero_steps_leave_the_params_alone():
    result = fit(
        loss,
        {"w": jnp.array(2.0)},
        jax.random.key(0),
        jnp.ones(4),
        num_steps=0,
        ema_decay=0.9,
        use_ema=True,
    )
    assert result.losses.shape == (0,)
    assert result.params["w"] == 2


#: Train only ``w``: ``fixed`` is a Param, but frozen.
ONLY_W = nnx.All(nnx.Param, lambda path, value: path[-1] == "w")


class TinyModel(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.array(0.0))
        self.fixed = nnx.Param(jnp.array(1.0))
        self.count = nnx.Variable(jnp.array(0))
        self.offset = nnx.Variable(jnp.array(0.0))

    def loss(self, rng, batch):
        self.count[...] += 1
        return jnp.mean((self.w[...] + self.fixed[...] + self.offset[...] - batch) ** 2)


def test_module_frozen_params_state_updates_and_repeated_fit():
    model = TinyModel()
    result = fit_module(
        model,
        jax.random.key(0),
        jnp.full((8,), 3.0),
        trainable=ONLY_W,
        num_steps=2,
        optimizer=optax.sgd(0.1),
        ema_decay=0.8,
        use_ema=True,
    )
    assert model.count[...] == 2
    assert model.fixed[...] == 1
    np.testing.assert_allclose(model.w[...], result.params["w"][...])
    model.offset[...] = jnp.array(10.0)
    before = float(model.w[...])
    fit_module(
        model,
        jax.random.key(1),
        jnp.full((8,), 3.0),
        trainable=ONLY_W,
        num_steps=1,
        optimizer=optax.sgd(0.1),
    )
    assert model.count[...] == 3
    assert model.w[...] < before  # sees the changed non-Param state


def test_module_batchnorm_statistics_survive_fit():
    class Model(nnx.Module):
        def __init__(self):
            self.norm = nnx.BatchNorm(2, momentum=0.5, rngs=nnx.Rngs(0))

        def loss(self, rng, batch):
            del rng
            return jnp.mean(self.norm(batch) ** 2)

    model = Model()
    model.train()
    fit_module(model, jax.random.key(0), jnp.full((8, 2), 4.0), num_steps=2)
    np.testing.assert_allclose(model.norm.mean[...], [3.0, 3.0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"ema_decay": 1.0},
        {"ema_decay": -0.1},
        {"every": 0},
        {"every": 1.5},
        {"num_steps": 2.5},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"clip_norm": -1.0},
        {"schedule": "cosine"},
    ],
)
def test_fit_invalid_options(overrides):
    with pytest.raises((ValueError, TypeError)):
        fit(loss, {"w": jnp.array(0.0)}, jax.random.key(0), jnp.ones(4), **overrides)


def test_use_ema_without_an_averaging_rate_is_rejected():
    with pytest.raises(ValueError, match="use_ema requires ema_decay"):
        fit(
            loss,
            {"w": jnp.array(0.0)},
            jax.random.key(0),
            jnp.ones(4),
            num_steps=1,
            use_ema=True,
        )


def test_a_callback_that_is_not_callable_is_rejected():
    with pytest.raises(TypeError, match="callback must be callable"):
        fit(
            loss,
            {"w": jnp.array(0.0)},
            jax.random.key(0),
            jnp.ones(4),
            num_steps=1,
            callback=object(),
        )


def test_stream_ema_and_callbacks_match_full_batch():
    args = (loss, {"w": jnp.array(0.0)}, jax.random.key(0))
    batch = jnp.ones(8)
    array = fit(*args, batch, num_steps=5, ema_decay=0.5)
    stream = fit(
        *args,
        [batch] * 5,
        num_steps=5,
        ema_decay=0.5,
        every=2,
        callback=lambda state, info: None,
    )
    # Constant-folded array data and streamed data can differ by one FP32 ulp.
    np.testing.assert_allclose(array.losses, stream.losses, rtol=2e-7)
    np.testing.assert_allclose(
        array.state.ema_params["w"], stream.state.ema_params["w"], rtol=2e-7
    )

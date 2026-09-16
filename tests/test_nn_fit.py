"""Training extensions, mutable module state, and subclass customization."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from probjax.nn import FitCallback, FitMixin, fit


def loss(params, rng, batch):
    del rng
    return jnp.mean((params['w'] - batch) ** 2)


def test_ema_matches_manual_updates_and_keeps_raw_parameters():
    initial = {'w': jnp.array(0.0)}
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
        return_result=True,
    )
    np.testing.assert_allclose(result.state.params['w'], raw, rtol=1e-6)
    np.testing.assert_allclose(result.params['w'], ema, rtol=1e-6)
    assert int(result.state.step) == 5
    assert initial['w'] == 0


def test_callback_lifecycle_and_early_stop_use_device_snapshots():
    events = []

    class Monitor(FitCallback):
        def on_fit_begin(self, state):
            events.append(('begin', int(state.step)))

        def on_step_end(self, state, value):
            assert isinstance(state.params['w'], jax.Array)
            assert jnp.isfinite(value.loss)
            events.append(('step', int(state.step)))
            return np.bool_(state.step < 6)

        def on_fit_end(self, result):
            events.append(('end', int(result.state.step)))

    result = fit(
        loss,
        {'w': jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(8),
        num_steps=20,
        callbacks=[Monitor()],
        callback_every=3,
        ema_decay=0.9,
        return_result=True,
    )
    assert events == [('begin', 0), ('step', 3), ('step', 6), ('end', 6)]
    assert result.losses.shape == (6,)
    assert result.state.stopped


def test_callback_chunks_preserve_randomness_and_final_partial_chunk():
    seen = []
    kwargs = dict(num_steps=13, batch_size=3, ema_decay=0.8, return_result=True)
    args = (loss, {'w': jnp.array(0.0)}, jax.random.key(5), jnp.arange(9.0))
    single = fit(*args, **kwargs)
    chunked = fit(
        *args,
        **kwargs,
        callbacks=[lambda state, loss: seen.append(int(state.step))],
        callback_every=5,
    )
    assert seen == [5, 10, 13]
    np.testing.assert_array_equal(single.losses, chunked.losses)
    np.testing.assert_array_equal(single.params['w'], chunked.params['w'])
    np.testing.assert_array_equal(
        single.state.ema_params['w'], chunked.state.ema_params['w']
    )
    np.testing.assert_array_equal(
        jax.random.key_data(single.state.rng), jax.random.key_data(chunked.state.rng)
    )


def test_callback_can_validate_with_jit_and_exception_propagates():
    evaluate = jax.jit(lambda params: params['w'] ** 2)
    values = []

    def callback(state, loss):
        values.append(float(evaluate(state.params)))
        raise ValueError('validation failed')

    with pytest.raises(ValueError, match='validation failed'):
        fit(
            loss,
            {'w': jnp.array(0.0)},
            jax.random.key(0),
            jnp.ones(4),
            num_steps=3,
            callbacks=[callback],
        )
    assert len(values) == 1


def test_legacy_stop_keeps_nan_loss_and_actual_update_count():
    def nonfinite(params, rng, data):
        return params['w'] * jnp.nan

    with pytest.warns(RuntimeWarning, match='non-finite'):
        result = fit(
            nonfinite,
            {'w': jnp.array(0.0)},
            jax.random.key(0),
            jnp.ones(4),
            num_steps=5,
            on_step=lambda step, loss: False,
            callbacks=[lambda state, loss: None],
            callback_every=3,
            return_result=True,
        )
    assert result.losses.shape == (1,)
    assert jnp.isnan(result.losses[0])
    assert int(result.state.step) == 1


def test_stop_before_training_and_zero_steps():
    class Stop(FitCallback):
        def on_fit_begin(self, state):
            return False

    for kwargs in ({'num_steps': 0}, {'num_steps': 5, 'callbacks': [Stop()]}):
        result = fit(
            loss,
            {'w': jnp.array(2.0)},
            jax.random.key(0),
            jnp.ones(4),
            ema_decay=0.9,
            use_ema=True,
            return_result=True,
            **kwargs,
        )
        assert result.losses.shape == (0,)
        assert result.params['w'] == 2


def test_stateful_functional_loss():
    def stateful(params, rng, batch, state):
        return loss(params, rng, batch), {'count': state['count'] + 1}

    result = fit(
        stateful,
        {'w': jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        num_steps=4,
        model_state={'count': jnp.array(0)},
        return_result=True,
    )
    assert result.state.model_state['count'] == 4


class TinyModel(nnx.Module, FitMixin):
    def __init__(self):
        self.w = nnx.Param(jnp.array(0.0))
        self.fixed = nnx.Param(jnp.array(1.0))
        self.count = nnx.Variable(jnp.array(0))
        self.offset = nnx.Variable(jnp.array(0.0))

    def _fit_param_filter(self):
        return nnx.All(nnx.Param, lambda path, value: path[-1] == 'w')

    def loss(self, rng, data):
        self.count[...] += 1
        return jnp.mean((self.w[...] + self.fixed[...] + self.offset[...] - data) ** 2)


def test_module_frozen_params_state_updates_and_repeated_fit():
    model = TinyModel()
    result = model.fit(
        jax.random.key(0),
        jnp.full((8,), 3.0),
        num_steps=2,
        optimizer=optax.sgd(0.1),
        ema_decay=0.8,
        use_ema=True,
        return_result=True,
    )
    assert model.count[...] == 2
    assert model.fixed[...] == 1
    np.testing.assert_allclose(model.w[...], result.params['w'][...])
    model.offset[...] = jnp.array(10.0)
    before = float(model.w[...])
    model.fit(
        jax.random.key(1), jnp.full((8,), 3.0), num_steps=1, optimizer=optax.sgd(0.1)
    )
    assert model.count[...] == 3
    assert model.w[...] < before  # sees the changed non-Param state


def test_module_batchnorm_statistics_survive_fit():
    class Model(nnx.Module, FitMixin):
        def __init__(self):
            self.norm = nnx.BatchNorm(2, momentum=0.5, rngs=nnx.Rngs(0))

        def loss(self, rng, data):
            return jnp.mean(self.norm(data) ** 2)

    model = Model()
    model.train()
    model.fit(jax.random.key(0), jnp.full((8, 2), 4.0), num_steps=2)
    np.testing.assert_allclose(model.norm.mean[...], [3.0, 3.0])


@pytest.mark.parametrize(
    'kwargs',
    [
        {'ema_decay': 1.0},
        {'ema_decay': -0.1},
        {'use_ema': True},
        {'callback_every': 0},
        {'callback_every': 1.5},
        {'num_steps': 2.5},
        {'batch_size': 0},
        {'callbacks': [object()]},
    ],
)
def test_fit_invalid_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        fit(loss, {'w': jnp.array(0.0)}, jax.random.key(0), jnp.ones(4), **kwargs)


def test_model_training_hooks_compose_with_caller_options():
    seen = []

    class Model(TinyModel):
        def _prepare_fit(self, data):
            self.offset[...] = jnp.array(2.0)
            return data + 1

        def _default_fit_kwargs(self):
            return {'num_steps': 5, 'optimizer': optax.sgd(0.1)}

        def _fit_callbacks(self):
            return (lambda state, loss: seen.append(('model', int(state.step))),)

        def _fit_loss(self, rng, batch):
            return super()._fit_loss(rng, batch) * 2

    model = Model()
    losses = model.fit(
        jax.random.key(0),
        jnp.full((4,), 3.0),
        num_steps=1,
        callbacks=[lambda state, loss: seen.append(('caller', int(state.step)))],
    )
    np.testing.assert_allclose(losses, [2.0])
    np.testing.assert_allclose(model.w[...], 0.4)
    assert seen == [('model', 1), ('caller', 1)]


def test_stream_ema_and_callbacks_match_full_batch():
    args = (loss, {'w': jnp.array(0.0)}, jax.random.key(0))
    batch = jnp.ones(8)
    kwargs = dict(num_steps=5, ema_decay=0.5, return_result=True)
    array = fit(*args, batch, **kwargs)
    stream = fit(
        *args,
        [batch, batch],
        callbacks=[lambda state, loss: None],
        callback_every=2,
        **kwargs,
    )
    # Constant-folded array data and streamed data can differ by one FP32 ulp.
    np.testing.assert_allclose(array.losses, stream.losses, rtol=2e-7)
    np.testing.assert_allclose(
        array.state.ema_params['w'], stream.state.ema_params['w'], rtol=2e-7
    )

"""The complete functional fit composes with JIT; host effects are opt-in."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from probjax.nn import FitCallback, FitResult, fit


def loss(params, key, batch):
    return jnp.mean((params - batch) ** 2)


def test_complete_fit_jits_with_legacy_and_detailed_returns():
    args = (jnp.array(0.0), jax.random.key(0), jnp.arange(5.0))
    kwargs = dict(num_steps=7, batch_size=3, ema_decay=0.8, use_ema=True)
    expected_params, expected_losses = fit(loss, *args, **kwargs)
    actual_params, actual_losses = jax.jit(
        lambda p, k, x: fit(loss, p, k, x, **kwargs)
    )(*args)
    np.testing.assert_array_equal(actual_params, expected_params)
    np.testing.assert_array_equal(actual_losses, expected_losses)
    result = jax.jit(lambda p, k, x: fit(loss, p, k, x, return_result=True, **kwargs))(
        *args
    )
    assert isinstance(result, FitResult)
    assert int(result.valid_steps) == 7
    assert np.all(result.valid)
    assert result.info.loss.shape == (7,)


def test_complete_fit_resumes_under_jit_and_preserves_auxiliary_state():
    def objective(p, k, x, state):
        return loss(p, k, x), (state + 1, {'counter': state})

    kwargs = dict(
        num_steps=3, optimizer=optax.adam(0.01), has_aux=True, return_result=True
    )
    first = fit(
        objective,
        jnp.array(0.0),
        jax.random.key(0),
        jnp.ones(4),
        model_state=jnp.array(0),
        **kwargs,
    )
    result = jax.jit(
        lambda state, x: fit(objective, None, None, x, initial_state=state, **kwargs)
    )(first.state, jnp.ones(4))
    assert int(result.state.step) == 6
    assert int(result.valid_steps) == 3
    assert int(result.state.model_state) == 6
    np.testing.assert_array_equal(result.info.metrics['counter'], [3, 4, 5])


def test_explicit_loss_callback_inside_jit_has_fixed_history():
    seen = []

    def on_step(step, value):
        seen.append(step)
        return step < 2

    result = jax.jit(
        lambda p, k, x: fit(
            loss, p, k, x, num_steps=8, on_step=on_step, return_result=True
        )
    )(jnp.array(0.0), jax.random.key(0), jnp.ones(4))
    jax.block_until_ready(result)
    assert seen == [0, 1, 2]
    assert result.losses.shape == (8,)
    assert int(result.valid_steps) == 3
    np.testing.assert_array_equal(result.valid, [True] * 3 + [False] * 5)
    assert result.state.stopped


def test_io_lifecycle_callbacks_execute_at_runtime_and_forward_state_info():
    seen = []
    keys = []

    class Monitor(FitCallback):
        def on_fit_begin(self, state):
            seen.append(('begin', int(state.step)))
            keys.append(np.asarray(jax.random.key_data(state.rng)))

        def on_step_end(self, state, info):
            assert np.isfinite(info.loss)
            assert isinstance(state.params, np.ndarray)
            seen.append(('step', int(state.step)))
            return int(state.step) < 4

        def on_fit_end(self, result):
            seen.append(('end', int(result.valid_steps)))

    train = jax.jit(
        lambda p, k, x: fit(
            loss,
            p,
            k,
            x,
            num_steps=9,
            callbacks=[Monitor()],
            callback_mode='io',
            callback_every=2,
            return_result=True,
        )
    )
    for seed in (0, 1):
        key = jax.random.key(seed)
        result = train(jnp.array(0.0), key, jnp.ones(4))
        jax.block_until_ready(result)
        jax.effects_barrier()
        assert int(result.valid_steps) == 4
        assert result.losses.shape == (9,)
        np.testing.assert_array_equal(keys[-1], jax.random.key_data(key))
    assert seen == [('begin', 0), ('step', 2), ('step', 4), ('end', 4)] * 2


def test_io_begin_stop_and_zero_steps_preserve_valid_count():
    class Stop(FitCallback):
        def on_fit_begin(self, state):
            return False

    for count in (0, 4):
        result = jax.jit(
            lambda p, k, x, count=count: fit(
                loss,
                p,
                k,
                x,
                num_steps=count,
                callbacks=[Stop()],
                callback_mode='io',
                return_result=True,
            )
        )(jnp.array(2.0), jax.random.key(0), jnp.ones(4))
        assert int(result.valid_steps) == 0
        assert result.params == 2
        assert result.losses.shape == (count,)
        assert not np.any(result.valid)


def test_host_callbacks_require_explicit_io_mode_under_jit():
    with pytest.raises(ValueError, match="callback_mode='io'"):
        jax.jit(
            lambda p, k, x: fit(
                loss, p, k, x, num_steps=1, callbacks=[lambda s, i: None]
            )
        )(jnp.array(0.0), jax.random.key(0), jnp.ones(4))


def test_pure_fit_contains_no_host_callback_and_differentiates():
    train = lambda p, x: fit(
        loss, p, jax.random.key(0), x, num_steps=2, optimizer=optax.sgd(0.1)
    )[0]
    text = str(jax.make_jaxpr(train)(jnp.array(0.0), jnp.ones(4)))
    assert 'io_callback' not in text
    assert 'debug_callback' not in text
    gradient = jax.jit(jax.grad(lambda p: train(p, jnp.ones(4))))(jnp.array(0.0))
    np.testing.assert_allclose(gradient, 0.8**2, rtol=1e-6)


def test_module_fit_under_nnx_jit_updates_parameters_and_mutable_state():
    from flax import nnx

    from probjax.nn import FitMixin

    class Model(nnx.Module, FitMixin):
        def __init__(self):
            self.w = nnx.Param(jnp.array(0.0))
            self.count = nnx.Variable(jnp.array(0))

        def loss(self, key, batch):
            self.count[...] += 1
            return jnp.mean((self.w[...] - batch) ** 2)

    model = Model()

    @nnx.jit
    def train(model, key, batch):
        return model.fit(key, batch, num_steps=3, return_result=True)

    result = train(model, jax.random.key(0), jnp.ones(4))
    assert model.w[...] > 0
    assert model.count[...] == 3
    assert int(result.valid_steps) == 3


def test_flow_fit_under_nnx_jit_standardizes_only_once():
    from flax import nnx

    from probjax.nn import maf

    model = maf(2, 1, rngs=nnx.Rngs(0))

    @nnx.jit
    def train(model, key, batch):
        return model.fit(key, batch, num_steps=2, return_result=True)

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

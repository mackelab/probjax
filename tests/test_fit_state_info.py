"""State/info forwarding through pure updates, fitting, callbacks, and resume."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from probjax.nn import FitCallback, FitInfo, FitKernel, FitMixin, build_fit_kernel, fit


class Metrics(NamedTuple):
    prediction: object
    error: object


def loss(params, rng, batch):
    residual = params['w'] - batch
    return jnp.mean(residual**2), {'nested': Metrics(params['w'], residual.mean())}


def assert_tree_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_array_equal(x, y)


def test_pure_kernel_scan_and_fit_forward_identical_state_info():
    params, key, batch = {'w': jnp.array(0.0)}, jax.random.key(0), jnp.ones(4)
    kernel = build_fit_kernel(loss, optax.sgd(0.1), ema_decay=0.8, has_aux=True)
    state = kernel.init(params, key)
    expected_state, expected_info = jax.jit(
        lambda state: jax.lax.scan(
            lambda state, _: kernel.step(state.rng, state, batch), state, None, length=4
        )
    )(state)
    result = fit(
        loss, params, key, batch, kernel=kernel, num_steps=4, return_result=True
    )
    assert_tree_equal(result.state, expected_state)
    assert_tree_equal(result.info, expected_info)
    np.testing.assert_array_equal(result.losses, result.info.loss)
    assert isinstance(result.info.metrics['nested'], Metrics)
    assert result.info.metrics['nested'].prediction.shape == (4,)


def test_auxiliary_metrics_reach_every_callback_and_partial_history():
    seen = []

    class Monitor(FitCallback):
        def on_step_end(self, state, info):
            assert isinstance(info, FitInfo)
            seen.append((int(state.step), info))

    result = fit(
        loss,
        {'w': jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        optimizer=optax.sgd(0.1),
        has_aux=True,
        num_steps=5,
        callbacks=[Monitor()],
        callback_every=2,
        return_result=True,
    )
    assert [step for step, _ in seen] == [2, 4, 5]
    for step, info in seen:
        assert_tree_equal(
            info, jax.tree.map(lambda x, step=step: x[step - 1], result.info)
        )


def test_resume_forwards_optimizer_rng_ema_and_absolute_step():
    params, key = {'w': jnp.array(0.0)}, jax.random.key(7)
    batch = jnp.arange(8.0)
    kwargs = dict(
        optimizer=optax.adam(0.01),
        has_aux=True,
        ema_decay=0.9,
        batch_size=3,
        return_result=True,
    )
    whole = fit(loss, params, key, batch, num_steps=9, **kwargs)
    first = fit(loss, params, key, batch, num_steps=4, **kwargs)
    seen = []
    resumed = fit(
        loss,
        None,
        None,
        batch,
        initial_state=first.state,
        num_steps=5,
        callbacks=[lambda state, info: seen.append(int(state.step))],
        callback_every=2,
        **kwargs,
    )
    assert seen == [6, 8, 9]
    assert_tree_equal(whole.state, resumed.state)
    assert_tree_equal(
        whole.info,
        jax.tree.map(lambda a, b: jnp.concatenate([a, b]), first.info, resumed.info),
    )
    assert resumed.losses.shape == (5,)


def test_early_stop_truncates_all_info_leaves():
    result = fit(
        loss,
        {'w': jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        has_aux=True,
        num_steps=7,
        on_step=lambda step, value: step < 2,
        callbacks=[lambda state, info: None],
        callback_every=4,
        return_result=True,
    )
    assert int(result.state.step) == 3
    assert all(x.shape[0] == 3 for x in jax.tree.leaves(result.info))


def test_empty_info_retains_metric_structure_and_dtypes():
    result = fit(
        loss,
        {'w': jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        has_aux=True,
        num_steps=0,
        callbacks=[lambda state, info: None],
        return_result=True,
    )
    assert isinstance(result.info.metrics['nested'], Metrics)
    assert all(x.shape == (0,) for x in jax.tree.leaves(result.info))


def test_custom_kernel_forwards_user_state_and_metrics():
    base = build_fit_kernel(lambda p, k, x: jnp.mean((p['w'] - x) ** 2), optax.sgd(0.1))

    def step(key, state, batch):
        state, info = base.step(key, state, batch)
        return state, FitInfo(info.loss, {'counter': state.step, 'vector': batch[:2]})

    kernel = FitKernel(base.init, step)
    result = fit(
        None,
        {'w': jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        kernel=kernel,
        num_steps=3,
        return_result=True,
    )
    np.testing.assert_array_equal(result.info.metrics['counter'], [1, 2, 3])
    assert result.info.metrics['vector'].shape == (3, 2)


def test_module_loss_aux_and_mutable_state_forward_together_on_resume():
    class Model(nnx.Module, FitMixin):
        def __init__(self):
            self.w = nnx.Param(jnp.array(0.0))
            self.count = nnx.Variable(jnp.array(0))

        def loss(self, rng, batch):
            self.count[...] += 1
            return jnp.mean((self.w[...] - batch) ** 2), {'count': self.count[...]}

    model = Model()
    kwargs = dict(
        num_steps=2, has_aux=True, optimizer=optax.sgd(0.1), return_result=True
    )
    first = model.fit(jax.random.key(0), jnp.ones(4), **kwargs)
    resumed = model.fit(None, jnp.ones(4), initial_state=first.state, **kwargs)
    np.testing.assert_array_equal(first.info.metrics['count'], [1, 2])
    np.testing.assert_array_equal(resumed.info.metrics['count'], [3, 4])
    assert model.count[...] == 4
    assert int(resumed.state.step) == 4


def test_resume_rejects_missing_ema_configuration():
    initial = build_fit_kernel(loss, optax.sgd(0.1), ema_decay=0.9, has_aux=True).init(
        {'w': jnp.array(0.0)}, jax.random.key(0)
    )
    with pytest.raises(ValueError, match='same EMA'):
        fit(
            loss,
            None,
            None,
            jnp.ones(4),
            initial_state=initial,
            num_steps=1,
            has_aux=True,
        )

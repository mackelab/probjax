"""State and metrics forwarding through updates, fitting, callbacks and resume."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from probjax.nn import DataSpec, FitCallback, FitConfig, FitMixin, fit


class Metrics(NamedTuple):
    prediction: object
    error: object


def loss(params, rng, primary):
    del rng
    residual = params["w"] - primary
    return jnp.mean(residual**2), {"nested": Metrics(params["w"], residual.mean())}


def _config(**overrides):
    return FitConfig(data=DataSpec(), **overrides)


def assert_tree_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_array_equal(x, y)


def test_metrics_are_stacked_into_the_result():
    result = fit(
        loss,
        {"w": jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        config=_config(max_steps=4, optimizer=optax.sgd(0.1)),
    )
    assert result.losses.shape == (4,)
    assert isinstance(result.metrics["nested"], Metrics)
    assert result.metrics["nested"].prediction.shape == (4,)
    assert result.metrics["nested"].error.shape == (4,)


def test_metrics_reach_every_callback():
    seen = []

    class Monitor(FitCallback):
        def on_step_end(self, state, info):
            seen.append((int(state.step), info))

    result = fit(
        loss,
        {"w": jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        config=_config(max_steps=5, optimizer=optax.sgd(0.1), callback_every=2),
        callbacks=[Monitor()],
    )
    assert [step for step, _ in seen] == [2, 4, 5]
    for step, info in seen:
        # The callback sees the last update of its chunk, which is the one
        # the result recorded at the same position.
        assert float(info.loss) == float(result.losses[step - 1])


def test_resume_forwards_optimizer_rng_ema_and_absolute_step():
    params, key = {"w": jnp.array(0.0)}, jax.random.key(7)
    batch = jnp.arange(8.0)
    config = _config(
        optimizer=optax.adam(0.01), ema_decay=0.9, batch_size=3, max_steps=9
    )
    whole = fit(loss, params, key, batch, config=config)
    first = fit(loss, params, key, batch, config=config.merged(max_steps=4))
    seen = []
    resumed = fit(
        loss,
        None,
        None,
        batch,
        state=first.state,
        config=config.merged(max_steps=5, callback_every=2),
        callbacks=[lambda state, info: seen.append(int(state.step))],
    )
    assert seen == [6, 8, 9]
    assert_tree_equal(whole.state, resumed.state)
    assert_tree_equal(
        whole.metrics,
        jax.tree.map(
            lambda a, b: jnp.concatenate([a, b]), first.metrics, resumed.metrics
        ),
    )
    assert resumed.losses.shape == (5,)


def test_early_stop_truncates_the_history():
    class Stop(FitCallback):
        def __init__(self):
            self.calls = 0

        def on_step_end(self, state, info):
            self.calls += 1
            return self.calls < 2

    result = fit(
        loss,
        {"w": jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        config=_config(max_steps=7, callback_every=4),
        callbacks=[Stop()],
    )
    # The callback saw the first chunk of four and the last one of three, and
    # stopped the run there: every update that ran is in the history.
    assert int(result.state.step) == 7
    assert all(x.shape[0] == 7 for x in jax.tree.leaves(result.metrics))
    assert result.losses.shape == (7,)


def test_a_run_without_updates_has_no_metrics():
    """No update means no metric: the history is empty, not zero-filled.

    Zero-filling would mean tracing the loss once more, per run, only to
    discover the shape of something nobody asked for.
    """
    result = fit(
        loss,
        {"w": jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        config=_config(max_steps=0),
    )
    assert result.losses.shape == (0,)
    assert result.metrics is None
    assert int(result.state.step) == 0


def test_module_loss_aux_and_mutable_state_forward_together_on_resume():
    class Model(nnx.Module, FitMixin):
        def __init__(self):
            self.w = nnx.Param(jnp.array(0.0))
            self.count = nnx.Variable(jnp.array(0))

        def loss(self, rng, primary):
            self.count[...] += 1
            return jnp.mean((self.w[...] - primary) ** 2), {"count": self.count[...]}

    model = Model()
    config = FitConfig(max_steps=2, optimizer=optax.sgd(0.1))
    first = model.fit(jax.random.key(0), jnp.ones(4), config=config)
    resumed = model.fit(None, jnp.ones(4), state=first.state, config=config)
    np.testing.assert_array_equal(first.metrics["count"], [1, 2])
    np.testing.assert_array_equal(resumed.metrics["count"], [3, 4])
    assert model.count[...] == 4
    assert int(resumed.state.step) == 4


def test_a_resume_without_an_averaging_accumulator_is_rejected():
    first = fit(
        loss,
        {"w": jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        config=_config(max_steps=2),
    )
    assert first.state.ema_params is None
    with pytest.raises(ValueError, match="use_ema requires ema_decay"):
        fit(
            loss,
            None,
            None,
            jnp.ones(4),
            state=first.state,
            config=_config(max_steps=1, use_ema=True),
        )

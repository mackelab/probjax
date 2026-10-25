"""State and metrics forwarding through updates, fitting, callbacks and resume."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from probjax.nn import fit, fit_module


class Metrics(NamedTuple):
    prediction: object
    error: object


def loss(params, rng, primary):
    del rng
    residual = params["w"] - primary
    return jnp.mean(residual**2), {"nested": Metrics(params["w"], residual.mean())}


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
        num_steps=4,
        optimizer=optax.sgd(0.1),
    )
    assert result.losses.shape == (4,)
    assert isinstance(result.metrics["nested"], Metrics)
    assert result.metrics["nested"].prediction.shape == (4,)
    assert result.metrics["nested"].error.shape == (4,)


def test_metrics_reach_every_callback():
    seen = []

    def monitor(state, info):
        seen.append((int(state.step), info))

    result = fit(
        loss,
        {"w": jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        num_steps=5,
        optimizer=optax.sgd(0.1),
        every=2,
        callback=monitor,
    )
    assert [step for step, _ in seen] == [2, 4, 5]
    for step, info in seen:
        # The callback sees the last update of its chunk, which is the one
        # the result recorded at the same position.
        assert float(info.loss) == float(result.losses[step - 1])


def test_resume_forwards_optimizer_rng_ema_and_absolute_step():
    params, key = {"w": jnp.array(0.0)}, jax.random.key(7)
    batch = jnp.arange(8.0)
    options = dict(optimizer=optax.adam(0.01), ema_decay=0.9, batch_size=3)
    whole = fit(loss, params, key, batch, num_steps=9, **options)
    first = fit(loss, params, key, batch, num_steps=4, **options)
    seen = []
    resumed = fit(
        loss,
        None,
        None,
        batch,
        state=first.state,
        num_steps=5,
        every=2,
        callback=lambda state, info: seen.append(int(state.step)),
        **options,
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
    calls = []

    def stop(state, info):
        calls.append(int(state.step))
        return False

    result = fit(
        loss,
        {"w": jnp.array(0.0)},
        jax.random.key(0),
        jnp.ones(4),
        num_steps=12,
        every=4,
        callback=stop,
    )
    # The callback saw the first chunk of four and stopped the run there:
    # every update that ran is in the history, and none that did not.
    assert calls == [4]
    assert int(result.state.step) == 4
    assert all(x.shape[0] == 4 for x in jax.tree.leaves(result.metrics))
    assert result.losses.shape == (4,)


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
        num_steps=0,
    )
    assert result.losses.shape == (0,)
    assert result.metrics is None
    assert int(result.state.step) == 0


def test_module_loss_aux_and_mutable_state_forward_together_on_resume():
    class Model(nnx.Module):
        def __init__(self):
            self.w = nnx.Param(jnp.array(0.0))
            self.count = nnx.Variable(jnp.array(0))

        def loss(self, rng, primary):
            self.count[...] += 1
            return jnp.mean((self.w[...] - primary) ** 2), {"count": self.count[...]}

    model = Model()
    options = dict(num_steps=2, optimizer=optax.sgd(0.1))
    first = fit_module(model, jax.random.key(0), jnp.ones(4), **options)
    resumed = fit_module(model, None, jnp.ones(4), state=first.state, **options)
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
        num_steps=2,
    )
    assert first.state.ema_params is None
    with pytest.raises(ValueError, match="no averaged parameters"):
        fit(
            loss,
            None,
            None,
            jnp.ones(4),
            state=first.state,
            num_steps=1,
            ema_decay=0.9,
        )

"""Extension points of ``probjax.nn.training``: losses, optimizers and recipes."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from probjax.nn import fit, fit_module
from probjax.nn.generative import maf

KEY = jax.random.key(0)
X = jnp.linspace(0.5, 1.5, 32)


def _line_loss(params, rng, batch):
    del rng
    return jnp.mean((batch["x"] * params["w"] - batch["y"]) ** 2)


def test_loop_trains_on_any_batch_pytree():
    batch = {"x": X, "y": 3.0 * X}  # keys are arbitrary: nothing is called "data"
    result = fit(
        _line_loss, {"w": jnp.array(0.0)}, KEY, batch, num_steps=400, learning_rate=0.05
    )
    assert abs(float(result.params["w"]) - 3.0) < 0.1
    assert result.losses[-1] < result.losses[0]


def test_a_tuple_batch_is_one_batch():
    def loss(params, rng, batch):
        x, y = batch
        return jnp.mean((x * params["w"] - y) ** 2)

    result = fit(
        loss,
        {"w": jnp.array(0.0)},
        KEY,
        (X, 3.0 * X),
        num_steps=400,
        learning_rate=0.05,
    )
    assert abs(float(result.params["w"]) - 3.0) < 0.1


def test_schedule_accepts_an_optax_schedule():
    def loss(params, rng, batch):
        del rng
        return jnp.mean((params["w"] - batch) ** 2)

    data = jnp.full((8,), 2.0)
    common = dict(num_steps=5, clip_norm=None)
    frozen = fit(
        loss,
        {"w": jnp.array(0.0)},
        KEY,
        data,
        schedule=optax.constant_schedule(0.0),
        **common,
    )
    moving = fit(loss, {"w": jnp.array(0.0)}, KEY, data, schedule="constant", **common)
    # The supplied schedule, not learning_rate, ran.
    assert float(frozen.params["w"]) == 0.0
    assert float(moving.params["w"]) > 0.0


def test_an_explicit_optimizer_wins_over_the_rate_options():
    def loss(params, rng, batch):
        del rng
        return jnp.mean((params["w"] - batch) ** 2)

    result = fit(
        loss,
        {"w": jnp.array(0.0)},
        KEY,
        jnp.full((8,), 2.0),
        num_steps=3,
        optimizer=optax.sgd(0.0),
        learning_rate=1.0,
    )
    assert float(result.params["w"]) == 0.0


class Regression(nnx.Module):
    def __init__(self):
        self.w = nnx.Param(jnp.array(0.0))


def test_fit_module_takes_a_custom_loss():
    mask = (jnp.arange(32) < 16).astype(jnp.float32)
    y = jnp.where(mask > 0, 3.0 * X, -100.0)  # the masked half is garbage

    def masked_loss(model, rng, batch):
        residual = (batch["x"] * model.w[...] - batch["y"]) ** 2
        return jnp.sum(batch["mask"] * residual) / jnp.sum(batch["mask"])

    model = Regression()
    fit_module(
        model,
        KEY,
        {"x": X, "y": y, "mask": mask},
        loss=masked_loss,
        num_steps=400,
        learning_rate=0.05,
    )
    assert abs(float(model.w[...]) - 3.0) < 0.1


def test_a_model_recipe_survives_a_call_that_changes_one_option(monkeypatch):
    """``model.fit`` keywords override the recipe; the rest of it is kept."""
    seen = {}
    import probjax.nn.generative.base as base

    def record(model, rng, data, **options):
        seen.update(options)

    monkeypatch.setattr(base, "fit_module", record)
    model = maf(2, 2, rngs=nnx.Rngs(0))
    model.fit(KEY, jnp.zeros((8, 2)), num_steps=7)
    assert seen["num_steps"] == 7
    assert seen["learning_rate"] == 3e-3
    assert seen["schedule"] == "warmup_cosine"


def test_generative_models_take_one_data_dict_of_named_inputs(monkeypatch):
    batches = []
    import probjax.nn.generative.base as base

    monkeypatch.setattr(
        base, "fit_module", lambda model, rng, data, **options: batches.append(data)
    )
    model = maf(2, 2, rngs=nnx.Rngs(0))
    x = jnp.zeros((4, 2))
    model.fit(KEY, {"data": x, "context": x}, weights=jnp.ones(4))
    assert set(batches[-1]) == {"data", "context", "weights"}
    model.fit(KEY, x)
    assert set(batches[-1]) == {"data"}
    model.fit(KEY, [x, x])
    assert [set(b) for b in batches[-1]] == [{"data"}, {"data"}]
    with pytest.raises(ValueError, match="alongside a stream"):
        model.fit(KEY, [x], weights=jnp.ones(4))
    with pytest.raises(ValueError, match="'data' key"):
        model.fit(KEY, {"context": x})

"""Extension points of ``probjax.nn.training``: no assumption about batch layout."""

from typing import cast

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from probjax.nn import DataSpec, FitConfig, FitMixin, fit
from probjax.nn.training.config import Schedule

KEY = jax.random.key(0)
X = jnp.linspace(0.5, 1.5, 32)


def _line_loss(params, rng, primary, **context):
    del rng
    return jnp.mean((primary * params["w"] - context["y"]) ** 2)


def test_loop_trains_on_any_batch_pytree():
    batch = {"x": X, "y": 3.0 * X}  # keys are arbitrary: nothing is called "data"
    config = FitConfig(data=DataSpec(primary="x"), max_steps=400, learning_rate=0.05)
    result = fit(_line_loss, {"w": jnp.array(0.0)}, KEY, batch, config=config)
    assert abs(float(result.params["w"]) - 3.0) < 0.1
    assert result.losses[-1] < result.losses[0]


def test_schedule_accepts_an_optax_schedule():
    def loss(params, rng, primary):
        del rng
        return jnp.mean((params["w"] - primary) ** 2)

    data = jnp.full((8,), 2.0)
    frozen = fit(
        loss,
        {"w": jnp.array(0.0)},
        KEY,
        data,
        config=FitConfig(
            data=DataSpec(),
            max_steps=5,
            clip_norm=None,
            schedule=optax.constant_schedule(0.0),
        ),
    )
    moving = fit(
        loss,
        {"w": jnp.array(0.0)},
        KEY,
        data,
        config=FitConfig(
            data=DataSpec(), max_steps=5, clip_norm=None, schedule="constant"
        ),
    )
    assert (
        float(frozen.params["w"]) == 0.0
    )  # the supplied schedule, not learning_rate, ran
    assert float(moving.params["w"]) > 0.0


def test_unknown_schedule_name_is_still_rejected():
    with pytest.raises(ValueError, match="schedule"):
        FitConfig(data=DataSpec(), schedule=cast(Schedule, "cosine"))


class Regression(nnx.Module, FitMixin):
    def __init__(self):
        self.w = nnx.Param(jnp.array(0.0))

    def loss(self, rng, primary, **context):
        del rng
        return jnp.mean((primary * self.w[...] - context["y"]) ** 2)


def test_fitmixin_hands_the_data_pytree_to_loss_unchanged():
    model = Regression()
    result = model.fit(
        KEY,
        {"x": X, "y": 3.0 * X},
        config=FitConfig(data=DataSpec(primary="x"), max_steps=400, learning_rate=0.05),
    )
    assert abs(float(model.w[...]) - 3.0) < 0.1
    assert result.losses[-1] < result.losses[0]


def test_fitmixin_rejects_unknown_keywords_by_default():
    with pytest.raises(TypeError, match=r"unexpected keyword arguments.*'mask'"):
        Regression().fit(
            KEY,
            {"x": X, "y": X},
            config=FitConfig(data=DataSpec(primary="x"), max_steps=2),
            mask=jnp.ones_like(X),
        )


class MaskedRegression(Regression):
    """A model that defines its own batch layout through ``_fit_batch``."""

    def _fit_batch(self, data, config, *, mask, **extras):
        x, y = data
        return {"x": x, "y": y, "mask": mask}

    def loss(self, rng, primary, **context):
        del rng
        residual = (primary * self.w[...] - context["y"]) ** 2
        return jnp.sum(context["mask"] * residual) / jnp.sum(context["mask"])


def test_fitmixin_forwards_extra_keywords_to_the_fit_batch_hook():
    mask = (jnp.arange(32) < 16).astype(jnp.float32)
    y = jnp.where(mask > 0, 3.0 * X, -100.0)  # the masked half is garbage
    model = MaskedRegression()
    model.fit(
        KEY,
        (X, y),
        config=FitConfig(data=DataSpec(primary="x"), max_steps=400, learning_rate=0.05),
        mask=mask,
    )
    assert abs(float(model.w[...]) - 3.0) < 0.1


def test_a_configuration_overrides_only_the_fields_it_names():
    """A model's own recipe survives a configuration that changes one thing."""

    class Model(nnx.Module, FitMixin):
        def __init__(self):
            self.w = nnx.Param(jnp.array(0.0))

        def default_config(self):
            return (
                super()
                .default_config()
                .merged(learning_rate=3e-3, schedule="warmup_cosine")
            )

        def loss(self, rng, primary):
            del rng
            return jnp.mean((self.w[...] - primary) ** 2)

    model = Model()
    model.fit(KEY, jnp.ones((8,)), config=FitConfig(max_steps=2))
    recipe = model.default_config()
    assert recipe.learning_rate == 3e-3
    assert recipe.schedule == "warmup_cosine"

    # A configuration that names one field keeps the other from the recipe.
    merged = recipe.merged(max_steps=7)
    assert merged.max_steps == 7
    assert merged.learning_rate == 3e-3
    assert merged.schedule == "warmup_cosine"


def test_generative_models_take_one_data_dict_of_named_inputs():
    from probjax.nn.generative import maf

    model = maf(2, 2, rngs=nnx.Rngs(0))
    x = jnp.zeros((4, 2))
    config = model.default_config()
    batch = model._fit_batch({"data": x, "context": x}, config, weights=jnp.ones(4))
    assert set(batch) == {"data", "context", "weights"}
    with pytest.raises(ValueError, match="alongside an iterable"):
        model._fit_batch(
            [x],
            config.merged(data=DataSpec(primary="data", stream=True)),
            weights=jnp.ones(4),
        )
    assert set(model._fit_batch(x, config)) == {"data"}
    with pytest.raises(ValueError, match="inside data"):
        model._fit_batch(x, config, context=x)
    with pytest.raises(ValueError, match="'data' key"):
        model._fit_batch({"context": x}, config)

"""Extension points of ``probjax.nn.training``: no assumption about batch layout."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from probjax.nn import FitMixin, fit
from probjax.nn.training import mixin
from probjax.nn.training.kernel import _build_optimizer

KEY = jax.random.key(0)
X = jnp.linspace(0.5, 1.5, 32)


def _line_loss(params, rng, batch):
    del rng
    return jnp.mean((batch["x"] * params["w"] - batch["y"]) ** 2)


def test_loop_trains_on_any_batch_pytree():
    batch = {"x": X, "y": 3.0 * X}  # keys are arbitrary: nothing is called "data"
    params, losses = fit(
        _line_loss, {"w": jnp.array(0.0)}, KEY, batch, num_steps=400, learning_rate=0.05
    )
    assert abs(float(params["w"]) - 3.0) < 0.1
    assert losses[-1] < losses[0]


def test_schedule_accepts_an_optax_schedule():
    def loss(params, rng, batch):
        del rng
        return jnp.mean((params["w"] - batch) ** 2)

    kwargs = dict(num_steps=5, clip_norm=None)
    data = jnp.full((8,), 2.0)
    frozen, _ = fit(
        loss,
        {"w": jnp.array(0.0)},
        KEY,
        data,
        schedule=optax.constant_schedule(0.0),
        **kwargs,
    )
    moving, _ = fit(
        loss, {"w": jnp.array(0.0)}, KEY, data, schedule="constant", **kwargs
    )
    assert float(frozen["w"]) == 0.0  # the supplied schedule, not learning_rate, ran
    assert float(moving["w"]) > 0.0


def test_unknown_schedule_name_is_still_rejected():
    with pytest.raises(ValueError, match="schedule"):
        _build_optimizer(1e-3, 10, "cosine", None)


class Regression(nnx.Module, FitMixin):
    def __init__(self):
        self.w = nnx.Param(jnp.array(0.0))

    def loss(self, rng, batch):
        del rng
        return jnp.mean((batch["x"] * self.w[...] - batch["y"]) ** 2)


def test_fitmixin_hands_the_data_pytree_to_loss_unchanged():
    model = Regression()
    losses = model.fit(KEY, {"x": X, "y": 3.0 * X}, num_steps=400, learning_rate=0.05)
    assert abs(float(model.w[...]) - 3.0) < 0.1
    assert losses[-1] < losses[0]


def test_fitmixin_rejects_unknown_keywords_by_default():
    with pytest.raises(TypeError, match=r"unexpected keyword arguments.*'mask'"):
        Regression().fit(KEY, {"x": X, "y": X}, mask=jnp.ones_like(X))


class MaskedRegression(Regression):
    """A model that defines its own batch layout through ``_fit_batch``."""

    def _fit_batch(self, data, *, mask):
        x, y = data
        return {"x": x, "y": y, "mask": mask}

    def loss(self, rng, batch):
        del rng
        residual = (batch["x"] * self.w[...] - batch["y"]) ** 2
        return jnp.sum(batch["mask"] * residual) / jnp.sum(batch["mask"])


def test_fitmixin_forwards_extra_keywords_to_the_fit_batch_hook():
    mask = (jnp.arange(32) < 16).astype(jnp.float32)
    y = jnp.where(mask > 0, 3.0 * X, -100.0)  # the masked half is garbage
    model = MaskedRegression()
    model.fit(KEY, (X, y), mask=mask, num_steps=400, learning_rate=0.05)
    assert abs(float(model.w[...]) - 3.0) < 0.1


def test_loop_arguments_are_not_mistaken_for_batch_extras():
    loop_arguments = mixin._LOOP_ARGUMENTS
    assert {
        "num_steps",
        "learning_rate",
        "schedule",
        "callbacks",
        "initial_state",
        "kernel",
        "return_result",
    } <= loop_arguments
    assert not loop_arguments & {"loss_fn", "params", "rng", "batch", "context"}


def test_generative_models_take_one_data_dict_of_named_inputs():
    from probjax.nn.generative import maf

    model = maf(2, 2, rngs=nnx.Rngs(0))
    x = jnp.zeros((4, 2))
    batch = model._fit_batch({"data": x, "context": x}, weights=jnp.ones(4))
    assert set(batch) == {"data", "context", "weights"}
    with pytest.raises(ValueError, match="alongside an iterable"):
        model._fit_batch([x], weights=jnp.ones(4))
    assert set(model._fit_batch(x)) == {"data"}
    with pytest.raises(ValueError, match="inside data"):
        model._fit_batch(x, context=x)
    with pytest.raises(ValueError, match="'data' key"):
        model._fit_batch({"context": x})

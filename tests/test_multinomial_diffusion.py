import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.diffusion.multinomial_diffusion import (
    CategoricalEDMPreconditioning,
    ImportanceContinuousTimeTrainingConfig,
    MultinomialDiffusion,
    MultinomialDiffusionSchedule,
    UniformContinuousTimeTrainingConfig,
)


def test_multinomial_schedule_closed_form_q():
    pi = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    schedule = MultinomialDiffusionSchedule.from_linear(
        num_steps=8,
        num_classes=4,
        beta_start=1e-3,
        beta_end=0.05,
        base_probs=pi,
    )
    x0 = jnp.array([[0, 2], [1, 3]], dtype=jnp.int32)
    t = jnp.full((2, 1), 5.0 / 8.0, dtype=jnp.float32)

    probs = schedule.q_xt_given_x0_probs(x0, t)
    alpha_bar_t = schedule.alpha_bar(t)[..., None]
    expected = alpha_bar_t * jax.nn.one_hot(x0, 4) + (1.0 - alpha_bar_t) * pi
    assert jnp.allclose(probs, expected, atol=1e-6, rtol=1e-6)


def test_multinomial_schedule_continuous_time_q():
    schedule = MultinomialDiffusionSchedule.from_cosine(
        num_steps=16,
        num_classes=4,
    )
    x0 = jnp.array([[0, 2], [1, 3]], dtype=jnp.int32)
    t = jnp.full((2, 1), 0.5, dtype=jnp.float32)

    probs = schedule.q_xt_given_x0_probs(x0, t)
    alpha_bar_t = schedule.alpha_bar(t)[..., None]
    expected = (
        alpha_bar_t * jax.nn.one_hot(x0, 4) + (1.0 - alpha_bar_t) * schedule.base_probs
    )

    assert probs.shape == x0.shape + (4,)
    assert jnp.allclose(probs, expected, atol=1e-6, rtol=1e-6)


def test_multinomial_time_bounds_and_float_only():
    schedule = MultinomialDiffusionSchedule.from_linear(
        num_steps=10,
        num_classes=4,
        t_min=0.2,
        t_max=0.9,
    )

    t = schedule.sample_timesteps(jax.random.key(0), (128, 1))
    assert jnp.logical_and(t >= 0.2, t <= 0.9).all()

    x0 = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    probs = schedule.q_xt_given_x0_probs(x0, jnp.full((2, 1), 0.55))
    assert probs.shape == x0.shape + (4,)

    try:
        _ = schedule.q_xt_given_x0_probs(x0, jnp.array([[5], [5]], dtype=jnp.int32))
        assert False, "Expected TypeError for integer time input."
    except TypeError:
        pass


def test_multinomial_posterior_mixture_normalized():
    schedule = MultinomialDiffusionSchedule.from_cosine(
        num_steps=10,
        num_classes=5,
    )
    x_t = jnp.array([[1, 2, 3], [0, 4, 1]], dtype=jnp.int32)
    x0_logits = jax.random.normal(jax.random.key(0), x_t.shape + (5,))
    x0_probs = jax.nn.softmax(x0_logits, axis=-1)
    t = jnp.array([[0.4], [0.7]], dtype=jnp.float32)
    t_prev = jnp.array([[0.3], [0.55]], dtype=jnp.float32)

    p = schedule.posterior_mixture_probs(x_t, x0_probs, t, t_prev)
    assert p.shape == x_t.shape + (5,)
    assert jnp.allclose(jnp.sum(p, axis=-1), 1.0, atol=1e-5, rtol=1e-5)
    assert jnp.isfinite(p).all()


def test_multinomial_diffusion_loss_and_sampling():
    class Net(nnx.Module):
        def __call__(self, t, x):
            del t
            return x

    schedule = MultinomialDiffusionSchedule.from_sigmoid(
        num_steps=6,
        num_classes=6,
    )
    model = MultinomialDiffusion(Net(), schedule=schedule)
    model_rb = MultinomialDiffusion(
        Net(),
        schedule=schedule,
        rao_blackwellize_xt=True,
        rao_blackwellize_xt_num_features=2,
    )

    x0 = jax.random.randint(jax.random.key(1), (16, 3), minval=0, maxval=6)
    loss = model.loss(jax.random.key(2), x0)
    loss_rb = model_rb.loss(jax.random.key(12), x0)
    assert loss.ndim == 0
    assert loss_rb.ndim == 0
    assert jnp.isfinite(loss)
    assert jnp.isfinite(loss_rb)

    samples = model.sample(jax.random.key(3), shape=(8, 3), num_sample_steps=4)
    assert samples.shape == (8, 3)
    assert jnp.logical_and(samples >= 0, samples < 6).all()

    cfg_u = UniformContinuousTimeTrainingConfig(
        num_steps=6,
        t_min=0.2,
        t_max=0.8,
        stratified=True,
        antithetic=True,
    )
    t_u = cfg_u.sample_times(jax.random.key(4), (64, 1))
    assert jnp.logical_and(t_u >= 0.2, t_u <= 0.8).all()
    assert jnp.allclose(jnp.mean(t_u), 0.5, atol=1e-2)

    cfg_i = ImportanceContinuousTimeTrainingConfig(
        schedule=schedule, t_min=0.2, t_max=0.8, stratified=True, antithetic=True
    )
    t_i = cfg_i.sample_times(jax.random.key(5), (64, 1))
    assert jnp.logical_and(t_i >= 0.2, t_i <= 0.8).all()


def test_uniform_time_sampler_stratification_bins():
    cfg = UniformContinuousTimeTrainingConfig(
        num_steps=10,
        t_min=0.0,
        t_max=1.0,
        stratified=True,
        antithetic=False,
    )
    t = cfg.sample_times(jax.random.key(13), (32,))
    t_sorted = jnp.sort(t)
    lower = jnp.arange(32, dtype=jnp.float32) / 32.0
    upper = (jnp.arange(32, dtype=jnp.float32) + 1.0) / 32.0
    assert jnp.logical_and(t_sorted >= lower, t_sorted < upper).all()


def test_multinomial_reparameterized_score_is_finite():
    class Net(nnx.Module):
        def __call__(self, t, x):
            return 0.5 * x + 0.1 * t[..., None]

    schedule = MultinomialDiffusionSchedule.from_logsnr(
        num_steps=12,
        num_classes=7,
    )
    model = MultinomialDiffusion(Net(), schedule=schedule)

    x_t = jax.random.randint(jax.random.key(10), (6, 4), minval=0, maxval=7)
    t = jnp.full((6, 1), 0.5, dtype=jnp.float32)
    score = model.score_reparameterized(
        jax.random.key(11),
        t,
        x_t,
        temperature=0.7,
        num_samples=3,
    )

    assert score.shape == x_t.shape + (7,)
    assert jnp.isfinite(score).all()


def test_categorical_edm_preconditioning_well_behaved():
    schedule = MultinomialDiffusionSchedule.from_cosine(
        num_steps=20,
        num_classes=5,
    )
    precond = CategoricalEDMPreconditioning()
    t = jnp.linspace(1.0 / 20.0, 1.0, 20, dtype=jnp.float32)

    c_in = precond.c_in(t, schedule)
    c_out = precond.c_out(t, schedule)
    c_skip = precond.c_skip(t, schedule)
    weight = precond.weight_ce(t, schedule)

    assert jnp.isfinite(c_in).all()
    assert jnp.isfinite(c_out).all()
    assert jnp.isfinite(c_skip).all()
    assert jnp.isfinite(weight).all()
    assert jnp.logical_and(c_skip >= 0.0, c_skip <= 1.0).all()
    assert jnp.logical_and(
        weight >= precond.min_weight, weight <= precond.max_weight
    ).all()

import jax
import jax.numpy as jnp
import pytest
import itertools

from probjax.nn.generative.diffusion import (
    build_denoising_loss,
    build_denoising_score_matching_loss,
    build_time_dependent_denoising_loss,
)
from probjax.nn.generative.flow_matching import build_flow_matching_loss
from probjax.nn.generative.discrete import (
    build_time_dependent_multinomial_diffusion_loss,
)
from probjax.nn.losses import (
    build_score_matching_loss,
    build_sliced_score_matching_loss,
    build_target_score_matching_loss,
)
from probjax.nn.generative.flow_matching.config import (
    AutodiffInterpolationSchedule,
    LinearInterpolationSchedule,
)


def model_fn(x):
    return x


@pytest.mark.parametrize(
    "builder, kwargs",
    [
        (build_denoising_loss, {"std": 1.0, "scale": 1.0}),
        (build_denoising_score_matching_loss, {"std": 1.0}),
        (build_score_matching_loss, {}),
        (build_sliced_score_matching_loss, {"num_slices": 1}),
        (build_target_score_matching_loss, {"score_fn": lambda x: x, "std": 1.0}),
    ],
)
def test_loss_fn(builder, kwargs):
    loss_fn = builder(model_fn, **kwargs)
    assert callable(loss_fn), "loss_fn is not callable"

    batch_vectors = jax.random.normal(jax.random.key(0), (100, 10))
    loss = loss_fn(batch_vectors, rng=jax.random.key(0))
    assert loss.ndim == 0, "loss is not a scalar"
    assert jax.numpy.isfinite(loss), "loss is not finite"


def test_denoising_loss_noise_mask_only_noises_masked_positions():
    captured = {}

    def capture_model_fn(x):
        captured["x_noisy"] = x
        return x

    loss_fn = build_denoising_loss(
        capture_model_fn,
        std=0.5,
        scale=1.0,
        prediction_target="x0",
    )

    x0 = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    noise_mask = jnp.array(
        [
            [True, False, True, False],
            [False, False, False, False],
            [True, True, False, False],
        ],
        dtype=bool,
    )
    _ = loss_fn(x0, rng=jax.random.key(0), noise_mask=noise_mask)

    x_noisy = captured["x_noisy"]
    assert jnp.array_equal(x_noisy[~noise_mask], x0[~noise_mask])
    assert bool(jnp.any(jnp.abs(x_noisy[noise_mask] - x0[noise_mask]) > 1e-7))


def test_time_dependent_denoising_loss_noise_mask_broadcast():
    captured = {}

    def capture_model_fn(t, x):
        del t
        captured["x_noisy"] = x
        return x

    loss_fn = build_time_dependent_denoising_loss(
        capture_model_fn,
        scale_fn=lambda t: jnp.ones_like(t),
        std_fn=lambda t: 0.5 * jnp.ones_like(t),
        weight_fn=lambda t: jnp.ones_like(t),
        prediction_target="x0",
    )

    x0 = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    t = jnp.ones((3, 1), dtype=jnp.float32)
    noise_mask = jnp.array([[True], [False], [True]], dtype=bool)  # broadcast to (3, 4)
    full_mask = jnp.broadcast_to(noise_mask, x0.shape)

    _ = loss_fn(t, x0, rng=jax.random.key(1), noise_mask=noise_mask)
    x_noisy = captured["x_noisy"]

    assert jnp.array_equal(x_noisy[~full_mask], x0[~full_mask])
    assert bool(jnp.any(jnp.abs(x_noisy[full_mask] - x0[full_mask]) > 1e-7))


def test_time_dependent_denoising_loss_noise_mask_shape_validation():
    def model_fn(t, x):
        del t
        return x

    loss_fn = build_time_dependent_denoising_loss(
        model_fn,
        scale_fn=lambda t: jnp.ones_like(t),
        std_fn=lambda t: 0.5 * jnp.ones_like(t),
        weight_fn=lambda t: jnp.ones_like(t),
    )

    x0 = jnp.ones((3, 4), dtype=jnp.float32)
    t = jnp.ones((3, 1), dtype=jnp.float32)
    bad_mask = jnp.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="broadcastable"):
        _ = loss_fn(t, x0, rng=jax.random.key(2), noise_mask=bad_mask)


def test_time_dependent_denoising_loss_uses_loss_mask_for_noising():
    captured = {}

    def capture_model_fn(t, x):
        del t
        captured["x_noisy"] = x
        return x

    loss_fn = build_time_dependent_denoising_loss(
        capture_model_fn,
        scale_fn=lambda t: jnp.ones_like(t),
        std_fn=lambda t: 0.5 * jnp.ones_like(t),
        weight_fn=lambda t: jnp.ones_like(t),
        prediction_target="x0",
    )

    x0 = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    t = jnp.ones((3, 1), dtype=jnp.float32)
    # True means "masked out from loss", so these entries should not be noised.
    loss_mask = jnp.array(
        [
            [True, False, True, False],
            [True, True, True, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )

    _ = loss_fn(t, x0, rng=jax.random.key(3), loss_mask=loss_mask)
    x_noisy = captured["x_noisy"]
    contributing = ~loss_mask

    assert jnp.array_equal(x_noisy[loss_mask], x0[loss_mask])
    assert bool(jnp.any(jnp.abs(x_noisy[contributing] - x0[contributing]) > 1e-7))


def test_flow_matching_schedule_zero_loss():
    schedule = LinearInterpolationSchedule()

    def model_fn(t, xt, x0, x1):
        return x1 - x0

    loss_fn = build_flow_matching_loss(model_fn, schedule=schedule)
    x0 = jax.numpy.ones((4, 3))
    x1 = 2.0 * jax.numpy.ones((4, 3))
    t = 0.3 * jax.numpy.ones((4, 1))
    loss = loss_fn(t, x0, x1, x0, x1, rng=jax.random.key(0))

    assert jax.numpy.allclose(loss, 0.0, atol=1e-5), "loss should be ~0"


def test_flow_matching_schedule_noise_zero_loss():
    schedule = LinearInterpolationSchedule(
        noise_fn=lambda t, x0, x1: 0.1 * jax.numpy.ones_like(x0),
        noise_velocity_fn=lambda t, x0, x1: jax.numpy.zeros_like(x0),
    )

    def model_fn(t, xt, x0, x1):
        return x1 - x0

    loss_fn = build_flow_matching_loss(model_fn, schedule=schedule)
    x0 = jax.numpy.ones((2, 5))
    x1 = 3.0 * jax.numpy.ones((2, 5))
    t = 0.6 * jax.numpy.ones((2, 1))
    loss = loss_fn(t, x0, x1, x0, x1, rng=jax.random.key(1))

    assert jax.numpy.allclose(loss, 0.0, atol=1e-5), "loss should be ~0"


def test_flow_matcher_model_loss():
    # Regression: FlowMatcher.loss used to pass stale kwargs to
    # build_flow_matching_loss, raising TypeError.
    from flax import nnx

    from probjax.nn.generative.flow_matching.model import LinearFlow

    class TinyFlowNet(nnx.Module):
        def __init__(self, rngs):
            self.proj = nnx.Linear(3, 3, rngs=rngs)

        def __call__(self, t, x, **kwargs):
            return self.proj(x)

    model = LinearFlow(TinyFlowNet(nnx.Rngs(0)))
    data = jax.random.normal(jax.random.key(1), (8, 3))
    loss = model.loss(jax.random.key(2), data)

    assert loss.ndim == 0, "loss is not a scalar"
    assert jnp.isfinite(loss), "loss is not finite"


def test_flow_matching_autodiff_schedule():
    def interp_fn(t, x0, x1):
        tt = t**2
        return (1.0 - tt) * x0 + tt * x1

    def interp_velocity_fn(t, x0, x1):
        return (-2.0 * t) * x0 + (2.0 * t) * x1

    schedule = AutodiffInterpolationSchedule(
        interp_fn=interp_fn, interp_velocity_fn=interp_velocity_fn
    )

    def model_fn(t, xt, x0, x1):
        return interp_velocity_fn(t, x0, x1)

    loss_fn = build_flow_matching_loss(model_fn, schedule=schedule)
    x0 = jax.numpy.ones((3, 2))
    x1 = -jax.numpy.ones((3, 2))
    t = 0.4 * jax.numpy.ones((3, 1))
    loss = loss_fn(t, x0, x1, x0, x1, rng=jax.random.key(2))

    assert jax.numpy.allclose(loss, 0.0, atol=1e-5), "loss should be ~0"


def test_multinomial_diffusion_loss_builder():
    num_classes = 5

    def q_sample_fn(rng, x0, t):
        del t
        probs = jax.nn.one_hot(x0, num_classes, dtype=jnp.float32)
        return jax.random.categorical(rng, jnp.log(probs), axis=-1)

    def sample_t_fn(rng, shape):
        return jax.random.randint(rng, shape=shape, minval=1, maxval=4)

    def model_fn(t, x_t):
        del t
        return jax.nn.one_hot(x_t, num_classes, dtype=jnp.float32)

    loss_fn = build_time_dependent_multinomial_diffusion_loss(
        model_fn,
        q_sample_fn=q_sample_fn,
        sample_timesteps_fn=sample_t_fn,
        weight_fn=lambda t: jnp.ones_like(t),
    )

    x0 = jax.random.randint(jax.random.key(0), (32, 4), minval=0, maxval=num_classes)
    loss = loss_fn(x0, rng=jax.random.key(1))
    assert loss.ndim == 0
    assert jnp.isfinite(loss)


def test_multinomial_diffusion_loss_builder_rao_blackwellized():
    num_classes = 4
    base_probs = jnp.ones((num_classes,), dtype=jnp.float32) / num_classes

    def q_sample_fn(rng, x0, t):
        del rng, t
        return x0

    def q_xt_given_x0_probs_fn(x0, t):
        del t
        return jax.nn.one_hot(x0, num_classes, dtype=jnp.float32)

    def sample_t_fn(rng, shape):
        return jax.random.uniform(rng, shape=shape, minval=0.0, maxval=1.0)

    def model_fn(t, x_t):
        del t
        return jax.nn.one_hot(x_t, num_classes, dtype=jnp.float32)

    loss_fn = build_time_dependent_multinomial_diffusion_loss(
        model_fn,
        q_sample_fn=q_sample_fn,
        sample_timesteps_fn=sample_t_fn,
        q_xt_given_x0_probs_fn=q_xt_given_x0_probs_fn,
        base_probs=base_probs,
        rao_blackwellize_xt=True,
        rao_blackwellize_xt_num_samples=3,
        num_classes=num_classes,
    )

    x0 = jax.random.randint(jax.random.key(10), (16, 3), minval=0, maxval=num_classes)
    loss = loss_fn(x0, rng=jax.random.key(11))
    assert loss.ndim == 0
    assert jnp.isfinite(loss)


def test_multinomial_diffusion_loss_builder_rao_blackwellized_matches_exact_expectation():
    num_classes = 7
    base_probs = jnp.array([0.05, 0.1, 0.2, 0.25, 0.15, 0.15, 0.1], dtype=jnp.float32)
    base_probs = base_probs / jnp.sum(base_probs)

    x0 = jax.random.randint(jax.random.key(20), (64, 5), minval=0, maxval=num_classes)
    t = jnp.full((x0.shape[0], 1), 0.37, dtype=jnp.float32)

    def alpha_bar_fn(tt):
        return 0.1 + 0.8 * jnp.exp(-2.0 * tt)

    def q_xt_given_x0_probs_fn(x0_idx, tt):
        a = alpha_bar_fn(tt)[..., None]
        return a * jax.nn.one_hot(x0_idx, num_classes) + (1.0 - a) * base_probs

    def q_sample_fn(rng, x0_idx, tt):
        q = q_xt_given_x0_probs_fn(x0_idx, tt)
        return jax.random.categorical(rng, jnp.log(q), axis=-1)

    w = jax.random.normal(jax.random.key(21), (num_classes, num_classes)) * 0.7
    u = jax.random.normal(jax.random.key(22), (num_classes,)) * 0.3

    def model_fn(tt, x_t):
        tt = jnp.asarray(tt, dtype=jnp.float32)
        if tt.shape != x_t.shape:
            tt = jnp.broadcast_to(tt, x_t.shape)
        return w[x_t] + tt[..., None] * u

    loss_fn = build_time_dependent_multinomial_diffusion_loss(
        model_fn,
        q_sample_fn=q_sample_fn,
        sample_timesteps_fn=lambda rng, shape: jnp.broadcast_to(t, shape),
        q_xt_given_x0_probs_fn=q_xt_given_x0_probs_fn,
        alpha_bar_fn=alpha_bar_fn,
        base_probs=base_probs,
        rao_blackwellize_xt=True,
        rao_blackwellize_xt_num_samples=16,
        num_classes=num_classes,
    )

    xt_all = jnp.broadcast_to(
        jnp.arange(num_classes, dtype=jnp.int32), x0.shape + (num_classes,)
    )
    logits_all = jax.vmap(lambda xt: model_fn(t, xt), in_axes=-1, out_axes=-2)(xt_all)
    logp_all = jax.nn.log_softmax(logits_all, axis=-1)
    ce_all = -jnp.take_along_axis(logp_all, x0[..., None, None], axis=-1).squeeze(-1)
    q = q_xt_given_x0_probs_fn(x0, t)
    exact = jnp.mean(jnp.sum(q * ce_all, axis=-1))

    vals = []
    for i in range(128):
        vals.append(loss_fn(x0, rng=jax.random.fold_in(jax.random.key(23), i), t=t))
    vals = jnp.asarray(vals)
    estimate = jnp.mean(vals)

    assert jnp.abs(estimate - exact) < 5e-3


def test_multinomial_diffusion_loss_builder_rao_blackwellized_vector_coupled_matches_exact():
    num_classes = 4
    dim = 3
    base_probs = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    base_probs = base_probs / jnp.sum(base_probs)

    x0 = jax.random.randint(jax.random.key(30), (8, dim), minval=0, maxval=num_classes)
    t = jnp.full((x0.shape[0], 1), 0.45, dtype=jnp.float32)

    def alpha_bar_fn(tt):
        return jnp.full_like(tt, 0.65)

    def q_xt_given_x0_probs_fn(x0_idx, tt):
        a = alpha_bar_fn(tt)[..., None]
        return a * jax.nn.one_hot(x0_idx, num_classes) + (1.0 - a) * base_probs

    def q_sample_fn(rng, x0_idx, tt):
        q = q_xt_given_x0_probs_fn(x0_idx, tt)
        return jax.random.categorical(rng, jnp.log(q), axis=-1)

    def model_fn(tt, x_t):
        tt = jnp.asarray(tt, dtype=jnp.float32)
        if tt.shape != x_t.shape:
            tt = jnp.broadcast_to(tt, x_t.shape)
        x_oh = jax.nn.one_hot(x_t, num_classes, dtype=jnp.float32)
        ctx = jnp.sum(x_oh, axis=1, keepdims=True)
        return 0.7 * x_oh + 0.3 * ctx + 0.1 * tt[..., None]

    loss_fn = build_time_dependent_multinomial_diffusion_loss(
        model_fn,
        q_sample_fn=q_sample_fn,
        sample_timesteps_fn=lambda rng, shape: jnp.broadcast_to(t, shape),
        q_xt_given_x0_probs_fn=q_xt_given_x0_probs_fn,
        alpha_bar_fn=alpha_bar_fn,
        base_probs=base_probs,
        rao_blackwellize_xt=True,
        rao_blackwellize_xt_num_samples=4,
        num_classes=num_classes,
    )

    states = jnp.array(
        list(itertools.product(range(num_classes), repeat=dim)), dtype=jnp.int32
    )  # [S, D]
    s = states.shape[0]

    q_probs = q_xt_given_x0_probs_fn(x0, t)  # [B, D, K]
    q_pick = jnp.take_along_axis(
        q_probs[:, None, :, :],
        states[None, :, :, None],
        axis=-1,
    ).squeeze(-1)  # [B, S, D]
    q_state = jnp.prod(q_pick, axis=-1)  # [B, S]

    x_states = jnp.broadcast_to(states[:, None, :], (s, x0.shape[0], dim))
    logits_states = jax.vmap(lambda x_t_state: model_fn(t, x_t_state))(x_states)
    logp_states = jax.nn.log_softmax(logits_states, axis=-1)
    ce_states = -jnp.take_along_axis(
        logp_states, x0[None, :, :, None], axis=-1
    ).squeeze(-1)  # [S, B, D]
    exact = jnp.mean(jnp.sum(q_state.T[:, :, None] * ce_states, axis=0))

    vals = []
    for i in range(128):
        vals.append(loss_fn(x0, rng=jax.random.fold_in(jax.random.key(31), i), t=t))
    vals = jnp.asarray(vals)
    estimate = jnp.mean(vals)

    assert jnp.abs(estimate - exact) < 1e-2


def test_multinomial_diffusion_loss_builder_rao_blackwellized_vector_subset_matches_exact_in_expectation():
    num_classes = 4
    dim = 3
    base_probs = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    base_probs = base_probs / jnp.sum(base_probs)

    x0 = jax.random.randint(jax.random.key(40), (8, dim), minval=0, maxval=num_classes)
    t = jnp.full((x0.shape[0], 1), 0.45, dtype=jnp.float32)

    def alpha_bar_fn(tt):
        return jnp.full_like(tt, 0.65)

    def q_xt_given_x0_probs_fn(x0_idx, tt):
        a = alpha_bar_fn(tt)[..., None]
        return a * jax.nn.one_hot(x0_idx, num_classes) + (1.0 - a) * base_probs

    def q_sample_fn(rng, x0_idx, tt):
        q = q_xt_given_x0_probs_fn(x0_idx, tt)
        return jax.random.categorical(rng, jnp.log(q), axis=-1)

    def model_fn(tt, x_t):
        tt = jnp.asarray(tt, dtype=jnp.float32)
        if tt.shape != x_t.shape:
            tt = jnp.broadcast_to(tt, x_t.shape)
        x_oh = jax.nn.one_hot(x_t, num_classes, dtype=jnp.float32)
        ctx = jnp.sum(x_oh, axis=1, keepdims=True)
        return 0.7 * x_oh + 0.3 * ctx + 0.1 * tt[..., None]

    loss_fn = build_time_dependent_multinomial_diffusion_loss(
        model_fn,
        q_sample_fn=q_sample_fn,
        sample_timesteps_fn=lambda rng, shape: jnp.broadcast_to(t, shape),
        q_xt_given_x0_probs_fn=q_xt_given_x0_probs_fn,
        alpha_bar_fn=alpha_bar_fn,
        base_probs=base_probs,
        rao_blackwellize_xt=True,
        rao_blackwellize_xt_num_samples=8,
        rao_blackwellize_xt_num_features=2,
        num_classes=num_classes,
    )

    states = jnp.array(
        list(itertools.product(range(num_classes), repeat=dim)), dtype=jnp.int32
    )  # [S, D]
    s = states.shape[0]
    q_probs = q_xt_given_x0_probs_fn(x0, t)  # [B, D, K]
    q_pick = jnp.take_along_axis(
        q_probs[:, None, :, :],
        states[None, :, :, None],
        axis=-1,
    ).squeeze(-1)  # [B, S, D]
    q_state = jnp.prod(q_pick, axis=-1)  # [B, S]

    x_states = jnp.broadcast_to(states[:, None, :], (s, x0.shape[0], dim))
    logits_states = jax.vmap(lambda x_t_state: model_fn(t, x_t_state))(x_states)
    logp_states = jax.nn.log_softmax(logits_states, axis=-1)
    ce_states = -jnp.take_along_axis(
        logp_states, x0[None, :, :, None], axis=-1
    ).squeeze(-1)  # [S, B, D]
    exact = jnp.mean(jnp.sum(q_state.T[:, :, None] * ce_states, axis=0))

    vals = []
    for i in range(256):
        vals.append(loss_fn(x0, rng=jax.random.fold_in(jax.random.key(41), i), t=t))
    vals = jnp.asarray(vals)
    estimate = jnp.mean(vals)

    assert jnp.abs(estimate - exact) < 2e-2

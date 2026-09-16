import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.nn.generative.diffusion import EDM, VE, VP, CosineDM
from probjax.nn.generative.diffusion.config import (
    BaseSolverConfig,
    DDIMSolverConfig,
    VSolverConfig,
)


class Zero(nnx.Module):
    def __call__(self, t, x, **kwargs):
        return jnp.zeros_like(x)


@pytest.mark.parametrize('mode', ['ode', 'sde'])
def test_overridden_endpoint_controls_initial_noise(mode):
    model = EDM(Zero(), event_spec=2, num_steps=3)
    dist = model.as_dist(t_max=2.0, mode=mode)
    path = dist.sample_path(jax.random.key(0), (2048,))
    np.testing.assert_allclose(path[0].std(), jnp.sqrt(5.0), rtol=0.04)


def test_explicit_state_controls_initial_noise_and_is_jittable():
    model = EDM(Zero(), event_spec=2, num_steps=8)
    dist = model.as_dist(t_max=2.0)
    state = dist.model_state()
    expected = dist.sample(jax.random.key(0), (4,))
    model.std0.set_value(3.0)
    actual = jax.jit(lambda state, key: dist.sample_with_state(state, key, (4,)))(
        state, jax.random.key(0)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    assert model.std0.get_value() == 3.0


@pytest.mark.parametrize('cls', [EDM, VE, VP, CosineDM])
@pytest.mark.parametrize('solver_cls', [DDIMSolverConfig, VSolverConfig])
def test_solver_drift_matches_probability_flow_including_endpoints(cls, solver_cls):
    model = cls(Zero(), event_spec=2, std0=2.0)
    baseline = BaseSolverConfig(model.schedule).build_ode_drift(model)
    candidate = solver_cls(model.schedule).build_ode_drift(model)
    for t in [
        model.train_cfg.t_min,
        model.train_cfg.t_max,
        (model.train_cfg.t_min + model.train_cfg.t_max) / 2,
    ]:
        x = jnp.ones(2)
        np.testing.assert_allclose(
            candidate(jnp.array([t]), x),
            baseline(jnp.array([t]), x),
            rtol=0.002,
            atol=2e-5,
        )


@pytest.mark.parametrize('architecture', ['mlp', 'transformer'])
def test_time_vector_broadcast_matches_explicit_column(architecture):
    from probjax.nn import DiffusionTransformer, TimeMLP

    if architecture == 'mlp':
        model = TimeMLP(
            3, hidden_dim=8, depth=1, time_embed_dim=8, fourier_dim=8, rngs=nnx.Rngs(0)
        )
        x = jnp.ones((2, 4, 3))
    else:
        model = DiffusionTransformer(
            3,
            model_dim=8,
            num_layers=1,
            num_heads=1,
            attn_size=4,
            time_embed_dim=8,
            fourier_dim=8,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((2, 4, 5, 3))
    t = jnp.linspace(0.1, 0.9, 4)
    expected = model(t[None, :, None], x)
    actual = nnx.jit(lambda m, t, x: m(t, x))(model, t, x)
    np.testing.assert_allclose(actual, expected, atol=2e-6)


def test_bfloat16_position_encoding_has_no_unsafe_cast():
    import warnings

    from probjax.nn.layers.encoding import PosEncode

    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        output = PosEncode(rngs=nnx.Rngs(0))(jnp.ones((3, 7), dtype=jnp.bfloat16))
    assert output.dtype == jnp.bfloat16

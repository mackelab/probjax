"""The shared kernel API: configuration, per-algorithm types, callability.

Every algorithm family -- MCMC, SMC, filtering, ODE and SDE -- builds its
kernels the same way. These tests hold that shape in place.
"""

import jax
import jax.numpy as jnp
import pytest

from probjax.inference.base import Kernel
from probjax.inference.filtering.base import FilterKernel
from probjax.inference.filtering.kalman_filter import kalman_filter
from probjax.inference.filtering.particle_filter import ParticleFilter
from probjax.inference.mcmc.hmc import hmc
from probjax.inference.mcmc.mala import mala
from probjax.inference.mcmc.pmmcmc import pseudo_marginal
from probjax.inference.smc.path import GeometricPath
from probjax.inference.smc.path_smc import path_smc
from probjax.utils.jaxutils import Configured, KernelAPI
from probjax.utils.odeutil.solvers.base import ODESolver
from probjax.utils.odeutil.solvers.rk_implicit import implicit_euler
from probjax.utils.sdeutil.base import SDESolver
from probjax.utils.sdeutil.solver.milstein import milstein


def _gaussian_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def _identity_transition(t_old, t):
    return jnp.eye(2), jnp.eye(2)


def _identity_observation(t):
    return jnp.eye(2), jnp.eye(2)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "api",
    [hmc, mala, kalman_filter, ParticleFilter, implicit_euler, milstein],
    ids=[
        "hmc",
        "mala",
        "kalman_filter",
        "ParticleFilter",
        "implicit_euler",
        "milstein",
    ],
)
def test_every_api_class_can_configure_options(api):
    configured = api.configure(num_integration_steps=3)
    assert isinstance(configured, Configured)
    assert configured.api is api
    assert dict(configured.options) == {"num_integration_steps": 3}


def test_configured_is_a_hashable_value():
    configured = hmc.configure(num_integration_steps=3)
    assert configured == hmc.configure(num_integration_steps=3)
    assert hash(configured) == hash(hmc.configure(num_integration_steps=3))
    assert configured != hmc.configure(num_integration_steps=4)


def test_configured_builds_the_kernel_when_the_model_arrives():
    configured = hmc.configure(num_integration_steps=3)
    kernel = configured(_gaussian_logdensity)
    assert isinstance(kernel, Kernel)
    assert type(kernel).__name__ == "hmc"
    # The configured options reached the step builder.
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)
    assert jnp.isfinite(params.step_size)


def test_a_single_model_function_also_has_the_shorthand():
    # The log-density is the only required argument, so options alone are
    # unambiguous and the shorthand is allowed.
    assert hmc(num_integration_steps=3) == hmc.configure(num_integration_steps=3)


def test_bound_options_are_routed_to_the_function_that_accepts_them():
    configured = hmc.configure(num_integration_steps=3, step_size=0.25)
    # ``step_size`` belongs to init_params, not to build_step.
    state = configured(_gaussian_logdensity).init(jax.random.key(0), jnp.zeros(2))
    params = configured.init_params(state)
    assert float(params.step_size) == pytest.approx(0.25)


def test_configured_carries_the_api_metadata():
    assert hmc.configure().default_adaptation_fn is not None
    assert hmc.configure().info == {}


# ---------------------------------------------------------------------------
# Per-algorithm kernel types
# ---------------------------------------------------------------------------


def test_built_mcmc_kernel_carries_its_algorithm_type():
    kernel = hmc(_gaussian_logdensity)
    assert type(kernel).__name__ == "hmc"
    assert isinstance(kernel, Kernel)
    assert type(kernel).__doc__ == hmc.__doc__


def test_built_filter_kernel_carries_its_algorithm_type():
    kernel = kalman_filter(_identity_transition, _identity_observation)
    assert type(kernel).__name__ == "kalman_filter"
    assert isinstance(kernel, FilterKernel)
    assert type(kernel).__doc__ == kalman_filter.__doc__


def test_built_ode_solver_carries_its_algorithm_type():
    solver = implicit_euler(lambda t, y: -y)
    assert type(solver).__name__ == "implicit_euler"
    assert isinstance(solver, ODESolver)


def test_built_sde_solver_carries_its_algorithm_type():
    solver = milstein(lambda t, y: -y, lambda t, y: 0.1 * y)
    assert type(solver).__name__ == "milstein"
    assert isinstance(solver, SDESolver)


# ---------------------------------------------------------------------------
# build_step is the canonical name
# ---------------------------------------------------------------------------


def test_filters_expose_build_step():
    assert not hasattr(kalman_filter, "build_kernel")
    kernel = kalman_filter.build_step(_identity_transition, _identity_observation)
    assert callable(kernel)


def test_the_api_base_class_is_shared():
    from probjax.inference.filtering.base import FilterAPI
    from probjax.inference.mcmc.base import MarkovKernelAPI
    from probjax.inference.smc.base import SMCKernelAPI
    from probjax.utils.odeutil.solvers.base import ODESolverAPI
    from probjax.utils.sdeutil.base import SDESolverAPI

    for api in (
        MarkovKernelAPI,
        SMCKernelAPI,
        FilterAPI,
        ODESolverAPI,
        SDESolverAPI,
    ):
        assert issubclass(api, KernelAPI)


# ---------------------------------------------------------------------------
# Configured kernels compose
# ---------------------------------------------------------------------------


def test_a_configured_kernel_can_drive_an_smc_kernel():
    configured = hmc.configure(num_integration_steps=3)
    kernel = path_smc(
        logprior_fn=_gaussian_logdensity,
        loglikelihood_fn=lambda x: -0.5 * jnp.sum((x - 1.0) ** 2),
        path=GeometricPath(),
        mcmc_kernel=configured,
    )
    assert type(kernel).__name__ == "path_smc"

    particles = jnp.zeros((8, 2))
    params = kernel.init_params(particles, path=GeometricPath(), mcmc_kernel=configured)
    assert set(params) == {"step_size", "inverse_mass_matrix"}


def test_configured_and_bare_kernel_agree():
    configured = hmc.configure(num_integration_steps=3)
    particles = jnp.zeros((8, 2))
    from_configured = path_smc(
        logprior_fn=_gaussian_logdensity,
        loglikelihood_fn=lambda x: -0.5 * jnp.sum((x - 1.0) ** 2),
        path=GeometricPath(),
        mcmc_kernel=configured,
    ).init_params(particles, path=GeometricPath(), mcmc_kernel=configured)
    from_kwargs = path_smc(
        logprior_fn=_gaussian_logdensity,
        loglikelihood_fn=lambda x: -0.5 * jnp.sum((x - 1.0) ** 2),
        path=GeometricPath(),
        mcmc_kernel=hmc,
        num_integration_steps=3,
    ).init_params(particles, path=GeometricPath(), mcmc_kernel=hmc)

    for name in from_configured:
        assert jnp.allclose(from_configured[name], from_kwargs[name])


def test_pseudo_marginal_accepts_a_configured_kernel():
    def noisy(x, key):
        return _gaussian_logdensity(x) + 0.01 * jax.random.normal(key)

    kernel = pseudo_marginal(hmc.configure(num_integration_steps=2), noisy)
    assert type(kernel).__name__ == "pseudo_marginal"
    assert kernel.default_adaptation is not None

    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)
    assert state.position.shape == (2,)
    assert jnp.isfinite(params.step_size)


# ---------------------------------------------------------------------------
# Runners share the progress machinery
# ---------------------------------------------------------------------------


def test_verbose_and_quiet_filters_agree():
    import numpy as np

    from probjax.inference.filter_smooth import Filter

    transition = jnp.array([[1.0, 0.1], [0.0, 0.9]])
    observation = jnp.array([[1.0, 0.0]])
    kernel = kalman_filter(
        lambda t_old, t: (transition, jnp.eye(2) * 0.1),
        lambda t: (observation, jnp.eye(1) * 0.5),
    )
    ts = jnp.arange(6, dtype=float)
    ys = jnp.array([[0.1], [0.3], [0.2], [0.4], [0.35], [0.5]])

    quiet = Filter(kernel, verbose=False).filter(
        jax.random.key(0), ts, ys, jnp.zeros(2), jnp.eye(2)
    )
    loud = Filter(kernel, verbose=True).filter(
        jax.random.key(0), ts, ys, jnp.zeros(2), jnp.eye(2)
    )
    assert np.allclose(quiet.states.mean, loud.states.mean)
    assert np.allclose(quiet.get_log_likelihood(), loud.get_log_likelihood())


def test_verbose_and_quiet_mcmc_agree():
    import numpy as np

    from probjax.inference import run

    kernel = mala(_gaussian_logdensity)
    state = kernel.init(jax.random.key(0), jnp.zeros(2))
    params = kernel.init_params(state)

    quiet = run(jax.random.key(1), kernel, state, params, 5)
    loud = run(jax.random.key(1), kernel, state, params, 5, progress=True)
    assert np.allclose(quiet.state.position, loud.state.position)


# ---------------------------------------------------------------------------
# Kernels are callable
# ---------------------------------------------------------------------------


def test_temporal_filter_is_callable():
    from probjax.inference.filtering.state_space import TemporalFilter

    calls = []

    def step(key, state, theta, t, y, observed=True):
        calls.append(t)
        return state, None

    filt = TemporalFilter(lambda *a: None, step, "exact")
    filt(jax.random.key(0), "state", None, 1.0, None)
    assert calls == [1.0]


def test_temporal_smc_is_callable():
    from probjax.inference.smc.temporal import TemporalSMC

    calls = []

    def step(key, state, *args, **kwargs):
        calls.append(state)
        return state, None

    kernel = TemporalSMC(lambda *a: None, step)
    kernel(jax.random.key(0), "state")
    assert calls == ["state"]

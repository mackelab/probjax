"""Regression tests for solver accuracy: tableau orders, dtype, adaptive terminal
state and Stratonovich handling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.utils.functions import additive_diffusion
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint
from probjax.utils.sdeutil.adaptive import StrongStepSizeAdaptor, WeakStepSizeAdaptor
from probjax.utils.sdeutil.brownian import (
    brownian_tree,
    get_iterated_integrals_fn,
)
from probjax.utils.sdeutil.solver.milstein import MilsteinState, build_milstein_step

# Fixed-step explicit methods and their classical order.
FIXED_STEP_ORDERS = {
    "euler": 1,
    "heun": 2,
    "midpoint": 2,
    "ralston": 2,
    "rk3": 3,
    "bosh3": 3,
    "heun3": 3,
    "vanderhouwen": 3,
    "ralston3": 3,
    "ssprk3": 3,
    "rk4": 4,
    "ralston4": 4,
    "rk38": 4,
    "rk6": 6,
}


def _pendulum(t, y):
    return jnp.array([y[1], -jnp.sin(y[0]) + 0.3 * jnp.cos(t) * y[0]])


@pytest.mark.parametrize("method,order", FIXED_STEP_ORDERS.items())
def test_fixed_step_methods_converge_at_their_order(method, order):
    y0 = jnp.array([1.0, 0.5])

    def solve(method, n):
        return odeint(
            _pendulum,
            y0,
            jnp.linspace(0, 1, n + 1),
            solver=method,
            dtype=jnp.float64,
            collect_trace=False,
        )

    with jax.enable_x64(True):
        # rk6 reaches round-off quickly; coarser grids keep its error measurable.
        n_coarse, n_fine = (2, 4) if order == 6 else (8, 16)
        reference = solve("rk6", 256) if order != 6 else solve("rk38", 4096)
        errors = [
            float(jnp.linalg.norm(solve(method, n) - reference))
            for n in (n_coarse, n_fine)
        ]
    measured_order = np.log2(errors[0] / errors[1])
    assert measured_order > order - 0.35, (method, measured_order)


def test_tableau_coefficients_follow_requested_dtype():
    # float32 coefficients would floor the RK4 error at ~1e-8.
    with jax.enable_x64(True):
        y = odeint(
            lambda t, y: -y,
            jnp.array([1.0]),
            jnp.linspace(0, 1, 65),
            solver="rk4",
            dtype=jnp.float64,
            collect_trace=False,
        )[0]
    assert abs(float(y) - np.exp(-1.0)) < 1e-9


@pytest.mark.parametrize("method", ["dopri5", "tsit5", "bosh3"])
def test_adaptive_terminal_state_matches_trace_end(method):
    ts = jnp.linspace(0, 1, 17)
    y0 = jnp.array([1.0])
    drift = lambda t, y: -y  # noqa: E731
    terminal = odeint(drift, y0, ts, solver=method, collect_trace=False)
    trace = odeint(drift, y0, ts, solver=method)
    np.testing.assert_allclose(terminal, trace[-1], rtol=1e-6)
    np.testing.assert_allclose(terminal, np.exp(-1.0), rtol=1e-4)


def _gbm_strong_error(n_steps, sde_type, n_paths=200):
    mu, sigma = 0.5, 0.8
    ts = jnp.linspace(0, 1, n_steps + 1)
    y0 = jnp.ones(1)

    def one_path(key):
        ys, aux = sdeint(
            key,
            lambda t, y: mu * y,
            lambda t, y: sigma * y,
            y0,
            ts,
            solver="milstein",
            sde_type=sde_type,
            dtype=jnp.float64,
            return_aux=True,
        )
        dW = aux.brownian
        drift_correction = -(sigma**2) / 2 if sde_type == "ito" else 0.0
        exact = jnp.exp((mu + drift_correction) + sigma * dW.sum(0))
        return jnp.abs(ys[-1] - exact)[0]

    keys = jax.random.split(jax.random.key(0), n_paths)
    return float(jax.vmap(one_path)(keys).mean())


@pytest.mark.parametrize("sde_type", ["ito", "stratonovich"])
def test_milstein_converges_for_both_sde_types(sde_type):
    with jax.enable_x64(True):
        coarse, fine = _gbm_strong_error(16, sde_type), _gbm_strong_error(64, sde_type)
    # Strong order 1: 4x finer grid should cut the error by well over 2x.
    assert fine < coarse / 2.5


def test_stratonovich_requires_milstein_for_multiplicative_noise():
    args = (
        jax.random.key(0),
        lambda t, y: y,
        lambda t, y: y,
        jnp.ones(1),
        jnp.linspace(0, 1, 5),
    )
    with pytest.raises(NotImplementedError, match="Stratonovich|stratonovich"):
        sdeint(*args, sde_type="stratonovich")
    with pytest.raises(NotImplementedError):
        sdeint(*args, sde_type="stratonovich", solver="sri1")
    with pytest.raises(NotImplementedError):
        sdeint(
            *args,
            sde_type="stratonovich",
            solver="adaptive",
            step_size_adaptor=WeakStepSizeAdaptor(),
        )


def test_stratonovich_allowed_for_additive_noise():
    out = sdeint(
        jax.random.key(0),
        lambda t, y: -y,
        additive_diffusion(lambda t: jnp.ones(1)),
        jnp.ones(1),
        jnp.linspace(0, 1, 5),
        sde_type="stratonovich",
    )
    assert out.shape == (5, 1)


def test_brownian_tree_has_brownian_statistics():
    keys = jax.random.split(jax.random.key(0), 20_000)
    tree = lambda key, t: brownian_tree(key, t, 0.0, 1.0, jnp.zeros(1), depth=20)  # noqa: E731
    w_a = jax.vmap(lambda k: tree(k, 0.3))(keys)[:, 0]
    w_b = jax.vmap(lambda k: tree(k, 0.5))(keys)[:, 0]
    w_c = jax.vmap(lambda k: tree(k, 0.8))(keys)[:, 0]
    w_small = jax.vmap(lambda k: tree(k, 0.3 + 1e-4))(keys)[:, 0]
    np.testing.assert_allclose(w_a.var(), 0.3, rtol=0.05)
    np.testing.assert_allclose(w_c.var(), 0.8, rtol=0.05)
    np.testing.assert_allclose((w_small - w_a).var(), 1e-4, rtol=0.05)
    assert abs(np.corrcoef(w_b - w_a, w_c - w_b)[0, 1]) < 0.03  # independent increments


_GBM = dict(mu=0.5, sigma=0.8)  # E[y_1] = 1.649, Var[y_1] = 2.437


def _gbm_terminal(adaptor, n_paths, rng=1):
    mu, sigma = _GBM["mu"], _GBM["sigma"]
    keys = jax.random.split(jax.random.key(rng), n_paths)
    return jax.vmap(
        lambda k: sdeint(
            k,
            lambda t, y: mu * y,
            lambda t, y: sigma * y,
            jnp.ones(1),
            jnp.linspace(0.0, 1.0, 3),
            solver="adaptive",
            step_size_adaptor=adaptor,
            collect_trace=False,
        )
    )(keys)[:, 0]


@pytest.mark.parametrize("adaptor_cls", [WeakStepSizeAdaptor, StrongStepSizeAdaptor])
def test_adaptive_sde_keeps_the_variance(adaptor_cls):
    # Rejecting steps on their own noise used to shrink the variance (~0.7 vs 2.4).
    adaptor = adaptor_cls(rtol=1e-3, atol=1e-4, max_inner_steps=1024)
    terminal = _gbm_terminal(adaptor, 3000)
    assert abs(terminal.mean() - 1.649) < 0.15
    assert terminal.var() > 1.7


def test_strong_adaptor_converges_pathwise():
    mu, sigma = _GBM["mu"], _GBM["sigma"]
    keys = jax.random.split(jax.random.key(0), 200)
    exact = jax.vmap(
        lambda k: jnp.exp(
            mu
            - sigma**2 / 2
            + sigma * brownian_tree(k, 1.0, 0.0, 1.0, jnp.zeros(1), depth=20)
        )
    )(keys)
    adaptor = StrongStepSizeAdaptor(rtol=1e-3, atol=1e-4, max_inner_steps=1024)
    terminal = jax.vmap(
        lambda k: sdeint(
            k,
            lambda t, y: mu * y,
            lambda t, y: sigma * y,
            jnp.ones(1),
            jnp.linspace(0.0, 1.0, 3),
            solver="adaptive",
            step_size_adaptor=adaptor,
            collect_trace=False,
        )
    )(keys)
    # The look-ahead rejection rule left a systematic +0.25 bias at any tolerance.
    assert abs(float((terminal - exact).mean())) < 0.06


@pytest.mark.parametrize("sde_type", ["ito", "stratonovich"])
def test_commutative_noise_milstein_converges(sde_type):
    mu, s = 0.5, jnp.array([0.6, 0.5])  # scalar state driven by two Brownian motions

    def strong_error(n_steps, n_paths=200):
        step = build_milstein_step(
            lambda t, y: mu * y,
            lambda t, y: (y[0] * s)[None, :],
            noise_type="commutative",
            noise_dim=2,
            sde_type=sde_type,
        )

        def one_path(key):
            def body(carry, k):
                state, w = carry
                state, info = step(k, state, 1.0 / n_steps)
                return (state, w + info.dWt), None

            init = (MilsteinState(jnp.array(0.0), jnp.ones(1)), jnp.zeros(2))
            (state, w), _ = jax.lax.scan(body, init, jax.random.split(key, n_steps))
            correction = -(s @ s) / 2 if sde_type == "ito" else 0.0
            return jnp.abs(state.y0[0] - jnp.exp(mu + correction + s @ w))

        keys = jax.random.split(jax.random.key(0), n_paths)
        return float(jax.vmap(one_path)(keys).mean())

    assert strong_error(64) < strong_error(8) / 4  # strong order ~1 (0.5 would give ~2.8)


def test_commutative_iterated_integrals():
    dW, dt = jnp.array([0.3, -0.2]), 0.1
    ito = get_iterated_integrals_fn("commutative", "ito")(None, dW, dt)
    strat = get_iterated_integrals_fn("commutative", "stratonovich")(None, dW, dt)
    np.testing.assert_allclose(ito, 0.5 * (np.outer(dW, dW) - dt * np.eye(2)), atol=1e-6)
    np.testing.assert_allclose(strat, 0.5 * np.outer(dW, dW), atol=1e-6)


def test_srk_is_an_alias_of_sri1():
    args = (
        jax.random.key(0),
        lambda t, y: 0.5 * y,
        lambda t, y: 0.8 * y,
        jnp.ones(1),
        jnp.linspace(0.0, 1.0, 20),
    )
    np.testing.assert_allclose(sdeint(*args, solver="srk"), sdeint(*args, solver="sri1"))

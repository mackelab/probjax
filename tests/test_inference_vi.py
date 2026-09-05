"""Flow VI and NeuTra.

The property that matters for NeuTra is that it is a *change of variables*, not
an approximation: MCMC on the transformed target stays asymptotically exact
however poor the flow is. So the tests split into two kinds -- exact identities,
checked to tolerance, and efficiency claims, checked as inequalities with margin.
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from blackjax.base import VIAlgorithm
from flax import nnx

from probjax.inference import MCMC, flow_vi, hmc, mala, neutra, nuts
from probjax.inference.vi import rebuild
from probjax.nn import maf, nsf

DIM = 3


def correlated_gaussian():
    """A target with strong correlation, where a mean-field family would fail."""
    mean = jnp.array([2.0, -1.0, 0.5])
    cov = jnp.array([[1.0, 0.95, 0.3], [0.95, 1.0, 0.2], [0.3, 0.2, 1.0]])
    precision = jnp.linalg.inv(cov)

    def logdensity(x):
        delta = x - mean
        return -0.5 * delta @ precision @ delta

    return logdensity, mean, cov


def funnel_logdensity(position):
    """Neal's funnel: the standard example of geometry defeating a sampler."""
    v, x = position[0], position[1:]
    return jax.scipy.stats.norm.logpdf(v, 0.0, 3.0).sum() + jnp.sum(
        jax.scipy.stats.norm.logpdf(x, 0.0, jnp.exp(v / 2))
    )


def fit_flow(logdensity, flow, num_steps=1200, seed=0, num_samples=128):
    algorithm = flow_vi(logdensity, flow, optax.adam(1e-3), num_samples=num_samples)

    def one(state, key):
        state, info = algorithm.step(key, state)
        return state, info.elbo

    state, objective = jax.lax.scan(
        jax.jit(one),
        algorithm.init(),
        jax.random.split(jax.random.key(seed), num_steps),
    )
    return algorithm, state, objective


def effective_sample_size(chain):
    """Initial-positive-sequence ESS, per dimension."""
    n = chain.shape[0]
    centred = chain - chain.mean(0)
    variance = (centred**2).mean(0)
    out = []
    for d in range(chain.shape[1]):
        acf = jnp.correlate(centred[:, d], centred[:, d], "full")[n - 1 :] / (
            n * variance[d]
        )
        below = jnp.where(acf < 0.05, jnp.arange(acf.size), acf.size)
        cutoff = int(jnp.min(below))
        out.append(n / (1 + 2 * float(jnp.sum(acf[1:cutoff]))))
    return jnp.asarray(out)


def run_chain(logdensity, initial, num_steps=4000, seed=0, kernel_fn=nuts):
    kernel = kernel_fn(logdensity)
    state = kernel.init(jax.random.key(seed), initial)
    params = kernel.init_params(state)
    return MCMC(kernel).sample(jax.random.key(seed + 1), state, num_steps, params)


# ---------------------------------------------------------------------------
# NeuTra is an exact change of variables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctor", [maf, nsf], ids=["maf", "nsf"])
def test_transformed_logdensity_matches_a_brute_force_jacobian(ctor):
    """The one place a sign error would hide, and it is checkable exactly.

    The implementation never forms a Jacobian -- it uses
    ``log|det J_T(z)| = log N(z) - log q(T(z))`` -- so this pins that identity
    against the determinant computed directly.
    """
    logdensity, *_ = correlated_gaussian()
    flow = ctor(DIM, 3, rngs=nnx.Rngs(0))
    flow.fit(
        jax.random.key(1),
        jax.random.normal(jax.random.key(2), (256, DIM)) * 2.0 + 1.0,
        num_steps=60,
        batch_size=64,
    )
    transform = neutra(logdensity, flow)

    forward = flow.transform
    for seed in range(3):
        z = jax.random.normal(jax.random.key(seed), (DIM,))
        jacobian = jax.jacobian(forward)(z)
        expected = logdensity(forward(z)) + jnp.log(jnp.abs(jnp.linalg.det(jacobian)))
        assert float(transform.logdensity(z)) == pytest.approx(
            float(expected), abs=1e-3
        )


def test_a_well_fitted_flow_turns_the_target_into_a_standard_normal():
    """The defining property: if q == p then log p~(z) == log N(z) + const."""
    logdensity, mean, cov = correlated_gaussian()
    flow = nsf(DIM, 4, rngs=nnx.Rngs(0))
    _, state, _ = fit_flow(logdensity, flow)
    fitted = rebuild(flow, state)
    transform = neutra(logdensity, fitted)

    zs = jax.random.normal(jax.random.key(9), (64, DIM))
    transformed = jax.vmap(transform.logdensity)(zs)
    reference = jax.vmap(lambda z: jnp.sum(jax.scipy.stats.norm.logpdf(z)))(zs)
    # Equal up to an additive constant, so the spread of the difference is what
    # matters. A perfect flow gives zero; loose enough for an imperfect fit.
    assert float(jnp.std(transformed - reference)) < 1.0


def test_forward_accepts_a_single_position_and_a_batch():
    logdensity, *_ = correlated_gaussian()
    flow = maf(DIM, 2, rngs=nnx.Rngs(0))
    transform = neutra(logdensity, flow)

    assert transform.forward(jnp.zeros(DIM)).shape == (DIM,)
    assert transform.forward(jnp.zeros((7, DIM))).shape == (7, DIM)


# ---------------------------------------------------------------------------
# Flow VI
# ---------------------------------------------------------------------------


def test_flow_vi_is_a_blackjax_vi_algorithm():
    logdensity, *_ = correlated_gaussian()
    algorithm = flow_vi(logdensity, maf(DIM, 2, rngs=nnx.Rngs(0)), optax.adam(1e-3))
    assert isinstance(algorithm, VIAlgorithm)

    state = algorithm.init()
    state, info = algorithm.step(jax.random.key(0), state)
    assert jnp.isfinite(info.elbo)
    assert algorithm.sample(jax.random.key(1), state, 5).shape == (5, DIM)


def test_flow_vi_recovers_a_correlated_gaussian():
    logdensity, mean, cov = correlated_gaussian()
    flow = maf(DIM, 4, rngs=nnx.Rngs(0))
    algorithm, state, objective = fit_flow(logdensity, flow)

    assert float(objective[-1]) < float(objective[0])

    draws = algorithm.sample(jax.random.key(7), state, 4000)
    assert float(jnp.max(jnp.abs(draws.mean(0) - mean))) < 0.15
    assert float(jnp.max(jnp.abs(jnp.cov(draws.T) - cov))) < 0.25


def test_the_objective_goes_down_and_is_the_negative_elbo():
    """`info.elbo` holds mean(log q - log p), matching blackjax's field name.

    It is the quantity being minimised, so it must decrease -- the name is
    kept for parity with ``MFVIInfo`` but the sign is the opposite of what it
    suggests.
    """
    logdensity, *_ = correlated_gaussian()
    _, _, objective = fit_flow(logdensity, maf(DIM, 3, rngs=nnx.Rngs(0)), num_steps=400)
    assert float(jnp.mean(objective[-50:])) < float(jnp.mean(objective[:50]))


@pytest.mark.parametrize("stl", [True, False], ids=["stl", "no_stl"])
def test_both_gradient_estimators_train(stl):
    logdensity, *_ = correlated_gaussian()
    flow = maf(DIM, 3, rngs=nnx.Rngs(0))
    algorithm = flow_vi(
        logdensity, flow, optax.adam(1e-3), num_samples=64, stl_estimator=stl
    )

    def one(state, key):
        state, info = algorithm.step(key, state)
        return state, info.elbo

    _, objective = jax.lax.scan(
        jax.jit(one), algorithm.init(), jax.random.split(jax.random.key(0), 300)
    )
    assert jnp.all(jnp.isfinite(objective))
    assert float(jnp.mean(objective[-50:])) < float(jnp.mean(objective[:50]))


# ---------------------------------------------------------------------------
# NeuTra with the existing samplers
# ---------------------------------------------------------------------------


def test_neutra_improves_mixing_on_the_funnel():
    """The efficiency claim. Measured 3.6x; asserted at 1.5x for headroom.

    Note this is a statement about *mixing*, not accuracy. Reverse-KL VI is
    mode-seeking and the fitted flow under-covers the funnel's neck, so neither
    chain is unbiased at this budget -- what NeuTra buys is that the chain
    explores, and MCMC remains exact in the limit.
    """
    dim = 5
    flow = nsf(dim, 5, rngs=nnx.Rngs(0))
    _, state, _ = fit_flow(funnel_logdensity, flow, num_steps=1200)
    fitted = rebuild(flow, state)
    transform = neutra(funnel_logdensity, fitted)

    plain = run_chain(funnel_logdensity, jnp.zeros(dim), seed=3)
    latent = run_chain(transform.logdensity, jnp.zeros(dim), seed=3)
    mapped = transform.forward(latent.samples)

    assert jnp.all(jnp.isfinite(mapped))
    plain_ess = float(effective_sample_size(plain.samples)[0])
    neutra_ess = float(effective_sample_size(mapped)[0])
    assert neutra_ess > 1.5 * plain_ess, f"{neutra_ess:.0f} against {plain_ess:.0f}"


@pytest.mark.parametrize("kernel_fn", [mala, hmc, nuts], ids=["mala", "hmc", "nuts"])
def test_the_transformed_target_drives_every_kernel(kernel_fn):
    """NeuTra adds no kernel, so all of them must work unchanged."""
    logdensity, *_ = correlated_gaussian()
    flow = maf(DIM, 3, rngs=nnx.Rngs(0))
    transform = neutra(logdensity, flow)

    result = run_chain(
        transform.logdensity, jnp.zeros(DIM), num_steps=200, kernel_fn=kernel_fn
    )
    draws = transform.forward(result.samples)
    assert draws.shape == (200, DIM)
    assert jnp.all(jnp.isfinite(draws))


def test_the_transformed_target_survives_jit_and_vmap_over_chains():
    logdensity, *_ = correlated_gaussian()
    flow = maf(DIM, 2, rngs=nnx.Rngs(0))
    transform = neutra(logdensity, flow)

    assert jnp.isfinite(jax.jit(transform.logdensity)(jnp.zeros(DIM)))
    starts = jax.random.normal(jax.random.key(0), (4, DIM))
    assert jax.vmap(transform.logdensity)(starts).shape == (4,)


def test_an_untrained_flow_still_gives_a_valid_target():
    """A useless flow must cost efficiency, never correctness.

    With an identity-ish flow the transformed target is the original one up to
    the base density, so sampling it and mapping back must still recover the
    target's moments.
    """
    logdensity, mean, cov = correlated_gaussian()
    flow = maf(DIM, 2, rngs=nnx.Rngs(0))  # never fitted
    transform = neutra(logdensity, flow)

    result = run_chain(transform.logdensity, jnp.zeros(DIM), num_steps=4000, seed=5)
    draws = transform.forward(result.samples)
    assert float(jnp.max(jnp.abs(draws.mean(0) - mean))) < 0.3

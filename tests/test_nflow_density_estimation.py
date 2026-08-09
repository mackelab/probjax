"""End-to-end density estimation on standard 2-D benchmark targets.

These guard the two properties that unit tests on a single layer cannot see:
that a *trained* flow reaches a sensible likelihood, and that its density still
integrates to one once the layers are far from the identity. The second is what
catches a lost log-determinant -- a coupling flow whose transformed half does
not contribute its Jacobian looks fine at initialisation (every layer is the
identity, so the log-det is trivially zero) and only misbehaves after training.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import probjax.nn.generative.nflows as N

# Analytic differential entropies, in nats, where they are known.
#   checkerboard: uniform on half of [-4, 4]^2  ->  log(32)
#   crescent:     N(0, 1) x N(., 0.5)           ->  0.5*log(2*pi*e) + 0.5*log(2*pi*e*0.25)
OPTIMAL_NLL = {
    "checkerboard": float(jnp.log(32.0)),
    "crescent": 2.1450,
    "spiral": None,
}
# Loose ceilings: comfortably above what the tuned defaults reach, but far
# below what a broken log-determinant or a collapsed flow produces.
# Affine coupling is genuinely weak on the spiral, hence the loose ceiling
# there; it is still far below what a broken log-determinant produces.
NLL_CEILING = {"checkerboard": 3.9, "crescent": 2.30, "spiral": 3.4}


def checkerboard(key, n):
    k1, k2, k3 = jax.random.split(key, 3)
    x1 = jax.random.uniform(k1, (n,)) * 4.0 - 2.0
    x2 = (
        jax.random.uniform(k2, (n,))
        - jax.random.randint(k3, (n,), 0, 2).astype(jnp.float32) * 2.0
        + jnp.floor(x1) % 2.0
    )
    return jnp.stack([x1, x2], -1) * 2.0


def spiral(key, n):
    k1, k2, k3 = jax.random.split(key, 3)
    t = jnp.sqrt(jax.random.uniform(k1, (n,))) * 3.0 * jnp.pi
    sign = jnp.where(jax.random.bernoulli(k2, 0.5, (n,)), 1.0, -1.0)
    pts = jnp.stack([sign * t * jnp.cos(t), sign * t * jnp.sin(t)], -1) / 3.0
    return pts + jax.random.normal(k3, (n, 2)) * 0.08


def crescent(key, n):
    k1, k2 = jax.random.split(key)
    x1 = jax.random.normal(k1, (n,))
    x2 = jax.random.normal(k2, (n,)) * 0.5 + 0.25 * x1**2 - 1.0
    return jnp.stack([x1, x2], -1)


DENSITIES = {"checkerboard": checkerboard, "spiral": spiral, "crescent": crescent}

# One representative per structural family: autoregressive, coupling, and a
# monotone-network bijector whose analytic direction is data -> base.
ARCHITECTURES = {
    "nsf": lambda rngs: N.nsf(2, 5, rngs),
    "nsf_coupling": lambda rngs: N.SplineCouplingFlow(2, 5, rngs),
    "realnvp": lambda rngs: N.realnvp(2, 5, rngs),
    "bpf": lambda rngs: N.bpf(2, 5, rngs),
}


def total_mass(flow, key, n=50_000, proposal_scale=6.0):
    """Importance-sampling estimate of ``\\int p(x) dx``; must be ~1."""
    z = jax.random.normal(key, (n, 2)) * proposal_scale
    log_q = jnp.sum(jax.scipy.stats.norm.logpdf(z, 0.0, proposal_scale), -1)
    lp = jax.vmap(flow._logpdf)(z)
    finite = jnp.isfinite(lp)
    mass = jnp.mean(jnp.where(finite, jnp.exp(lp - log_q), 0.0))
    return float(mass), float(jnp.mean(finite))


@pytest.fixture(scope="module")
def trained():
    """Train every architecture on every density once; reuse across tests."""
    out = {}
    for dname, sampler in DENSITIES.items():
        train = sampler(jax.random.key(0), 8000)
        test = sampler(jax.random.key(1), 2000)
        for aname, build in ARCHITECTURES.items():
            flow = build(nnx.Rngs(0))
            losses = flow.fit(jax.random.key(2), train, num_steps=800, batch_size=512)
            out[(aname, dname)] = (flow, test, losses)
    return out


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
@pytest.mark.parametrize("density", list(DENSITIES))
def test_reaches_reasonable_likelihood(trained, arch, density):
    flow, test, losses = trained[(arch, density)]
    assert jnp.all(jnp.isfinite(losses)), "training diverged"
    assert losses[-1] < losses[0], "training did not reduce the loss"

    nll = float(-jnp.mean(jax.vmap(flow._logpdf)(test)))
    assert jnp.isfinite(nll)
    assert nll < NLL_CEILING[density], f"{arch} on {density}: NLL {nll:.3f}"

    optimum = OPTIMAL_NLL[density]
    if optimum is not None:
        # A flow cannot beat the true entropy by any meaningful margin; doing so
        # means the density is not normalised.
        assert nll > optimum - 0.15, (
            f"{arch} on {density}: NLL {nll:.3f} implausibly below the "
            f"entropy {optimum:.3f}"
        )


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
@pytest.mark.parametrize("density", list(DENSITIES))
def test_trained_density_is_normalised(trained, arch, density):
    """The regression test for a dropped log-determinant."""
    flow, _, _ = trained[(arch, density)]
    mass, frac_finite = total_mass(flow, jax.random.key(3))
    assert frac_finite > 0.95, f"{arch} on {density}: logpdf non-finite off-support"
    assert 0.9 < mass < 1.1, f"{arch} on {density}: total mass {mass:.4f}"


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_samples_are_finite_and_on_support(trained, arch):
    flow, test, _ = trained[(arch, "crescent")]
    samples = flow.as_dist().rvs(jax.random.key(4), (1024,))
    assert samples.shape == (1024, 2)
    assert jnp.all(jnp.isfinite(samples))

    lo, hi = jnp.min(test, 0) - 2.0, jnp.max(test, 0) + 2.0
    inside = jnp.mean(jnp.all((samples >= lo) & (samples <= hi), axis=-1))
    assert inside > 0.9, f"{arch}: only {float(inside):.2f} of samples near the data"

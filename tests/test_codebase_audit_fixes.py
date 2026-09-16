"""Regression coverage for the confirmed September 2026 audit findings."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sp

from probjax.core import condition, joint_sample, log_joint_fn
from probjax.stats import (
    binomial,
    genpareto,
    norm,
    pareto,
    poisson,
    transformed,
    truncnorm,
)
from probjax.utils.linalg import (
    batched_pcg_solve,
    is_diagonal_matrix,
    is_triangular_matrix,
    lanczos_logdet,
)
from probjax.utils.stats import differential_entropy


@pytest.mark.parametrize('method', ['vasicek', 'van es', 'ebrahimi', 'correa', 'auto'])
def test_sample_entropy_reference_and_jit(method):
    x = jnp.linspace(0.01, 2.0, 80).reshape(2, 40) ** 2
    result = jax.jit(lambda x: differential_entropy(x, method=method, axis=-1))(x)
    np.testing.assert_allclose(
        result,
        sp.differential_entropy(np.asarray(x), method=method, axis=-1),
        atol=2e-6,
    )


@pytest.mark.parametrize('scale', [1.0, 1e-4, 1e-10])
def test_pcg_small_rhs(scale):
    a = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    b = scale * jnp.array([[1.0], [2.0]])
    x, info = batched_pcg_solve(lambda x: a @ x, b, block_size=1, tol=1e-5)
    np.testing.assert_allclose(a @ x, b, rtol=1e-5, atol=scale * 1e-6)
    assert jnp.all(info.converged)


@pytest.mark.parametrize(
    'a',
    [
        jnp.array([[2.0]]),
        jnp.array([[2.0, 1.0], [1.0, 2.0]]),
        1e-10 * jnp.array([[2.0, 1.0], [1.0, 2.0]]),
    ],
)
def test_full_basis_lanczos_logdet(a):
    result = jax.jit(lambda a: lanczos_logdet(a, num_steps=a.shape[0]))(a)
    np.testing.assert_allclose(result, jnp.linalg.slogdet(a)[1], rtol=2e-6, atol=2e-6)


def test_matrix_predicates():
    assert is_triangular_matrix(jnp.array([[1.0, 2.0], [0.0, 1.0]]), lower=False)
    assert not is_triangular_matrix(jnp.array([[1.0, 2.0], [3.0, 1.0]]), lower=False)
    np.testing.assert_array_equal(
        is_diagonal_matrix(jnp.stack([jnp.eye(2), jnp.ones((2, 2))])), [True, False]
    )


def test_multistep_import():
    importlib.import_module('probjax.utils.odeutil.solvers.multistep')


@pytest.mark.parametrize(
    'rule',
    [
        'merwe_sigma_point',
        'julier_uhlmann_sigma_points',
        'spherical_simplex_sigma_points',
    ],
)
@pytest.mark.parametrize("dimension", [1, 3, 5, 20])
def test_sigma_point_covariance(rule, dimension):
    uk = importlib.import_module('probjax.inference.filtering.unscented_kalman_filter')
    mu = jnp.linspace(-2.0, 1.0, dimension)
    factor = jax.random.normal(
        jax.random.key(dimension), (dimension, dimension)
    ) / jnp.sqrt(dimension)
    cov = factor @ factor.T + jnp.eye(dimension)
    actual_mu, actual_cov = uk.unscented_transform(*getattr(uk, rule)(mu, cov))
    np.testing.assert_allclose(actual_mu, mu, atol=2e-6)
    np.testing.assert_allclose(actual_cov, cov, atol=2e-6)


@pytest.mark.parametrize('callable_r', [False, True])
def test_ukf_linear_gaussian_update(callable_r):
    uk = importlib.import_module('probjax.inference.filtering.unscented_kalman_filter')
    r = (lambda t: jnp.eye(2)) if callable_r else jnp.eye(2)
    step = jax.jit(uk.build_kernel(lambda x, t0, t1: x, jnp.eye(2), lambda x, t: x, r))
    state, info = step(uk.init(jnp.zeros(2), jnp.eye(2), 0), 1, jnp.ones(2))
    np.testing.assert_allclose(state.mean, jnp.full(2, 2 / 3), atol=1e-6)
    np.testing.assert_allclose(state.cov, jnp.eye(2) * 2 / 3, atol=1e-6)
    np.testing.assert_allclose(
        info.log_likelihood,
        sp.multivariate_normal.logpdf(np.ones(2), cov=3 * np.eye(2)),
        atol=1e-6,
    )


def test_gaussian_imh_covariance_density():
    from probjax.inference.mcmc.imh import gaussian_imh, proposal_gaussian_logpdf

    kernel = gaussian_imh(lambda x: -0.5 * jnp.sum(x * x))
    state = kernel.init(jax.random.key(0), jnp.array([1.0, -2.0]))
    params = kernel.init_params(state, mean=jnp.zeros(2), cov=jnp.array([4.0, 9.0]))
    np.testing.assert_allclose(
        proposal_gaussian_logpdf(state, params=params),
        sp.multivariate_normal.logpdf([1.0, -2.0], cov=np.diag([4.0, 9.0])),
        atol=1e-6,
    )


def test_pareto_samples():
    samples = pareto.rvs(jax.random.key(0), shape=(20000,), b=1.0, alpha=3.0)
    assert 1 <= samples.min() < 1.01
    np.testing.assert_allclose(samples.mean(), 1.5, atol=0.04)


@pytest.mark.parametrize('c', [0.0, 0.5, -0.5, -1.0, -2.0])
def test_genpareto_reference(c):
    x = jnp.array([-1.0, 0.0, 0.2, 0.5, 1.0, 2.0, jnp.inf])
    for name in ['pdf', 'cdf', 'sf']:
        np.testing.assert_allclose(
            getattr(genpareto, name)(x, c=c),
            getattr(sp.genpareto, name)(np.asarray(x), c),
            atol=2e-6,
            rtol=2e-6,
        )
    q = jnp.array([0.0, 0.01, 0.5, 0.99, 1.0])
    for name in ['ppf', 'isf']:
        np.testing.assert_allclose(
            jax.jit(lambda q: getattr(genpareto, name)(q, c=c))(q),
            getattr(sp.genpareto, name)(np.asarray(q), c),
            rtol=2e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize(
    'n,p',
    [(1, 0.5), (4, 0.5), (100, 0.02), (1000, 0.5), (10000, 0.5), (4, 0.0), (4, 1.0)],
)
def test_binomial_entropy(n, p):
    with jax.enable_x64():
        np.testing.assert_allclose(
            jax.jit(binomial.entropy)(n, p),
            sp.binom.entropy(n, p),
            atol=2e-6,
            rtol=2e-6,
        )


def test_binomial_fractional_cdf():
    x = jnp.array([-0.5, 0.0, 0.5, 1.9, 4.0, 5.0])
    np.testing.assert_allclose(
        binomial.cdf(x, 4, 0.5), sp.binom.cdf(np.asarray(x), 4, 0.5), atol=1e-6
    )


@pytest.mark.parametrize('rate', [0.0, 0.1, 10.0, 256.0, 300.0, 10000.0])
def test_poisson_entropy(rate):
    with jax.enable_x64():
        np.testing.assert_allclose(
            jax.jit(poisson.entropy)(rate),
            (
                -np.sum(
                    sp.poisson.pmf(
                        np.arange(max(1024, int(rate + 20 * np.sqrt(rate)))), rate
                    )
                    * np.nan_to_num(
                        sp.poisson.logpmf(
                            np.arange(max(1024, int(rate + 20 * np.sqrt(rate)))), rate
                        ),
                        neginf=0,
                    )
                )
            ),
            atol=2e-6,
            rtol=2e-6,
        )


@pytest.mark.parametrize(
    'a,b', [(-1.0, 1.0), (-np.inf, np.inf), (2.0, 4.0), (-np.inf, 0.0), (0.0, np.inf)]
)
def test_truncnorm_reference(a, b):
    with jax.enable_x64():
        for method in ['mean', 'var', 'entropy']:
            reference = (
                getattr(sp.truncnorm, method)(a, b)
                if method != "entropy" or (np.isfinite(a) and np.isfinite(b))
                else sp.norm.entropy()
                - (0 if a == -np.inf and b == np.inf else np.log(2))
            )
            np.testing.assert_allclose(
                getattr(truncnorm, method)(a=a, b=b), reference, atol=1e-8
            )
        q = jnp.array([0.0, 1e-5, 0.1, 0.5, 0.9, 1 - 1e-5, 1.0])
        for method in ['ppf', 'isf']:
            np.testing.assert_allclose(
                jax.jit(lambda q: getattr(truncnorm, method)(q, a=a, b=b))(q),
                getattr(sp.truncnorm, method)(np.asarray(q), a, b),
                atol=1e-8,
            )


@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_monotone_transformed_distribution(sign):
    dist = transformed(norm(loc=2.0, scale=3.0), lambda x: sign * x)
    np.testing.assert_allclose(
        dist.cdf(1.0), sp.norm.cdf(1.0, loc=sign * 2.0, scale=3.0), atol=1e-6
    )
    np.testing.assert_allclose(
        dist.ppf(0.9), sp.norm.ppf(0.9, loc=sign * 2.0, scale=3.0), atol=1e-6
    )


def test_nested_jit_named_site():
    @jax.jit
    def model(key):
        return norm.rvs(key, name='x')

    assert set(joint_sample(model)(jax.random.key(0))) == {'x'}
    np.testing.assert_allclose(
        log_joint_fn(model)(x=jnp.array(2.0)), sp.norm.logpdf(2.0), atol=1e-6
    )
    np.testing.assert_allclose(
        condition(model, {'x': jnp.array(2.0)})(jax.random.key(0)), 2.0
    )


def test_flash3_backend_adapters(monkeypatch):
    flash = importlib.import_module(
        'probjax.nn.pallas_kernels.kernels.flash_attention3'
    )

    def forward(q, k, v, config, save_residuals=False):
        assert (q, k, v, config) == (1, 2, 3, 4)
        return (5, (6,)) if save_residuals else 5

    def backward(config, save_residuals, res, do):
        assert (config, save_residuals, res, do) == (4, False, (1, 2, 3, 5, 6), 7)
        return (8, 9, 10)

    monkeypatch.setattr(flash, '_attention_forward_impl', forward)
    monkeypatch.setattr(flash, '_attention_with_pipeline_emitter_impl', forward)
    monkeypatch.setattr(flash, '_attention_backward_impl', backward)
    for pipeline in [False, True]:
        assert (
            flash._flash_fwd_impl(1, 2, 3, config=4, use_pipeline_emitter=pipeline) == 5
        )
        assert flash._flash_fwd_res_impl(
            1, 2, 3, config=4, use_pipeline_emitter=pipeline
        ) == (5, 6)
    assert flash._flash_bwd_impl(7, 1, 2, 3, 5, 6, config=4) == (8, 9, 10)


def test_transformer_distinct_reproducible_dropout_keys():
    from flax import nnx
    from probjax.nn import Transformer

    seen = []

    class RecordingDropout(nnx.Module):
        def __init__(self, rate, rngs):
            pass

        def __call__(self, x, deterministic=None, rngs=None):
            seen.append(np.asarray(jax.random.key_data(rngs)))
            return x

    model = Transformer(
        8,
        2,
        3,
        4,
        dropout_rate=0.2,
        dropout_rate_attn=0.0,
        dropout_cls=RecordingDropout,
        rngs=nnx.Rngs(0),
    )
    for _ in range(2):
        model(jnp.ones((2, 4, 8)), rng=jax.random.key(8), deterministic=False)
    assert len(seen) == 6
    assert len({key.tobytes() for key in seen[:3]}) == 3
    np.testing.assert_array_equal(seen[:3], seen[3:])


def test_distribution_quantile_gradients():
    np.testing.assert_allclose(
        jax.grad(lambda c: genpareto.ppf(0.5, c=c))(0.0),
        0.5 * np.log(2) ** 2,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        jax.grad(lambda scale: truncnorm.ppf(0.7, scale=scale))(1.0),
        sp.norm.ppf(0.7),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        jax.grad(lambda q: truncnorm.ppf(q, a=-1.0, b=2.0))(0.7),
        1 / truncnorm.pdf(truncnorm.ppf(0.7, a=-1.0, b=2.0), a=-1.0, b=2.0),
        rtol=2e-6,
    )


def test_truncnorm_moments():
    with jax.enable_x64():
        for order in range(5):
            np.testing.assert_allclose(
                truncnorm.moment(order, loc=1.0, scale=2.0, a=-1.0, b=4.0),
                sp.truncnorm.moment(order, -1.0, 1.5, loc=1.0, scale=2.0),
                atol=2e-9,
            )
        skew, kurtosis = sp.truncnorm.stats(-1.0, 1.5, moments='sk')
        np.testing.assert_allclose(truncnorm.skew(a=-1.0, b=1.5), skew, atol=2e-9)
        np.testing.assert_allclose(
            truncnorm.kurtosis(a=-1.0, b=1.5), kurtosis, atol=2e-9
        )


def test_lanczos_stochastic_probes():
    # Diagonal matrices have zero Rademacher trace variance, even with one probe.
    a = jnp.diag(jnp.linspace(1.0, 2.0, 20))
    estimate = jax.jit(
        lambda a: lanczos_logdet(a, num_steps=20, num_probes=2, key=jax.random.key(42))
    )(a)
    np.testing.assert_allclose(estimate, jnp.linalg.slogdet(a)[1], rtol=2e-6)


@pytest.mark.parametrize('a,b', [(1.0, 1.001), (8.0, 10.0), (20.0, 21.0)])
def test_truncnorm_fp32_variance(a, b):
    a, b = np.float32(a), np.float32(b)
    reference = sp.truncnorm.var(float(a), float(b))
    np.testing.assert_allclose(truncnorm.var(a=a, b=b), reference, rtol=2e-4)


def test_binomial_fp32_large_skewed_entropy():
    np.testing.assert_allclose(
        binomial.entropy(1000000, 1e-5), sp.binom.entropy(1000000, 1e-5), rtol=2e-6
    )

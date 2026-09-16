"""Expected-behavior tests: intentionally fail on the audited worktree."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats as sp
from probjax.utils.linalg import lanczos_logdet, batched_pcg_solve
from probjax.stats import genpareto, truncnorm
from probjax.nn.utils import call_with_optional_rng, filter_supported_kwargs


def test_logdet_gradient_scaled_identity():
    with jax.enable_x64():
        actual = jax.grad(lambda s: lanczos_logdet(s*jnp.eye(2), num_steps=2))(2.)
        np.testing.assert_allclose(actual, 1., atol=1e-12)


def test_genpareto_explicit_shape_parameter_sampling():
    sample = genpareto.rvs(jax.random.key(0), c=.5, shape=(5,))
    assert sample.shape == (5,)
    assert jnp.all(jnp.isfinite(sample))


def test_truncnorm_small_positive_quantile():
    with jax.enable_x64():
        actual = truncnorm.ppf(1e-100, a=-1., b=1.)
        assert jnp.isfinite(actual)
        np.testing.assert_allclose(actual, sp.truncnorm.ppf(1e-100, -1., 1.), atol=1e-14)


def test_truncnorm_quantile_second_derivative():
    with jax.enable_x64():
        q = .7
        actual = jax.grad(jax.grad(lambda p: truncnorm.ppf(p)))(q)
        reference = sp.norm.ppf(q)/sp.norm.pdf(sp.norm.ppf(q))**2
        np.testing.assert_allclose(actual, reference, rtol=1e-8)


def test_truncnorm_one_sided_tail_variance():
    actual = truncnorm.var(a=jnp.float32(20.), b=jnp.inf)
    np.testing.assert_allclose(actual, sp.truncnorm.var(20.,np.inf),rtol=1e-3)


def test_pcg_small_representable_rhs():
    b = jnp.full((2,1),1e-20,dtype=jnp.float32)
    actual,info = batched_pcg_solve(lambda x:x,b,block_size=1)
    # Check in float64 on the host so the assertion itself cannot underflow.
    np.testing.assert_allclose(np.asarray(actual,dtype=np.float64),np.asarray(b,dtype=np.float64),rtol=1e-5,atol=0)
    assert jnp.all(info.converged)


def test_optional_rng_plain_function():
    assert call_with_optional_rng(lambda x:x+1,1.,rng=jax.random.key(0)) == 2.


def test_forwarding_constructor_kwargs():
    class ForwardingLayer:
        def __init__(self,**kwargs):
            self.kwargs=kwargs
    kwargs={'kernel_sharding':('data',None)}
    assert filter_supported_kwargs(ForwardingLayer,**kwargs)==kwargs


def test_genpareto_negative_shape_mode():
    # At c=-2, the density increases to a singularity at its upper endpoint.
    np.testing.assert_allclose(genpareto.mode(c=-2.),.5)

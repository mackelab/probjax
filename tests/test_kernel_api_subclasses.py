"""Explicit API subclasses for built MCMC/SMC kernels."""

import jax
import jax.numpy as jnp

from probjax.inference.base import Kernel
from probjax.inference.mcmc.hmc import hmc
from probjax.inference.mcmc.mala import mala
from probjax.inference.mcmc.mh import mh
from probjax.inference.mcmc.slice import slice
from probjax.inference.smc.path_smc import path_smc


def _gaussian_logdensity(x):
    return -0.5 * jnp.sum(x**2)


def test_built_kernels_carry_per_algorithm_types():
    built = hmc(_gaussian_logdensity)
    assert type(built).__name__ == "hmc"
    assert isinstance(built, Kernel)
    assert "hmc" in repr(built).lower() or type(built).__doc__ is hmc.__doc__
    assert type(built).__doc__ == hmc.__doc__

    built_mala = mala(_gaussian_logdensity)
    assert type(built_mala).__name__ == "mala"
    assert isinstance(built_mala, Kernel)


def test_built_kernels_keep_default_adaptation():
    built = hmc(_gaussian_logdensity)
    assert built.default_adaptation is not None
    built_slice = slice(_gaussian_logdensity)
    assert built_slice.default_adaptation is not None


def test_kernel_init_tolerates_missing_rng_key():
    # BlackJAX inits take no rng_key; the subclass shim absorbs it.
    built = hmc(_gaussian_logdensity)
    state = built.init(jax.random.key(0), jnp.zeros(2))
    assert state is not None


def test_smc_built_kernel_carries_algorithm_type():
    from probjax.inference.smc.path import GeometricPath

    kernel = path_smc(
        logprior_fn=lambda x: -0.5 * jnp.sum(x**2),
        loglikelihood_fn=lambda x: -0.5 * jnp.sum((x - 1.0) ** 2),
        path=GeometricPath(),
        mcmc_kernel=mh,
    )
    assert type(kernel).__name__ == "path_smc"
    assert isinstance(kernel, Kernel)

import jax
import jax.numpy as jnp
import pytest

from probjax.core.custom_primitives.random_variable import rv_p
from probjax.stats import norm

VMAP_BATCH = 256


def _eager_call():
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)

    def run_once(key):
        out = rv_p.bind(key, loc, scale, dist=norm, name="x")
        return jax.block_until_ready(out)

    return run_once


def _jit_call():
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)

    def fn(rng, loc_, scale_):
        return rv_p.bind(rng, loc_, scale_, dist=norm, name="x")

    jitted = jax.jit(fn)

    def run_once(key):
        out = jitted(key, loc, scale)
        return jax.block_until_ready(out)

    return run_once


def _baseline_eager_call():
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)

    def run_once(key):
        out = norm._rvs_impl(key, loc, scale)
        return jax.block_until_ready(out)

    return run_once


def _baseline_jit_call():
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)

    def fn(rng, loc_, scale_):
        return norm._rvs_impl(rng, loc_, scale_)

    jitted = jax.jit(fn)

    def run_once(key):
        out = jitted(key, loc, scale)
        return jax.block_until_ready(out)

    return run_once


@pytest.mark.benchmark(group="rv")
def test_benchmark_rv_01_eager_bind(benchmark):
    run_once = _eager_call()
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    idx = {"i": 0}

    def target():
        out = run_once(keys[idx["i"] % len(keys)])
        idx["i"] += 1
        return out

    out = benchmark(target)
    assert out.shape == ()


@pytest.mark.benchmark(group="rv")
def test_benchmark_rv_03_jit_steady(benchmark):
    run_once = _jit_call()
    keys = jax.random.split(jax.random.PRNGKey(1), 4097)
    _ = run_once(keys[0])
    idx = {"i": 1}

    def target():
        out = run_once(keys[idx["i"] % len(keys)])
        idx["i"] += 1
        return out

    out = benchmark(target)
    assert out.shape == ()


@pytest.mark.benchmark(group="rv")
def test_benchmark_rv_02_baseline_eager(benchmark):
    run_once = _baseline_eager_call()
    keys = jax.random.split(jax.random.PRNGKey(10), 4096)
    idx = {"i": 0}

    def target():
        out = run_once(keys[idx["i"] % len(keys)])
        idx["i"] += 1
        return out

    out = benchmark(target)
    assert out.shape == ()


@pytest.mark.benchmark(group="rv")
def test_benchmark_rv_04_baseline_jit_steady(benchmark):
    run_once = _baseline_jit_call()
    keys = jax.random.split(jax.random.PRNGKey(12), 4097)
    _ = run_once(keys[0])
    idx = {"i": 1}

    def target():
        out = run_once(keys[idx["i"] % len(keys)])
        idx["i"] += 1
        return out

    out = benchmark(target)
    assert out.shape == ()

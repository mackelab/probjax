import jax
import jax.numpy as jnp
import pytest

from probjax.core.custom_primitives.custom_inverse import custom_inverse
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.stats import norm


def _build_custom_inverse():
    ci = custom_inverse(lambda x: x + 1.0)

    @ci.definv_and_logdet
    def inv_and_logdet(y):
        return y - 1.0, jnp.asarray(0.0)

    return ci


def _custom_inverse_eager_call():
    ci = _build_custom_inverse()
    x = jnp.asarray(0.5)

    def run_once():
        out = ci(x)
        return jax.block_until_ready(out)

    return run_once


def _custom_inverse_jit_call():
    ci = _build_custom_inverse()
    x = jnp.asarray(0.5)
    jitted = jax.jit(ci)
    _ = jitted(x)

    def run_once():
        out = jitted(x)
        return jax.block_until_ready(out)

    return run_once


def _custom_inverse_baseline_eager_call():
    x = jnp.asarray(0.5)

    def run_once():
        out = x + 1.0
        return jax.block_until_ready(out)

    return run_once


def _custom_inverse_baseline_jit_call():
    x = jnp.asarray(0.5)
    jitted = jax.jit(lambda z: z + 1.0)
    _ = jitted(x)

    def run_once():
        out = jitted(x)
        return jax.block_until_ready(out)

    return run_once


@pytest.mark.benchmark(group="custom_inverse")
def test_benchmark_custom_inverse_01_eager(benchmark):
    run_once = _custom_inverse_eager_call()
    out = benchmark(run_once)
    assert out.shape == ()


@pytest.mark.benchmark(group="custom_inverse")
def test_benchmark_custom_inverse_03_jit_steady(benchmark):
    run_once = _custom_inverse_jit_call()
    out = benchmark(run_once)
    assert out.shape == ()


@pytest.mark.benchmark(group="custom_inverse")
def test_benchmark_custom_inverse_02_baseline_eager(benchmark):
    run_once = _custom_inverse_baseline_eager_call()
    out = benchmark(run_once)
    assert out.shape == ()


@pytest.mark.benchmark(group="custom_inverse")
def test_benchmark_custom_inverse_04_baseline_jit_steady(benchmark):
    run_once = _custom_inverse_baseline_jit_call()
    out = benchmark(run_once)
    assert out.shape == ()


def _rv_eager_call():
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)

    def run_once(key):
        out = rv_p.bind(key, loc, scale, dist=norm, name="x")
        return jax.block_until_ready(out)

    return run_once


def _rv_jit_call():
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)

    def fn(rng, loc_, scale_):
        return rv_p.bind(rng, loc_, scale_, dist=norm, name="x")

    jitted = jax.jit(fn)

    def run_once(key):
        out = jitted(key, loc, scale)
        return jax.block_until_ready(out)

    return run_once


def _rv_baseline_eager_call():
    loc = jnp.asarray(0.0)
    scale = jnp.asarray(1.0)

    def run_once(key):
        out = norm._rvs_impl(key, loc, scale)
        return jax.block_until_ready(out)

    return run_once


def _rv_baseline_jit_call():
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
    run_once = _rv_eager_call()
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
    run_once = _rv_jit_call()
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
    run_once = _rv_baseline_eager_call()
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
    run_once = _rv_baseline_jit_call()
    keys = jax.random.split(jax.random.PRNGKey(12), 4097)
    _ = run_once(keys[0])
    idx = {"i": 1}

    def target():
        out = run_once(keys[idx["i"] % len(keys)])
        idx["i"] += 1
        return out

    out = benchmark(target)
    assert out.shape == ()

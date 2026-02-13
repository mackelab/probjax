import jax
import jax.numpy as jnp
import pytest

from probjax.core.custom_primitives.custom_inverse import custom_inverse

VMAP_BATCH = 256


def _build_custom_inverse():
    ci = custom_inverse(lambda x: x + 1.0)

    @ci.definv_and_logdet
    def inv_and_logdet(y):
        return y - 1.0, jnp.asarray(0.0)

    return ci


def _eager_call():
    ci = _build_custom_inverse()
    x = jnp.asarray(0.5)

    def run_once():
        out = ci(x)
        return jax.block_until_ready(out)

    return run_once


def _jit_call():
    ci = _build_custom_inverse()
    x = jnp.asarray(0.5)
    jitted = jax.jit(ci)
    _ = jitted(x)

    def run_once():
        out = jitted(x)
        return jax.block_until_ready(out)

    return run_once


def _baseline_eager_call():
    x = jnp.asarray(0.5)

    def run_once():
        out = x + 1.0
        return jax.block_until_ready(out)

    return run_once


def _baseline_jit_call():
    x = jnp.asarray(0.5)
    jitted = jax.jit(lambda z: z + 1.0)
    _ = jitted(x)

    def run_once():
        out = jitted(x)
        return jax.block_until_ready(out)

    return run_once


@pytest.mark.benchmark(group="custom_inverse")
def test_benchmark_custom_inverse_01_eager(benchmark):
    run_once = _eager_call()
    out = benchmark(run_once)
    assert out.shape == ()


@pytest.mark.benchmark(group="custom_inverse")
def test_benchmark_custom_inverse_03_jit_steady(benchmark):
    run_once = _jit_call()
    out = benchmark(run_once)
    assert out.shape == ()


@pytest.mark.benchmark(group="custom_inverse")
def test_benchmark_custom_inverse_02_baseline_eager(benchmark):
    run_once = _baseline_eager_call()
    out = benchmark(run_once)
    assert out.shape == ()


@pytest.mark.benchmark(group="custom_inverse")
def test_benchmark_custom_inverse_04_baseline_jit_steady(benchmark):
    run_once = _baseline_jit_call()
    out = benchmark(run_once)
    assert out.shape == ()

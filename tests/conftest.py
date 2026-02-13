import os
import sys


def _marker_expr_from_args(argv):
    if "-m" in argv:
        idx = argv.index("-m")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    for arg in argv:
        if arg.startswith("-m") and len(arg) > 2:
            return arg[2:]
    return ""


def _keyword_expr_from_args(argv):
    if "-k" in argv:
        idx = argv.index("-k")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    for arg in argv:
        if arg.startswith("-k") and len(arg) > 2:
            return arg[2:]
    return ""


def _mesh_marker_enabled():
    expr = _marker_expr_from_args(sys.argv)
    expr = expr.strip()
    if not expr:
        # Also enable when -k selects mesh tests.
        kexpr = _keyword_expr_from_args(sys.argv).strip()
        return "mesh" in kexpr if kexpr else False
    return expr == "mesh"


cpu_devices = 4
enable_multi = _mesh_marker_enabled()
if enable_multi and cpu_devices > 1:
    xla_flags = os.environ.get("XLA_FLAGS", "")
    flag = f"--xla_force_host_platform_device_count={cpu_devices}"
    if flag not in xla_flags:
        os.environ["XLA_FLAGS"] = f"{xla_flags} {flag}".strip()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from probjax.utils.odeutil.solvers.base import get_methods as get_methods_ode
from probjax.utils.sdeutil import get_methods as get_methods_sde

try:
    import pytest_benchmark.plugin as _pytest_benchmark_plugin
except ImportError:
    _pytest_benchmark_plugin = None

# Remove the hardcoded CPU configuration
jax.config.update("jax_platform_name", "cpu")
# Set a fixed random key for all tests
np.random.seed(0)

key = random.PRNGKey(0)


# Invertile function testcase fixtures


# Simple invertible 1d transformations
def for_loop_sum(x):
    x0 = x
    for _ in range(10):
        x0 += 2
    return x0


def for_loop_mul(x):
    x0 = x
    for _ in range(10):
        x0 *= 2
    return x0


# Reshape and revert
def reshape_and_revert(x):
    y = x.reshape((1, 1, 1, 1, 1) + x.shape)
    return y.reshape(x.shape)


def broad_cast_and_revert(x):
    y = x[..., None, None, None, None]
    return y[..., 0, 0, 0, 0]


# jnp.where does not yet work! -> thus also not leaky relu and so on...
INVERTIBLE_FUNCTIONS_1d = [
    jnp.log,
    jnp.log2,
    jnp.log10,
    jnp.log1p,
    # lambda x: jnp.logaddexp(x,1.), # TODO jnp.where does not yet work!
    # lambda x: jnp.logaddexp2(x,1.), # TODO jnp.where does not yet work!
    jnp.exp,
    jnp.exp2,
    # jnp.flip, # TODO ERROR
    for_loop_sum,
    for_loop_mul,
    reshape_and_revert,
    broad_cast_and_revert,
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: x**3,
    lambda x: x * 2,
    lambda x: x / 2,
]


@pytest.fixture(params=INVERTIBLE_FUNCTIONS_1d)
def invertible_function_1d(request):
    return request.param


# SDE problems fixtures ---------------------------------------------------------


METHODS = get_methods_sde()


@pytest.fixture(params=METHODS, ids=METHODS)
def sde_method(request):
    return request.param


# ODE problems fixtures ---------------------------------------------------------


METHODS = get_methods_ode()


@pytest.fixture(params=METHODS, ids=METHODS)
def ode_method(request):
    return request.param


def pytest_addoption(parser):
    parser.addoption(
        "--gpu", action="store_true", default=False, help="run tests requiring GPU"
    )
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="run benchmark tests (disabled by default)",
    )
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "gpu"],
        help="device to run tests on (cpu or gpu)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU to run")
    config.addinivalue_line("markers", "mesh: requires multi-device mesh to run")
    config.addinivalue_line(
        "markers", "benchmark: performance benchmark tests (opt-in)"
    )
    # Set JAX platform based on device option
    device = config.getoption("--device")
    jax.config.update("jax_platform_name", device)


def pytest_collection_modifyitems(config, items):
    device = config.getoption("--device")
    run_benchmarks = config.getoption("--run-benchmarks")
    if device == "gpu":
        skip_benchmark = pytest.mark.skip(reason="need --run-benchmarks option to run")
        for item in items:
            if "benchmark" in item.keywords and not run_benchmarks:
                item.add_marker(skip_benchmark)
        return
    skip_benchmark = pytest.mark.skip(reason="need --run-benchmarks option to run")
    for item in items:
        if "benchmark" in item.keywords and not run_benchmarks:
            item.add_marker(skip_benchmark)
    if enable_multi:
        skip_non_mesh = pytest.mark.skip(reason="requires -m mesh to run")
        for item in items:
            if "mesh" not in item.keywords:
                item.add_marker(skip_non_mesh)
    skip_gpu = pytest.mark.skip(reason="need --device gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


if _pytest_benchmark_plugin is None:

    @pytest.fixture
    def benchmark():
        pytest.skip("pytest-benchmark is not installed; install dev extras to run")

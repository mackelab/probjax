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


cpu_devices = 8
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

from probjax.utils.odeutil.solvers.base import get_methods as get_methods_ode
from probjax.utils.sdeutil import get_methods as get_methods_sde

try:
    import pytest_benchmark.plugin as _pytest_benchmark_plugin
except ImportError:
    _pytest_benchmark_plugin = None

# Set a fixed random key for all tests
np.random.seed(0)


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


SDE_METHODS = get_methods_sde()
# Methods that require split_drift-wrapped drift (dedicated tests only).
SPLIT_DRIFT_SDE_METHODS = ["exp_euler_maruyama"]
# linear_exact_sde requires linear_drift + const_diffusion; covered by its own tests.
GENERIC_SDE_METHODS = [
    m for m in SDE_METHODS if m not in SPLIT_DRIFT_SDE_METHODS + ["linear_exact_sde"]
]


@pytest.fixture(params=SDE_METHODS, ids=SDE_METHODS)
def sde_method(request):
    return request.param


@pytest.fixture(params=GENERIC_SDE_METHODS, ids=GENERIC_SDE_METHODS)
def generic_sde_method(request):
    """SDE methods that accept plain drift/diffusion callables."""
    return request.param


@pytest.fixture(params=SPLIT_DRIFT_SDE_METHODS, ids=SPLIT_DRIFT_SDE_METHODS)
def split_drift_sde_method(request):
    """SDE methods that require a split_drift-wrapped drift."""
    return request.param


# ODE problems fixtures ---------------------------------------------------------


ODE_METHODS = get_methods_ode()
# Exponential methods that require a split_drift-wrapped drift.
SPLIT_DRIFT_ODE_METHODS = ["exp_ab2_scalarL", "exp_ab3_scalarL"]
# linear_exact requires a linear_drift-wrapped drift; plain-callable tests
# (nonlinear, pytree, ...) cannot run it.
GENERIC_ODE_METHODS = [m for m in ODE_METHODS if m not in SPLIT_DRIFT_ODE_METHODS]
PLAIN_ODE_METHODS = [m for m in GENERIC_ODE_METHODS if m != "linear_exact"]

assert len(ODE_METHODS) == 28, ODE_METHODS
assert len(GENERIC_ODE_METHODS) + len(SPLIT_DRIFT_ODE_METHODS) == 28


@pytest.fixture(params=GENERIC_ODE_METHODS, ids=GENERIC_ODE_METHODS)
def generic_ode_method(request):
    """ODE methods usable with linear_drift (excl. split_drift-only methods)."""
    return request.param


@pytest.fixture(params=PLAIN_ODE_METHODS, ids=PLAIN_ODE_METHODS)
def plain_ode_method(request):
    """ODE methods usable with plain callables (excl. split_drift, linear_exact)."""
    return request.param


@pytest.fixture(params=SPLIT_DRIFT_ODE_METHODS, ids=SPLIT_DRIFT_ODE_METHODS)
def split_drift_ode_method(request):
    """ODE methods that require a split_drift-wrapped drift."""
    return request.param


# Shared PRNG fixtures ----------------------------------------------------------


@pytest.fixture
def rng():
    """Shared base PRNG key (identical to ``jax.random.PRNGKey(0)``)."""
    return jax.random.key(0)


@pytest.fixture
def rng_split(rng):
    """Yield fresh subkeys derived from the shared ``rng`` fixture key."""
    state = {"n": 0}

    def _next():
        key = jax.random.fold_in(rng, state["n"])
        state["n"] += 1
        return key

    return _next


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
        default=None,
        choices=["cpu", "gpu"],
        help="device to run tests on (cpu or gpu)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU to run")
    config.addinivalue_line("markers", "mesh: requires multi-device mesh to run")
    config.addinivalue_line(
        "markers", "benchmark: performance benchmark tests (opt-in)"
    )
    config.addinivalue_line(
        "markers", "slow: long-running tests (training loops, full grids)"
    )
    config.addinivalue_line("markers", "docs: documentation tests")
    # Set JAX platform based on device option
    device = config.getoption("--device")
    if config.getoption("--gpu"):
        if device == "cpu":
            raise pytest.UsageError("--gpu conflicts with --device cpu")
        device = "gpu"
    device = device or "cpu"
    config.option.device = device
    jax.config.update("jax_platform_name", device)


def pytest_collection_modifyitems(config, items):
    device = config.getoption("--device")
    run_benchmarks = config.getoption("--run-benchmarks")

    # Deselect items that shouldn't run, collect remaining items
    selected = []
    deselected = []

    for item in items:
        # Deselect benchmark tests unless --run-benchmarks is passed
        if "benchmark" in item.keywords and not run_benchmarks:
            deselected.append(item)
            continue

        # Deselect mesh tests unless -m mesh is explicitly requested
        if "mesh" in item.keywords and not enable_multi:
            deselected.append(item)
            continue

        # When running mesh tests, deselect non-mesh tests
        if enable_multi and "mesh" not in item.keywords:
            deselected.append(item)
            continue

        # Deselect GPU tests unless --device gpu is passed
        if "gpu" in item.keywords and device != "gpu":
            deselected.append(item)
            continue

        selected.append(item)

    # Update the items list and report deselected
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


if _pytest_benchmark_plugin is None:

    @pytest.fixture
    def benchmark():
        pytest.skip("pytest-benchmark is not installed; install dev extras to run")

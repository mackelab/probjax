import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from probjax.utils.odeutil.solvers.base import get_methods as get_methods_ode
from probjax.utils.sdeutil import get_methods as get_methods_sde

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
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "gpu"],
        help="device to run tests on (cpu or gpu)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires GPU to run")
    # Set JAX platform based on device option
    device = config.getoption("--device")
    jax.config.update("jax_platform_name", device)


def pytest_collection_modifyitems(config, items):
    device = config.getoption("--device")
    if device == "gpu":
        return
    skip_gpu = pytest.mark.skip(reason="need --device gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)

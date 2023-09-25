import pytest

import jax
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(0)

# Invertile function testcase


def for_loop_sum(x):
    x0 = x
    for i in range(10):
        x0 += 2
    return x0


def for_loop_mul(x):
    x0 = x
    for i in range(10):
        x0 *= 2
    return x0


# jnp.where does not yet work!
INVERTIBLE_FUNCTIONS = [
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
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: x**3,
    lambda x: x * 2,
    lambda x: x / 2,
]


@pytest.fixture(params=INVERTIBLE_FUNCTIONS)
def invertible_function(request):
    return request.param

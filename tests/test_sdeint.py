from probjax.utils.sdeint import sdeint, get_methods
from probjax.inference.kalman_filter import matrix_fraction_decomposition

import pytest

import jax
import jax.numpy as jnp


METHODS = get_methods()


# Scalar sde with ground truth solution
x0 = jnp.array([0.5])
t = jnp.linspace(0., 10., 200)

def f(t, x):
    return -(1 / 10)**2 * jnp.sin(x)*jnp.cos(x)**3 

def g(t, x):
    return 1/10 * jnp.cos(x)**2

def f_true(Wt,t, x0):
    return jnp.arctan(1/10 * Wt + jnp.tan(x0))

@pytest.mark.parametrize("method", METHODS)
def test_sdeint_scalar(method):
    key = jax.random.PRNGKey(0)
    f_approx, W = sdeint(key,f,g, x0, t, method=method, return_brownian=True)
    f_sol = f_true(W,t,x0)
    error = jnp.mean((f_approx - f_sol) ** 2)
    assert error < 1e-1, "Solver failed on dense grid to match true solution"


# Multidiemnsional sde with diagonal cov ground truth solution

d = 2
x0_2 = jnp.array([0.5, 0.5])
t_2 = jnp.linspace(0., 10., 200)

def f2(t,x):
    return 0.5 * x*(1-x) * (1- 2*x)

def g2(t,x):
    return x * (1-x)

def f2_true(Wt,t,x0):
    return 1/(1 + jnp.exp(-Wt + jnp.log(x0/(1-x0)).reshape(-1,2)))







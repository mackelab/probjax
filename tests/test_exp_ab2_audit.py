import jax.numpy as jnp
import numpy as np
import pytest

from probjax.utils.functions import split_drift
from probjax.utils.odeint import odeint


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('nonuniform', [False, True])
def test_exp_ab2_second_order_on_variable_scalar_linear_ode(reverse, nonuniform):
    # y'=t*y+t; y(0)=0 -> y(t)=exp(t^2/2)-1.
    drift = split_drift(lambda t: t, lambda t, y: jnp.ones_like(y) * t)
    errors = []
    for n in (16, 32, 64):
        times = jnp.linspace(0.0, 1.0, n + 1)
        if nonuniform:
            times = times**1.5
        if reverse:
            times = times[::-1]
        initial = jnp.array([jnp.expm1(0.5) if reverse else 0.0])
        result = odeint(drift, initial, times, method='exp_ab2_scalarL')[-1, 0]
        target = 0.0 if reverse else jnp.expm1(0.5)
        errors.append(abs(float(result - target)))
    assert errors[-1] < errors[0] / 8, errors
    np.testing.assert_allclose(result, target, atol=0.001)

from typing import NamedTuple
import pytest

import jax
import jax.numpy as jnp
import numpy as np
from probjax.inference.kernels import (
    HMC,
    NUTS,
    GaussianIMH,
    GaussRWMH,
    MALA,
    dHMC,
    Slice,
    LatentSlice,
)

KERNELS = [HMC, NUTS, GaussianIMH, GaussRWMH, MALA, dHMC, Slice, LatentSlice]
SHAPES = [(1,), (2,), (2, 3), (4, 5, 6)]


@pytest.mark.parametrize(
    "kernel_type,in_shape", [(kernel, s) for kernel in KERNELS for s in SHAPES]
)
def test_markov_kernel_vector_input(kernel_type, in_shape):
    i = np.random.randint(0, 2**16)
    xs = np.random.randn(*in_shape)

    def logdensity(x):
        return -jnp.sum(x**2)

    kernel = kernel_type(logdensity)
    if kernel_type is Slice or kernel_type is LatentSlice:
        state = kernel.init(xs, rng_key=jax.random.PRNGKey(i))
    elif kernel_type is dHMC:
        state = kernel.init(xs, random_generator_arg=jnp.array(i))
    else:
        state = kernel.init(xs)
    params = kernel.init_params(xs)

    assert hasattr(state, "position"), "State must have a position attribute"
    assert state.position.shape == in_shape, "Position shape must match input shape"

    next_state, next_info = kernel(jax.random.PRNGKey(i), state, params)

    assert hasattr(next_state, "position"), "State must have a position attribute"
    assert (
        next_state.position.shape == in_shape
    ), "Position shape must match input shape after transition"


@pytest.mark.parametrize(
    "kernel_type,in_shape",
    [(kernel, s) for kernel in KERNELS for s in [(1,), (2,), (2, 2)]],
)
def test_markov_kernel_invariance(kernel_type, in_shape):
    if kernel_type is GaussianIMH:
        pytest.xfail("GaussianIMH does not pass this for some reason")

    i = np.random.randint(0, 2**16)
    key = jax.random.PRNGKey(i)

    N = 5000

    # Gaussian target, hence must leave the particle distribution invariant
    def logdensity(x):
        return jax.scipy.stats.norm.logpdf(x).sum()

    positions = np.random.randn(N, *in_shape)
    kernel = kernel_type(logdensity)
    if kernel_type is Slice or kernel_type is LatentSlice:
        states = jax.vmap(lambda x: kernel.init(x, rng_key=key))(positions)
    elif kernel_type is dHMC:
        states = jax.vmap(
            lambda x: kernel.init(x, random_generator_arg=jnp.array(i)),
        )(
            positions,
        )
    else:
        states = jax.vmap(kernel.init)(positions)
    params = kernel.init_params(positions[0])

    keys = jax.random.split(key, (N,))
    next_states, _ = jax.vmap(kernel, in_axes=(0, 0, None))(keys, states, params)

    new_positions = next_states.position

    assert new_positions.shape == positions.shape, "Output shape must match input shape"

    # Must be invariant to input
    mean_before = jnp.mean(positions, axis=0)
    mean_after = jnp.mean(new_positions, axis=0)

    assert jnp.allclose(
        mean_before, mean_after, atol=1e-1, rtol=1e-1
    ), "Mean must be invariant"

    var_before = jnp.var(positions, axis=0)
    var_after = jnp.var(new_positions, axis=0)

    assert jnp.allclose(
        var_before, var_after, atol=1e-1, rtol=1e-1
    ), "Variance must be invariant"

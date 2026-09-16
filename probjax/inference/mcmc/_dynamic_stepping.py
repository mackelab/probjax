"""Shared dynamic trajectory-length stepping for MCLMC and dynamic HMC."""

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array


def halton_next_random_arg_fn(index: Array):
    return jnp.array(index + 1, dtype=jnp.int32)


def random_next_random_arg_fn(random_arg: Array):
    return jax.random.split(random_arg)[1]


def halton_trajectory_length_fns(length_fn, average):
    """Halton-sequence stepping with a caller-supplied length function."""

    def halton_next_integration_steps_fn(random_arg: Array, **kwargs):
        return length_fn(random_arg, average)

    return (
        halton_next_random_arg_fn,
        halton_next_integration_steps_fn,
    )


def random_trajectory_length_fns(average):
    """Random stepping with uniform lengths in ``[1, 2 * average)``."""

    def random_next_integration_steps_fn(random_arg: Array, **kwargs):
        return jax.random.randint(random_arg, shape=(), minval=1, maxval=2 * average)

    return (
        random_next_random_arg_fn,
        random_next_integration_steps_fn,
    )


def get_dynamic_stepping(integration_steps_sequence, average, *, length_fn):
    """Resolve ``(random_arg_next_fn, integration_steps_fn)`` by sequence kind."""
    if isinstance(integration_steps_sequence, str):
        if integration_steps_sequence == "halton":
            random_arg_next_fn, integration_steps_fn = halton_trajectory_length_fns(
                length_fn, average
            )
        elif integration_steps_sequence == "random":
            random_arg_next_fn, integration_steps_fn = random_trajectory_length_fns(
                average
            )
        else:
            raise ValueError(
                "integration_steps_sequence must be 'halton', 'random', or a tuple"
            )
    else:
        random_arg_next_fn, integration_steps_fn = integration_steps_sequence

    return random_arg_next_fn, integration_steps_fn


def init_dynamic_arg(rng_key, integration_steps_sequence):
    """Initial random-generator argument for dynamic stepping."""
    if integration_steps_sequence == "random":
        return rng_key
    return jax.random.randint(
        rng_key, shape=(), minval=0, maxval=2**31 - 1, dtype=jnp.int32
    )

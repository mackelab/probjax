"""Shared dynamic trajectory-length stepping for MCLMC and dynamic HMC."""

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array


def halton_next_random_arg_fn(index: Array):
    return jnp.array(index + 1, dtype=jnp.int32)


def random_next_random_arg_fn(random_arg: Array):
    # The state carries an int32 counter (the same kind of argument the Halton
    # sequence uses), but an explicit sequence pair may carry a PRNG key.
    if jnp.issubdtype(jnp.result_type(random_arg), jnp.integer):
        return random_arg + jnp.asarray(1, random_arg.dtype)
    return jax.random.split(random_arg)[1]


def _as_key(random_arg: Array):
    """A PRNG key for ``random_arg``, folding an integer counter if needed."""
    if jnp.issubdtype(jnp.result_type(random_arg), jnp.integer):
        return jax.random.fold_in(jax.random.key(0), random_arg)
    return random_arg


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
        return jax.random.randint(
            _as_key(random_arg), shape=(), minval=1, maxval=2 * average
        )

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


def init_dynamic_arg(rng, integration_steps_sequence):
    """Initial random-generator argument for dynamic stepping.

    Both sequences carry an int32 counter: the Halton step uses it as the
    sequence index, and the random step folds it into a key. A PRNG key cannot
    be stored here because it reports an int32 scalar type inside a jitted
    kernel, which makes ``jax.random`` reject it (see ``_as_key``).
    """
    del integration_steps_sequence
    return jax.random.randint(
        rng, shape=(), minval=0, maxval=2**31 - 1, dtype=jnp.int32
    )

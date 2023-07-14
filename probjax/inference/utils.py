from typing import Callable, Union, Tuple
from jaxtyping import PyTree, Array

import jax
import jax.numpy as jnp
import numpy as np
from functools import wraps, partial


def flatten_fun(fun: Callable, in_tree: PyTree) -> Callable:
    """Flattens a function and its input tree."""

    @wraps(fun)
    def flat_fun(*args):
        input = jax.tree_util.tree_unflatten(in_tree, args)
        return fun(input)

    return flat_fun


def flatten_and_concat_fun(fun: Callable, in_vals: PyTree):
    """Process the potential function to return a function that takes in a single argument."""
    leaves, in_tree = jax.tree_util.tree_flatten(in_vals)

    # Casting to tuple to make them hashable -> static_argnums
    shapes = jax.tree_map(lambda x: jnp.shape(x), leaves)
    lengths = jax.tree_map(lambda x: jnp.size(x), leaves)
    cum_lengths = tuple(np.cumsum(lengths))[:-1]

    # Has only a single argument!
    def _flatten(*args):
        leaves, _ = jax.tree_util.tree_flatten(args)
        flatten_leaves = jax.tree_map(lambda x: jnp.ravel(x), leaves)
        return jnp.concatenate(flatten_leaves)

    def _unflatten(x):
        flattened_leaves = jnp.split(x, cum_lengths)
        leaves = jax.tree_map(
            lambda x, s: jnp.reshape(x, s), tuple(flattened_leaves), shapes
        )
        return jax.tree_util.tree_unflatten(in_tree, leaves)

    def _flatten_fun(x):
        x = _unflatten(x)
        return fun(*x)

    return _flatten, _unflatten, _flatten_fun


def sliced_potential_fn(flatten_potential_fn, loc, direction):
    """Returns a function that slices the potential function in a given direction."""

    def _sliced_potential_fn(t):
        return flatten_potential_fn(loc + t * direction)

    return _sliced_potential_fn


def conditional_potential_fn(flatten_potential_fn, x, indices):
    """Returns a function that slices the potential function in a given direction."""

    def _conditional_potential_fn(sub_x):
        return flatten_potential_fn(x.at[indices].set(sub_x))

    return _conditional_potential_fn

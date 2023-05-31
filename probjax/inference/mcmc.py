
import jax 
import jax.numpy as jnp
import jax.random as jrandom
from .mcmc_kernels import init_state, MCMCKernel

from typing import Any, Callable, Tuple, Union
from jaxtyping import PyTree, Array

from functools import partial
from itertools import accumulate

def flatten_potential_fn(potential_fn: Callable, in_vals: PyTree):
    """Process the potential function to return a function that takes in a single argument."""
    leaves, in_tree = jax.tree_util.tree_flatten(in_vals)

    # Casting to tuple to make them hashable -> static_argnums
    shapes = tuple(jax.tree_map(lambda x: jnp.shape(x), leaves))
    lengths = jax.tree_map(lambda x: jnp.size(x), leaves)
    cum_lengths = tuple(accumulate(lengths))[:-1]

    def _flatten(x):
        leaves, _ = jax.tree_util.tree_flatten(x)
        flatten_leaves = jax.tree_map(lambda x: jnp.ravel(x), leaves)
        return jnp.concatenate(flatten_leaves)
    
    @partial(jax.jit, static_argnums=(1,2))
    def _unflatten(x, cum_lengths, shapes):
        flattened_leaves = jnp.split(x, cum_lengths)
        leaves = jax.tree_map(lambda x, s: jnp.reshape(x, s), tuple(flattened_leaves), shapes)
        return jax.tree_util.tree_unflatten(in_tree, leaves)

    def _potential_fn(x):
        x = _unflatten(x, cum_lengths, shapes)
        print(x)
        return potential_fn(*x)
    return _flatten, _unflatten, _potential_fn


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


# Track statistics of the chain

class MCMC():

    def __init__(self, kernel, potential_fn, init_vals) -> None:
        
        self.kernel = kernel
        self.potential_fn = potential_fn
        self.init_vals = init_vals

    def init_state(self, key):
        return init_state(key, self.init_vals)
    
    def update_state(self, state):
        out = jax.tree_map(lambda x: self.kernel(*x), state, is_leaf=lambda x: isinstance(x, tuple))

    @partial(jax.jit, static_argnums=(0,))
    def run(self, key, num_steps):
        state = init_state(key, self.init_vals)
        out = jax.lax.fori_loop(0, num_steps, lambda i, x: update_state(self.kernel, x), state)
        return out
    

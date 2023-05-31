from typing import Any, Callable
from jaxtyping import PyTree, Array
from jaxtyping.random import PRNGKey

import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import KeyArray

from abc import abstractmethod
from functools import partial

from probjax.inference.mcmc_kernels import MCMCState


# This code is used to represent the state of the MCMC chain.
# It is used by the MCMC kernel to keep track of the current state of the chain.
# The state consists of a random number generator key, a value x, and a dictionary of statistics and paramaters.


@register_pytree_node_class
class MCMCState:
    """MCMC state object."""

    def __init__(
        self, key: KeyArray, x: Array, stats: dict = {}, params: dict = {}
    ) -> None:
        """This class represents the state of the MCMC chain.

        Args:
            key (KeyArray): A random number generator key.
            x (Array): Current value of the chain.
            stats (dict, optional): Statistics that are tracked for the chain. Defaults to {}.
            params (dict, optional): Adaptive parameters for the MCMCKernel. Defaults to {}.
        """
        self.key = key
        self.x = x
        self.params = params
        self.stats = stats

    def __repr__(self) -> str:
        return f"MCMCState(key={self.key}, x={self.x})"

    # Jax stuff
    def tree_flatten(self):
        return (self.key, self.x, self.stats, self.params), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class MCMCKernel:
    def __init__(self, potential_fn: Callable) -> None:
        self.potential_fn = potential_fn

    def __call__(self, state: PyTree[MCMCState]) -> PyTree[MCMCState]:
        new_state = jax.tree_map(
            self._update_mcmc_state, state, is_leaf=lambda x: isinstance(x, MCMCState)
        )
        return new_state

    def _update_mcmc_state(self, state: MCMCState) -> MCMCState:
        key1, key2 = jrandom.split(state.key)
        x_new = self._sample(key1, state.x, **state.params)
        stats = self.update_stats(state.x, x_new, state.stats, state.params)
        params = self.update_params(state.x, x_new, state.stats, state.params)
        new_state = MCMCState(key2, x_new, stats, params)
        return new_state

    @abstractmethod
    def _sample(self, key, x, **params) -> Array:
        pass

    def init_params(self) -> dict:
        return {}

    def init_stats(self) -> dict:
        return {}

    def update_params(self, x, x_new, stats, params) -> dict:
        return params

    def update_stats(self, x, x_new, stats, params) -> dict:
        return stats


class MHMCMCKernel(MCMCKernel):
    """Metropolis Hastings MCMC Kernel."""

    symmetric: bool = True

    def __call__(self, state: PyTree[MCMCState]) -> PyTree[MCMCState]:
        new_state = super().__call__(state)
        log_acceptance_probability = self._log_acceptance_probability(state, new_state)
        

    def _log_acceptance_probability(self, old_state: PyTree[MCMCState], new_state: PyTree[MCMCState]):
        vals_flat_old, _ = jax.tree_map(
            old_state.x, old_state, is_leaf=lambda x: isinstance(x, MCMCState)
        )
        vals_flat_new, _ = jax.tree_map(
            new_state.x, new_state, is_leaf=lambda x: isinstance(x, MCMCState)
        )
        log_potential_diff = self.potential_fn(*vals_flat_new) -  self.potential_fn(*vals_flat_old)
        if self.symmetric:
            raise NotImplementedError()
        return log_potential_diff
    
    

    # Only required if not symmetric
    def log_potential(self, x, **params):
        pass 


class GaussianKernel(MCMCKernel):
    def _sample(self, key, x, step_size=0.1):
        return x + jrandom.normal(key, shape=x.shape) * step_size


def kernelize(f):
    def kernelized_f(x, state):
        (key, num_steps, params), tree = jax.tree_util.tree_flatten(
            state, is_leaf=lambda x: not isinstance(x, MCMCState)
        )
        key1, key2 = jrandom.split(key)
        x_new = f(x, key1, **params)
        new_state = jax.tree_util.tree_unflatten(tree, (key2, num_steps + 1, params))
        return x_new, new_state

    return kernelized_f


@kernelize
def gaussian_kernel(x, key, step_size=0.1):
    return x + jrandom.normal(key, shape=x.shape) * step_size


def init_state(key, vars):
    children, tree = jax.tree_util.tree_flatten(vars)
    num_keys = len(children)
    keys = jrandom.split(key, num_keys)

    state = jax.tree_map(lambda x, k: (x, MCMCState(k)), vars, tree.unflatten(keys))
    return state


@partial(jax.jit, static_argnums=(0,))
def update_state(kernel, state):
    out = jax.tree_map(
        lambda x: kernel(*x), state, is_leaf=lambda x: isinstance(x, tuple)
    )
    # Metripolis Hastings
    log_accept_ratio = 0

    # Parameter updater if adaptive
    return out

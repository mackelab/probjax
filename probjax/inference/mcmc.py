import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax._src.util import safe_map as map
from .mcmc_kernels import MCMCKernel, MCMCState

from typing import Any, Callable, Tuple, Union, Sequence
from jaxtyping import PyTree, Array

from functools import partial
from itertools import accumulate



# Track statistics of the chain
def unzip_vals(states: PyTree[MCMCState] | MCMCState) -> PyTree[Array] | Array:
    """Unzips the states into a tuple of (x, key)"""
    x = jax.tree_map(lambda x: x.x, states, is_leaf=lambda x: isinstance(x, MCMCState))
    return x

def init_state(key, vars: Sequence[PyTree[Array] | Array]) -> PyTree[MCMCState] | MCMCState:
    children, tree = jax.tree_util.tree_flatten(vars)
    num_keys = len(children)
    keys = jrandom.split(key, num_keys)

    state = jax.tree_map(lambda x, k: MCMCState(k,x), vars, tree.unflatten(keys))
    return state



class MCMC:
    def __init__(
        self,
        kernel: PyTree[MCMCKernel] | MCMCKernel,
        potential_fn: Callable[[PyTree[Array] | Array], Array],
        init_vals: Sequence[PyTree[Array] | Array],
    ) -> None:
        self.kernel = kernel
        self.potential_fn = potential_fn
        self.init_vals = init_vals

        # Todo also check if kernel PyTree is identical to init_vals PyTree
        try:
            self.potential_fn(*init_vals)
        except:
            assert False, "Potential function must evaluatable given init_vals as input."

    def _requires_metropolis_hastings(self) -> bool:
        flatten_kernels, _ = jax.tree_util.tree_flatten(self.kernel)
        out = map(lambda x: x.requires_mh, flatten_kernels)
        return any(out)
    
    def _is_symmetric(self) -> bool:
        flatten_kernels, _ = jax.tree_util.tree_flatten(self.kernel)
        out = map(lambda x: x.symmetric, flatten_kernels)
        return all(out)
    
    def _set_potentail_if_required(self):
        flatten_kernels, _ = jax.tree_util.tree_flatten(self.kernel)
        for kernel in flatten_kernels:
            if kernel.requires_potential:
                kernel.set_potential_fn(self.potential_fn)
    
    def _mh_hastings_logratio(self, val_old: PyTree[Array] | Array, val_new: PyTree[Array] | Array, is_symmetric: bool) -> Array:
        logratio =  self.potential_fn(*val_new) - self.potential_fn(*val_old)
        if is_symmetric:
            return jnp.clip(logratio, a_max=0)
        else:
            pass

    @partial(jax.jit, static_argnums=(0,))
    def run(self, key, num_steps: int):
        # Initialize the state
        key_accept, key_state = jrandom.split(key)
        state = init_state(key_state, self.init_vals)

        # MCMC kernel requirements
        requires_mh = self._requires_metropolis_hastings()
        is_symmetric = self._is_symmetric()
        self._set_potentail_if_required()

        def body_fn(i, carry):
            state, key_accept = carry
            new_state = jax.tree_map(lambda kernel, x: kernel(x), self.kernel, state)
            if requires_mh:
                # Unzip the states
                val_old, val_new = unzip_vals((state, new_state))
                # Metropolis Hastings
                logratio = self._mh_hastings_logratio(val_old, val_new, is_symmetric)
                key, key_accept = jrandom.split(key_accept)
                accept = jnp.log(jrandom.uniform(key, logratio.shape)) < logratio
                # Update the state
                val = jax.tree_map(lambda v_new, v_old: jnp.where(accept,v_new, v_old), val_new, val_old)
                new_state = jax.tree_map(lambda s, v: s.set_x(v), new_state, val, is_leaf=lambda x: isinstance(x, MCMCState))

            return new_state, key_accept

        out_state, key = jax.lax.fori_loop(0, num_steps, body_fn, (state, key_accept))
        vals = unzip_vals(out_state)

        return vals

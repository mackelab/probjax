
from chex import PRNGKey
from jaxtyping import Array, PyTree


from typing import Any, Callable, NamedTuple, Optional, Tuple
from jax.tree_util import register_pytree_node_class

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jax.scipy.stats as stats
import numpy as np

from functools import partial
import matplotlib.pyplot as plt


import blackjax

from blackjax.base import State, Info

from probjax.inference.kernels.base import MCMCKernel

class GibbsKernel(MCMCKernel):
    
    def __init__(self, logdensity_fn: Callable, inner_kernels: dict | MCMCKernel) -> None:
        super().__init__()
        self.logdensity_fn = logdensity_fn
        self.inner_kernels = inner_kernels
        
    def init_state(self, position: dict) -> dict[State]:
        state = {}
        for k in position.keys():
            def logdensity_k(value):
                kwargs = {_k: position[_k] for _k in position.keys()}
                kwargs[k] = value
                return self.logdensity_fn(**kwargs)
            
            self.inner_kernels[k].logdensity_fn = logdensity_k
            state[k] = self.inner_kernels[k].init_state(position[k])
        return state
            
    
    def __call__(self, key: PRNGKey, state: dict[State]) -> Tuple:
        rng_keys = jax.random.split(key, num=len(state))
        rng_keys = dict(zip(state.keys(), rng_keys))

        state = state.copy()
        info = {}
        
        for k in state.keys():
            def logdensity_k(value):
                kwargs = {_k: state[_k].position for _k in state.keys()}
                kwargs[k] = value
                return self.logdensity_fn(**kwargs)
            
            self.inner_kernels[k].logdensity_fn = logdensity_k
            new_state_k, new_info_k = self.inner_kernels[k](
                rng_keys[k],
                state[k],
            )
            state[k] = new_state_k
            info[k] = new_info_k
        
        return state, info

# def mwg_kernel_general(rng_key, state, logdensity_fn, step_fn, init, parameters):
#     rng_keys = jax.random.split(rng_key, num=len(state))
#     rng_keys = dict(zip(state.keys(), rng_keys))

#     # avoid modifying argument state as JAX functions should be pure
#     state = state.copy()

#     for k in state.keys():
#         # logdensity of component k conditioned on all other components in state
#         def logdensity_k(value):
#             kwargs = {_k: state[_k].position for _k in state.keys()}
#             kwargs[k] = value
#             return logdensity_fn(**kwargs)

#         # give state[k] the right log_density
#         state[k] = init[k](
#             position=state[k].position,
#             logdensity_fn=logdensity_k
#         )

#         # update state[k]
#         state[k], _ = step_fn[k](
#             rng_key=rng_keys[k],
#             state=state[k],
#             logdensity_fn=logdensity_k,
#             **parameters[k]
#         )

#     return state
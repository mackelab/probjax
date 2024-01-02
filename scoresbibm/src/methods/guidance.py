import jax 
import jax.numpy as jnp

from functools import wraps


def unconditional_score_fn(base_fn,params ,t ,x ,node_id ,condition_mask ,edge_mask=None):
    # Zero out condition_mask
    condition_mask = jnp.zeros_like(condition_mask)
    return base_fn(params,t,x,node_id,condition_mask,edge_mask)

import jax 
import jax.numpy as jnp

from probjax.utils.sdeint import register_method
from functools import partial


def register_repaint_step_fn(model, condition_mask, x_o):
    
    def step_fn(drift, diffusion, t0, y0, f0, g0, dt, dWt, dWtdWs, is_diagonal):
        if is_diagonal:
            reduction = "i,i -> i"
        else:
            reduction = "ij, j -> i"
            
        x_t_mean = model.sde.marginal_mean(1-(t0+ dt), x_o)
        x_t_std = model.sde.marginal_stddev(1-(t0 + dt), x_o)
        x_t = x_t_mean + x_t_std * dWt[condition_mask] / jnp.sqrt(dt)

        y1 = y0 + dt * f0 + jnp.einsum(reduction, g0, dWt)
        y1 = y1.at[condition_mask].set(x_t)
        
        f1 = drift(t0 + dt, y1)
        g1 = diffusion(t0 + dt, y1)
        
        return y1, f1, g1, None 
    
    register_method("repaint", step_fn,{})
    
def register_classifier_free_guidance(model, old_condition_mask, x_o, likelihood_scale=3., prior_scale=1.):
    
    def classifier_free_score_fn(params ,t ,x ,node_id ,condition_mask ,meta_data=None,edge_mask=None):
        # Zero out condition_mask
        unconditional_score = model.model_fn(params,t,x,node_id,jnp.zeros_like(condition_mask),meta_data=meta_data,edge_mask=edge_mask)
        x_conditional = x.at[...,old_condition_mask,0].set(x_o.reshape(-1))
        conditional_score = model.model_fn(params,t,x_conditional,node_id,old_condition_mask,meta_data=meta_data,edge_mask=edge_mask)
        condition_mask = old_condition_mask.reshape(unconditional_score.shape)
        likelihood_part = (conditional_score - unconditional_score)*(1 - condition_mask)
        prior_part = unconditional_score
        
        return likelihood_scale * likelihood_part + prior_scale * prior_part
    
    model.score_fn = classifier_free_score_fn
        
        

def register_naive_inpaint_guidance(model, condition_mask, x_o):
    def unconditional_score_fn(params ,t ,x ,node_id ,condition_mask ,meta_data=None,edge_mask=None):
        # Zero out condition_mask
        condition_mask = jnp.zeros_like(condition_mask)
        return model.model_fn(params,t,x,node_id,condition_mask,meta_data=meta_data,edge_mask=edge_mask)
    model.score_fn = unconditional_score_fn
    

def register_generalized_guidance(model, condition_mask, x_o, score_manipulator="conditional", **score_manipulator_kwargs):
    score_manipulator = get_score_manipulator_fn(score_manipulator, **score_manipulator_kwargs)
    def additive_score_fn(params ,t ,x ,node_id ,local_condition_mask ,meta_data=None,edge_mask=None):
        # Zero out condition_mask
        joint_score = model.model_fn(params,t,x,node_id,local_condition_mask | condition_mask,meta_data=meta_data,edge_mask=edge_mask)
        manipulation_score = score_manipulator(t,x,condition_mask,x_o)
        manipulation_score = manipulation_score.reshape(joint_score.shape)
        #print(joint_score, manipulation_score)  
        return joint_score + manipulation_score
    
    model.score_fn = additive_score_fn
    
def get_score_manipulator_fn(name, **kwargs):
    if name == "interval":
        return jax.grad(lambda *args: log_step_fn(*args, **kwargs).sum(), argnums=1)
    elif name == "linear":
        return jax.grad(lambda *args: log_linear_fn_approximation(*args, **kwargs).sum(), argnums=1)
    elif name == "conditional":
        return jax.grad(lambda *args: log_conditional_fn(*args, **kwargs).sum(), argnums=1)
    elif name == "polytope":
        return jax.grad(lambda *args: log_polytope_fn_approximation(*args, **kwargs).sum(), argnums=1)
    else:
        raise NotImplementedError(f"Score manipulator {name} not implemented")



def exp_power_scaling(t, scaling=10, max_steepness=50000, order=2):
    return jnp.exp(-t**order*scaling)*max_steepness


# Numerical stability! (Sigmoid is numerically unstable for large values) -> Use log_sigmoid
def log_step_fn(t,x,condition_mask, x_o,a,b, scaling_fn = exp_power_scaling):
    scale = scaling_fn(t)
    x = x.reshape(x.shape[0],-1)
    x1 = jax.nn.log_sigmoid(scale * (x - a)*condition_mask)
    x2 = jax.nn.log_sigmoid(scale * (b- x)*condition_mask)
    return x1 + x2

def log_linear_fn_approximation(t,x,condition_mask, x_o, a , scaling_fn = exp_power_scaling):
    scale = scaling_fn(t)
    x = x.reshape(x.shape[0],-1)
    x1 = jax.nn.log_sigmoid(scale * (jnp.sum(x * a, axis=1)))
    x2 = jax.nn.log_sigmoid(-scale * (jnp.sum(x * a, axis=1)))
    return x1 + x2

def log_conditional_fn(t,x, condition_mask, x_o, scaling_fn = exp_power_scaling):
    x = x.reshape(x.shape[0],-1)
    scale = scaling_fn(t)
    x_cond = x.at[...,condition_mask].set(x_o)
    x1 = scale * (-jnp.sum(jnp.abs(x - x_cond), axis=1))
    return x1
    
def smaller_equal_constraint_fn(t,x,condition_mask, x_o, constrating_fn, constrain_value, scaling_fn = exp_power_scaling):
    scale = scaling_fn(t)
    constraint = jax.nn.relu(scale * (constrating_fn(x) - constrain_value)).max(axis=-1)
    constraint = jax.nn.log_sigmoid(-constraint) 
    return constraint

def log_polytope_fn_approximation(t,x,condition_mask, x_o ,A , scaling_fn = exp_power_scaling):
    return smaller_equal_constraint_fn(t,x,condition_mask, x_o, lambda x: x@A.T, 1., scaling_fn=scaling_fn)

step_fn_score = jax.grad(lambda *args: log_step_fn(*args).sum(), argnums=1)
linear_fn_score = jax.grad(lambda *args: log_linear_fn_approximation(*args).sum(), argnums=1)
log_conditional_fn_score = jax.grad(lambda *args: log_conditional_fn(*args).sum(), argnums=1)
log_polytope_fn_score = jax.grad(lambda *args: log_polytope_fn_approximation(*args).sum(), argnums=1)
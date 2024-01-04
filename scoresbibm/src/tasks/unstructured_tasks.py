from scoresbibm.src.tasks.all_conditional_task import AllConditionalTask
import jax
import jax.numpy as jnp
import jax.random as jrandom

from probjax.utils.odeint import _odeint
from probjax.utils.jaxutils import ravel_args
from probjax.core import joint_sample, log_potential_fn,rv 
from probjax.distributions import Normal, Uniform, Independent
from probjax.distributions.transformed_distribution import TransformedDistribution
from probjax.inference.mcmc import MCMC 
from probjax.inference.marcov_kernels import HMCKernel, GaussianMHKernel, SliceKernel, LangevianMHKernel

from functools import partial

def drift_lotka_volterra(t, data, alpha,beta, gamma, delta):
    predator, prey = data
    d_predator = alpha * predator - beta * predator * prey
    d_prey = -gamma * prey + delta * predator * prey
    return d_predator, d_prey

def lotka_volterra(ts_dense):
        def model(key, ts1, ts2, ode_method="rk4", ode_ts_grid=ts_dense):
            key_theta0, key_theta1, key_theta2, key_theta3, key_predator, key_prey = jrandom.split(key, 6)
            prior = TransformedDistribution(Normal(jnp.zeros(1), jnp.ones(1)), lambda x: jax.nn.sigmoid(x)*2 + 1.)
            theta0 = rv(prior, name="theta0")(key_theta0)
            theta1 = rv(prior, name="theta1")(key_theta1)
            theta2 = rv(prior, name="theta2")(key_theta2)
            theta3 = rv(prior, name="theta3")(key_theta3)
            
            predator, prey = _odeint(drift_lotka_volterra, (1., 0.5), ode_ts_grid, theta0, theta1, theta2, theta3, method=ode_method)
            
            predator_observed_mean = jnp.interp(ts1, ode_ts_grid, predator)
            prey_observed_mean = jnp.interp(ts2, ode_ts_grid, prey)
            
            x0 = rv(Independent(Normal(predator_observed_mean, 0.01),1), name="x0")(key_predator)
            x1 = rv(Independent(Normal(prey_observed_mean, 0.01),1), name="x1")(key_prey)
        
        joint_sampler = joint_sample(model)
        var_names = ["theta0", "theta1", "theta2", "theta3", "x0", "x1"]
        
    

# class UnstructuredTask(AllConditionalTask):
    
#     def __init__(self, name: str, backend: str = "jax") -> None:
#         super().__init__(name, backend)
        
    
    
# class LotkaVolterraTask(UnstructuredTask):
    
#     def __init__(self, backend: str = "jax") -> None:
#         super().__init__(name="lotka_volterra", backend=backend)
        
#     def get_reference_sampler(self):
        
#         @partial(jax.vmap, in_axes = [0, None, None, None, None])
#         def sample_fn(key, condition_mask, x_o, *x_meta_data):

#             conditioned_names = [var_names[i] for i in range(len(var_names)) if condition_mask[i]]
#             conditioned_nodes = {var: val for var, val in zip(conditioned_names, x_o)}
            

#             init_vals = joint_sampler(key, *x_meta_data)

#             for var in conditioned_nodes:
#                 del init_vals[var]

#             init_vals_flat, unravel = ravel_args(init_vals)
#             potential_fn = log_potential_fn(joint_sampler, *x_meta_data)


#             @jax.jit
#             def potential_fn_wrapper(vals):
#                 vals = unravel(vals)
#                 return potential_fn(**vals, **conditioned_nodes)

#             kernel = HMCKernel(step_size=1e-8)
#             kernel2 = GaussianMHKernel(step_size=0.5)
#             state = kernel.init_state(key,init_vals_flat)
#             mcmc = MCMC(kernel, potential_fn_wrapper)
#             mcmc2 = MCMC(kernel2, potential_fn_wrapper)
#             samples, state = mcmc.run(state, 200)
#             samples, state = mcmc2.run(state, 2000)
#             return samples
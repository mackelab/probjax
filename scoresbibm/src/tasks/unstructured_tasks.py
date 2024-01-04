from scoresbibm.src.methods.models import AllConditionalReferenceModel
from scoresbibm.src.tasks.all_conditional_tasks import AllConditionalTask
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

def lotka_volterra(time_start = 0,time_end = 20, eval_time_points=200):

        def dense_meta_data():
            ts_dense = jnp.linspace(time_start, time_end, eval_time_points)
            meta_data = { "theta0": jnp.array([jnp.nan]),"theta1": jnp.array([jnp.nan]),"theta2": jnp.array([jnp.nan]),"theta3": jnp.array([jnp.nan]), "x0": ts_dense, "x1": ts_dense}
            return meta_data
            

        def model(key, ts1, ts2, ode_method="rk4", ode_ts_grid=jnp.linspace(time_start, time_end, eval_time_points)):
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
        
        
        var_names = ["theta0", "theta1", "theta2", "theta3", "x0", "x1"]
        
        return model, dense_meta_data, var_names
        
    

class UnstructuredTask(AllConditionalTask):
    
    def __init__(self, name: str, builder, backend: str = "jax") -> None:
        model, meta_data, var_names = builder()
        self.var_names = var_names
        self.model = model
        self.dense_meta_data = meta_data()
        self.joint_sampler = joint_sample(model)
        
        super().__init__(name, backend)
        
    def get_data(self, num_samples: int, rng=None):
        rngs = jax.random.split(rng, (num_samples,))
        required_meta_data = [self.dense_meta_data[var] for var in self.var_names if not jnp.isnan(self.dense_meta_data[var]).any()]
        samples = jax.vmap(self.joint_sampler, in_axes=(0,) + (None,)*len(required_meta_data))(rngs, *required_meta_data)
        thetas = jnp.concatenate([samples[var] for var in self.var_names if var.startswith("theta")], axis=-1)
        xs = jnp.concatenate([samples[var] for var in self.var_names if var.startswith("x")], axis=-1)
        dense_meta_data = jnp.concatenate([self.dense_meta_data[var] for var in self.var_names], axis=-1)
        data = {"thetas": thetas, "xs": xs, "metadata":dense_meta_data}

        return data
        
    
    def _prepare_for_mcmc(self, key, condition_mask, x_o, *x_meta_data):
        conditioned_names = [self.var_names[i] for i in range(len(self.var_names)) if condition_mask[i]]
        conditioned_nodes = {var: val for var, val in zip(conditioned_names, x_o)}
        

        init_vals = self.joint_sampler(key, *x_meta_data)

        for var in conditioned_nodes:
            del init_vals[var]

        init_vals_flat, unravel = ravel_args(init_vals)
        potential_fn = log_potential_fn(self.model, *x_meta_data)

        return init_vals_flat, potential_fn, unravel
        
    def _get_conditional_sample_fn(self):
        raise NotImplementedError

    def _get_joint_sample_fn(self):
        @partial(jax.vmap, in_axes = [0, None, None])
        def sample_fn(key, ts1, ts2, *args, **kwargs):
            samples = self.joint_sampler(key)
            return jnp.concatenate([samples[var] for var in self.var_names], axis=-1)

        return sample_fn

    def get_reference_sampler(self):
        conditional_sample_fn = self._get_conditional_sample_fn()
        joint_sample_fn = self._get_joint_sample_fn()

        def sample_fn_wrapper(num_samples, x_o, rng=None, condition_mask=None, **kwargs):
            rngs = jax.random.split(rng, (num_samples,))
            if jnp.any(condition_mask):
                samples = conditional_sample_fn(rngs, condition_mask, x_o)
            else:
                samples = joint_sample_fn(rngs)
            return samples

        model = AllConditionalReferenceModel(sample_fn_wrapper)
        model.set_default_node_id(self.var_names)
        return model
        
    
    
class LotkaVolterraTask(UnstructuredTask):
    
    def __init__(self, backend: str = "jax") -> None:
        super().__init__("lotka_volterra", lotka_volterra, backend)
        
    def _get_conditional_sample_fn(self):
        
        @partial(jax.vmap, in_axes = [0, None, None, None, None])
        def sample_fn(key, condition_mask, x_o, x_meta_data):

            init_vals_flat, potential_fn_wrapper, unravel = self._prepare_for_mcmc(key, condition_mask, x_o, *x_meta_data)

            kernel = HMCKernel(step_size=1e-8)
            kernel2 = GaussianMHKernel(step_size=0.5)
            state = kernel.init_state(key,init_vals_flat)
            mcmc = MCMC(kernel, potential_fn_wrapper)
            mcmc2 = MCMC(kernel2, potential_fn_wrapper)
            samples, state = mcmc.run(state, 200)
            samples, state = mcmc2.run(state, 2000)
            return samples
        
        return sample_fn
        
        
        
        
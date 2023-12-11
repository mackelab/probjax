from sbibm import get_task as _get_torch_task

import jax
import jax.numpy as jnp


class SBIBMTask:
    observations = range(1, 11)

    def __init__(self, name: str, backend: str = "torch") -> None:
        self.name = name
        self.backend = backend

    def get_prior(self):
        if self.backend == "torch":
            return _get_torch_task(self.name).get_prior_dist()
        else:
            raise NotImplementedError()

    def get_simulator(self):
        if self.backend == "torch":
            return _get_torch_task(self.name).get_simulator()
        else:
            raise NotImplementedError()
    
    def get_graph_mask(self):

        raise NotImplementedError()


    def get_thetas_xs(self, num_samples: int):
        try:
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            return thetas, xs
        except:
            old_backed = self.backend
            self.backend = "torch"
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            self.backend = old_backed
            if self.backend == "numpy":
                thetas = thetas.numpy()
                xs = xs.numpy()
            elif self.backend == "jax":
                thetas = jnp.array(thetas)
                xs = jnp.array(xs)
            return thetas, xs

    def get_observation(self, index: int):
        if self.backend == "torch":
            return _get_torch_task(self.name).get_observation(index)
        else:
            out = _get_torch_task(self.name).get_observation(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_reference_posterior_samples(self, index: int):
        if self.backend == "torch":
            return _get_torch_task(self.name).get_reference_posterior_samples(index)
        else:
            out = _get_torch_task(self.name).get_reference_posterior_samples(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_true_parameters(self, index: int):
        if self.backend == "torch":
            return _get_torch_task(self.name).get_true_parameters(index)
        else:
            out = _get_torch_task(self.name).get_true_parameters(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)


class VariableConditionalTask(SBIBMTask):
    pass


class LinearGaussian(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_linear", backend=backend)
        
    def get_graph_mask(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.eye((x_dim)), x_i_mask]])
        return base_mask.astype(jnp.bool_)



class MixtureGaussian(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_mixture", backend=backend)
        
    def get_graph_mask(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])

        return base_mask.astype(jnp.bool_)
        
    


class TwoMoons(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="two_moons", backend=backend)
        
    def get_graph_mask(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])

        return base_mask.astype(jnp.bool_)
        

class SLCP(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="slcp", backend=backend)
        
    def get_graph_mask(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_) 
        # TODO This could be triangular -> DAG
        x_i_dim = x_dim // 4
        x_i_mask = jax.scipy.linalg.block_diag(*tuple([jnp.tril(jnp.ones((x_i_dim,x_i_dim), dtype=jnp.bool_))]*4)) 
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim,x_dim))], [jnp.ones((x_dim, theta_dim)), x_i_mask]]) 
        return base_mask.astype(jnp.bool_)



def get_task(name: str, backend: str = "torch"):
    if name == "gaussian_linear":
        return LinearGaussian(backend=backend)
    elif name == "gaussian_mixture":
        return MixtureGaussian(backend=backend)
    elif name == "two_moons":
        return TwoMoons(backend=backend)
    elif name == "slcp":
        return SLCP(backend=backend)
    else:
        raise NotImplementedError()

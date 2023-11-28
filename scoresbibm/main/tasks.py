
from sbibm import get_task as _get_torch_task

import jax 
import jax.numpy as jnp

class Task:
    
    observations = range(1, 11)
    
    def __init__(self, name:str, backend:str = "torch") -> None:
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
        
    def get_thetas_xs(self, num_samples:int):
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
        
        
    def get_reference_posterior_samples(self, index:int):
        if self.backend == "torch":
            return _get_torch_task(self.name).get_reference_posterior_samples(index)
        else:
            out = _get_torch_task(self.name).get_reference_posterior_samples(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)
            
    def get_true_parameters(self, index:int):
        if self.backend == "torch":
            return _get_torch_task(self.name).get_true_parameters(index)
        else:
            out = _get_torch_task(self.name).get_true_parameters(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)
            
class LinearGaussian(Task):
    def __init__(self, backend:str = "torch") -> None:
        super().__init__(name="gaussian_linear", backend=backend)
        
class MixtureGaussian(Task):
    def __init__(self, backend:str = "torch") -> None:
        super().__init__(name="gaussian_mixture", backend=backend)

class TwoMoons(Task):
    def __init__(self, backend:str = "torch") -> None:
        super().__init__(name="two_moons", backend=backend)
        
class SLCP(Task):
    def __init__(self, backend:str = "torch") -> None:
        super().__init__(name="slcp", backend=backend)
    
    
    
    
def get_task(name:str, backend:str = "torch"):
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
        
        



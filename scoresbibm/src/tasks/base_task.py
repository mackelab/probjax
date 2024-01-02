

from abc import ABC, abstractmethod




class Task(ABC):
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        self.name = name
        self.backend = backend    
        
    @property
    def theta_dim(self):
        return self.get_theta_dim()
    
    @property
    def x_dim(self):
        return self.get_x_dim()
    
    def get_theta_dim(self):
        raise NotImplementedError()
    
    def get_x_dim(self):
        raise NotImplementedError()
    

    def get_base_mask_fn(self):
        raise NotImplementedError()
    
    
class InferenceTask(Task):
    
    observations = range(1, 11)
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        
    def get_prior(self):
        raise NotImplementedError()
        
    def get_simulator(self):
        raise NotImplementedError()
    
    def get_thetas_xs(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_observation(self, index: int):
        raise NotImplementedError()
    
    def get_reference_posterior_samples(self, index: int):
        raise NotImplementedError()
    
    def get_true_parameters(self, index: int):
        raise NotImplementedError()
    
    
class AllConditionalTask(Task):
    
    var_names: list[str]
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        
    def get_joint_sampler(self):
        raise NotImplementedError()
    
    def get_thetas_xs(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_observation_generator(self):
        raise NotImplementedError()
    
    def get_base_mask_fn(self):
        raise NotImplementedError()
        
    def get_reference_sampler(self):
        raise NotImplementedError()
    
    
class UnstructuredTask(AllConditionalTask):
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        
    def get_joint_sampler(self):
        raise NotImplementedError()
    
    def get_thetas_xs(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_observation_generator(self):
        raise NotImplementedError()
    
    def get_base_mask_fn(self):
        raise NotImplementedError()
        
    def get_reference_sampler(self):
        raise NotImplementedError()
    
    
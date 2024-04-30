import jax
import jax.numpy as jnp

from typing import Optional, Tuple, Any, Callable
import haiku as hk


class ExchangableLayer(hk.Module):
    """Exchangable layer for permutation invariant functions."""

    def __init__(
        self,
        output_dim: int,
        trial_net_builder: Optional[Callable] = None,
        summary_net_builder: Optional[Callable] = None,
        latent_dim: int = 100,
        aggregation_fn: Callable = jnp.sum,
        aggregation_dim: int = -1,
        name: str = "exchangable_layer",
    ):
        self.output_dim = output_dim
        self.trial_net_builder = trial_net_builder
        self.summary_net_builder = summary_net_builder
        self.latent_dim = latent_dim
        self.aggregation_fn = aggregation_fn
        self.aggregation_dim = aggregation_dim
        super().__init__(name=name)
        

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        
        z = self.trial_net(inputs)
        h = self.aggregation_fn(z, axis=self.aggregation_dim)
        return self.summary_net(h)
    
    
    @hk.transparent
    def trial_net(self, x):
        if self.trial_net_builder is not None:
            return self.trial_net_builder(self.latent_dim)(x)
        else:
            return hk.nets.MLP([50,50, self.latent_dim])(x)
    
    @hk.transparent
    def summary_net(self, x):
        if self.summary_net_builder is not None:
            return self.summary_net_builder(self.output_dim)(x)
        else:
            return hk.nets.MLP([50,50, self.output_dim])(x)
        

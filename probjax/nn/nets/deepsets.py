from typing import Callable, Optional

import flax.nnx as nnx
import jax.numpy as jnp


class DeepSet(nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        phi: nnx.Module | Callable,
        rho: nnx.Module | Callable,
        *,
        reduction: Callable = jnp.sum,
        axis: int = -2,
        phi_kwargs: Optional[dict] = None,
        rho_kwargs: Optional[dict] = None,
    ):
        self.phi = phi
        self.rho = rho
        self.reduction = reduction
        self.axis = axis
        self.phi_kwargs = phi_kwargs if phi_kwargs is not None else {}
        self.rho_kwargs = rho_kwargs if rho_kwargs is not None else {}

    def __call__(self, *args, **kwargs):
        # Apply phi to each element
        kwargs = {**self.phi_kwargs, **kwargs}
        phi_x = self.phi(*args, **kwargs)
        # Aggregate
        h = self.reduction(phi_x, axis=self.axis)
        # Apply rho
        return self.rho(h, **self.rho_kwargs)

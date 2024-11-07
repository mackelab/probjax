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
        """Initializes the DeepSets module. For permutation invariant functions.

        Args:
            phi (nnx.Module | Callable): A neural network module or callable function
                that processes individual elements of the input set.
            rho (nnx.Module | Callable): A neural network module or callable function
                that processes the aggregated output of phi.
            reduction (Callable, optional): A reduction function to aggregate the
                outputs of phi. Defaults to jnp.sum.
            axis (int, optional): The axis along which to apply the reduction.
                Defaults to -2.
            phi_kwargs (Optional[dict], optional): Additional keyword arguments to
                pass to the phi module or callable. Defaults to None.
            rho_kwargs (Optional[dict], optional): Additional keyword arguments to
                pass to the rho module or callable. Defaults to None.
        """

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

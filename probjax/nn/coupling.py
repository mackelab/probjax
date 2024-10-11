from typing import Callable, List

import haiku as hk
import jax.numpy as jnp
from jaxtyping import Array


class CouplingMLP(hk.Module):
    def __init__(
        self,
        split_index: int,
        bijector: Callable[[Array, Array], Array],
        num_bijector_params: int,
        context_dim: int | None,
        hidden_dims: List[int] = [
            50,
        ],
        name: str = "coupling_mlp",
        **kwargs,
    ):
        """This is a invertible MLP that splits the input into two parts and applies a bijector to the second part. The parameters of the bijector are conditioned on the first part of the input.

        Args:
            split_index (int): Where to split the array into two parts.
            bijector (Callable[[Array, Array], Array]): A bijector f: params, x -> y that takes in the parameters and the input and returns the transformed input.
            num_bijector_params (int): The number of paramters the bijector takes in.
            context (Array | None, optional): The context. Defaults to None.
            hidden_dims (List[int], optional): Hidden dimensions. Defaults to [ 50, ].
            name (str, optional): Name. Defaults to "coupling_mlp".
        """
        super().__init__(name=name)
        self.split_index = split_index
        self.context_dim = context_dim if context_dim is not None else 0
        self.bijector = bijector
        self.num_bijector_params = num_bijector_params
        self._hidden_dims = hidden_dims
        self._mlp_params = kwargs

    def __call__(self, x: Array, context: Array) -> Array:
        conditionor = hk.nets.MLP(
            [self.split_index + self.context_dim]
            + self._hidden_dims
            + [self.num_bijector_params],
            **self._mlp_params,
        )
        x1, x2 = jnp.split(x, [self.split_index], axis=-1)
        y1 = x1
        if context is not None:
            x1 = jnp.hstack([x1, context])
        params = conditionor(x1)
        y2 = self.bijector(params, x2)

        y = jnp.concatenate([y1, y2], axis=-1)
        return y

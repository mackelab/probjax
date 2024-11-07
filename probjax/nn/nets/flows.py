from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from flax import nnx
from matplotlib.patheffects import Normal

from probjax.distributions.independent import Independent
from probjax.distributions.transformed_distribution import TransformedDistribution
from probjax.nn.bijective import (
    affine_bijector,
    additive_bijector,
    rational_quadratic_spline,
)
from probjax.nn.nets.coupling import CouplingMLP
from probjax.nn.utils import Flip, Sequential


class AdditiveCouplingFlow(
    TransformedDistribution, nnx.Module, experimental_pytree=True
):
    def __init__(
        self,
        input_dim,
        num_transforms,
        rngs,
        last_transform=None,
        coupling_class=CouplingMLP,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = input_dim - split_dim
        coupling_net = partial(coupling_class, split_dim, params_dim, additive_bijector)

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [Flip()]
        transforms += [last_transform] if last_transform is not None else []

        self.transform = Sequential(*transforms)
        self.base_dist = nnx.Variable(Independent(Normal(self.mu0, self.std0), 1))

        super().__init__(self.base_dist, self.transform)


class AffineCouplingFlow(TransformedDistribution, nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        input_dim,
        num_transforms,
        rngs,
        last_transform=None,
        coupling_class=CouplingMLP,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = (input_dim - split_dim) * 2
        coupling_net = partial(coupling_class, split_dim, params_dim, affine_bijector)

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [Flip()]
        transforms += [last_transform] if last_transform is not None else []

        self.transform = Sequential(*transforms)
        self.base_dist = nnx.Variable(Independent(Normal(self.mu0, self.std0), 1))

        super().__init__(self.base_dist, self.transform)


class SplineCouplingFlow(TransformedDistribution, nnx.Module, experimental_pytree=True):
    def __init__(
        self,
        input_dim: int,
        num_bins: int,
        num_transforms: int,
        rngs,
        last_transform=None,
        coupling_class=CouplingMLP,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = (num_bins * 3) * (input_dim - split_dim)
        coupling_net = partial(
            coupling_class, split_dim, params_dim, rational_quadratic_spline
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [Flip()]
        transforms += [last_transform] if last_transform is not None else []

        self.transform = Sequential(*transforms)
        self.base_dist = nnx.Variable(Independent(Normal(self.mu0, self.std0), 1))

        super().__init__(self.base_dist, self.transform)

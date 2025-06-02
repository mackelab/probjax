from functools import partial
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.bijective import (
    additive_bijector,
    affine_bijector,
    rational_quadratic_spline,
)
from probjax.nn.nets.autoregressive import AutoregressiveMLP
from probjax.nn.nets.coupling import CouplingMLP
from probjax.nn.utils import Flip, Sequential
from probjax.stats.continuous import norm
from probjax.stats.transformed import transformed


class Flow(transformed, nnx.Module, experimental_pytree=True):
    def __init__(
        self, base_dist, transformation: Callable[..., Any], name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.base_dist = base_dist
        self.transformation = transformation

    def __call__(self, *args, **kwds):
        return self.freeze(self.base_dist, self.transformation, **kwds)


class AdditiveCouplingFlow(Flow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        last_transform: Optional[Callable] = None,
        coupling_class: nnx.Module = CouplingMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = input_dim - split_dim
        coupling_net = partial(coupling_class, split_dim, params_dim, additive_bijector)

        # Build the flow transformation
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = norm(mu0, std0)

        super().__init__(q0, transform, name=name)


class AffineCouplingFlow(Flow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        last_transform: Optional[Callable] = None,
        coupling_class: nnx.Module = CouplingMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
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
                transforms += [mixing_class(rngs=rngs)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = norm(mu0, std0)

        super().__init__(q0, transform, name=name)


class SplineCouplingFlow(Flow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        num_bins: int = 10,
        last_transform: Optional[Callable] = None,
        coupling_class: nnx.Module = CouplingMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        dims = input_dim - split_dim
        params_dim = (num_bins * 3) * dims
        spline = partial(
            rational_quadratic_spline,
            range_min_x=-10.0,
            range_max_x=10.0,
            range_min_y=-10.0,
            range_max_y=10.0,
        )

        def spline_fn(params, x):
            params = jnp.reshape(params, (dims, num_bins * 3))
            return jax.vmap(spline)(params, x)

        coupling_net = partial(coupling_class, split_dim, params_dim, spline_fn)

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = norm(mu0, std0)

        super().__init__(q0, transform, name=name)


class AdditiveAutoregressiveFlow(Flow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 1
        autoregressive = partial(
            autoregressive_class, input_dim, params_per_dim, additive_bijector
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [autoregressive(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = norm(mu0, std0)

        super().__init__(q0, transform, name=name)


class AffineAutoregressiveFlow(Flow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 2
        autoregressive = partial(
            autoregressive_class, input_dim, params_per_dim, affine_bijector
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [autoregressive(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = norm(mu0, std0)

        super().__init__(q0, transform, name=name)


class SplineAutoregressiveFlow(Flow):
    def __init__(
        self,
        input_dim: int,
        num_transforms: int,
        rngs,
        *,
        num_bins: int = 10,
        last_transform: Optional[Callable] = None,
        autoregressive_class: nnx.Module = AutoregressiveMLP,
        mixing_class: nnx.Module = Flip,
        name: Optional[str] = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 3 * num_bins
        spline = partial(
            rational_quadratic_spline,
            range_min_x=-10.0,
            range_max_x=10.0,
            range_min_y=-10.0,
            range_max_y=10.0,
        )

        def spline_fn(params, x):
            params = jnp.reshape(params, (self.input_dim, num_bins * 3))
            return jax.vmap(spline)(params, x)

        autoregressive = partial(
            autoregressive_class, input_dim, params_per_dim, spline_fn
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [autoregressive(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = norm(mu0, std0)

        super().__init__(q0, transform, name=name)

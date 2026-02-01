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
from probjax.nn.layers.bijective import Flip
from probjax.nn.nets.autoregressive import AutoregressiveMLP
from probjax.nn.nets.coupling import CouplingMLP
from probjax.nn.nets.simple import Sequential

from probjax.stats.continuous import norm
from probjax.stats.independent import independent
from probjax.stats.transformed import transformed


class NormalizingFlow(nnx.Module):
    def __init__(
        self,
        base_dist,
        transformation: Callable[..., Any],
        name: Optional[str] = None,
        *,
        sharding: jax.sharding.Mesh | None = None,
    ):
        self.base_dist = base_dist
        self.transformation = transformation
        self._mesh = sharding
        super().__init__()

    def transform(self, x):
        return self.transformation(x)

    def __call__(self, x):
        return self.transform(x)

    def sample(self, rng, shape=()):
        """Sample from the flow distribution."""
        return transformed.rvs(
            rng, shape=shape, base_dist=self.base_dist, bijector=self.transformation
        )

    def logpdf(self, x):
        return transformed.logpdf(
            x, base_dist=self.base_dist, bijector=self.transformation
        )


class AdditiveCouplingFlow(NormalizingFlow):
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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = input_dim - split_dim
        coupling_net = partial(
            coupling_class, split_dim, params_dim, additive_bijector, sharding=sharding
        )

        # Build the flow transformation
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


class AffineCouplingFlow(NormalizingFlow):
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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        split_dim = input_dim // 2
        params_dim = (input_dim - split_dim) * 2
        coupling_net = partial(
            coupling_class, split_dim, params_dim, affine_bijector, sharding=sharding
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


class SplineCouplingFlow(NormalizingFlow):
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
        sharding: jax.sharding.Mesh | None = None,
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

        coupling_net = partial(
            coupling_class, split_dim, params_dim, spline_fn, sharding=sharding
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [coupling_net(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


class AdditiveAutoregressiveFlow(NormalizingFlow):
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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 1
        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            additive_bijector,
            sharding=sharding,
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [autoregressive(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


class AffineAutoregressiveFlow(NormalizingFlow):
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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 2
        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            affine_bijector,
            sharding=sharding,
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [autoregressive(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)


class SplineAutoregressiveFlow(NormalizingFlow):
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
        sharding: jax.sharding.Mesh | None = None,
    ) -> None:
        self.input_dim = input_dim
        params_per_dim = 3 * num_bins * input_dim
        spline = partial(
            rational_quadratic_spline,
            range_min_x=-10.0,
            range_max_x=10.0,
            range_min_y=-10.0,
            range_max_y=10.0,
        )

        def spline_fn(params, x):
            return spline(params, x)

        autoregressive = partial(
            autoregressive_class,
            input_dim,
            params_per_dim,
            spline_fn,
            sharding=sharding,
        )

        # Build the flow
        transforms = []
        for i in range(num_transforms):
            transforms += [autoregressive(rngs=rngs)]
            if i < num_transforms - 1:
                transforms += [mixing_class(rngs=rngs, sharding=sharding)]
        transforms += [last_transform] if last_transform is not None else []

        transform = Sequential(*transforms, sharding=sharding)

        # Build the base distribution
        mu0 = jnp.zeros((input_dim,))
        std0 = jnp.ones((input_dim,))
        q0 = independent(norm(mu0, std0))

        super().__init__(q0, transform, name=name, sharding=sharding)

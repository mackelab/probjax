from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Tuple, runtime_checkable

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array, ArrayLike


@runtime_checkable
class InterpolationScheduleProtocol(Protocol):
    """Interpolation path + optional noise."""

    def interpolation_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array: ...

    def interpolation_noise_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None: ...

    def a_t(self, t: ArrayLike) -> Array: ...
    def b_t(self, t: ArrayLike) -> Array: ...

    def path_mean(self, t: ArrayLike, mu0: ArrayLike, mu1: ArrayLike) -> Array: ...
    def path_std(self, t: ArrayLike, std0: ArrayLike, std1: ArrayLike) -> Array: ...


@dataclass
class LinearInterpolationSchedule(InterpolationScheduleProtocol):
    """
    Default linear interpolation schedule with optional additive noise.
    """

    noise_fn: Callable[[ArrayLike, Array, Array], Array] | None = None

    def a_t(self, t: ArrayLike) -> Array:
        return 1.0 - jnp.asarray(t)

    def b_t(self, t: ArrayLike) -> Array:
        return jnp.asarray(t)

    def interpolation_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array:
        a = self.a_t(t)
        b = self.b_t(t)
        return a * x0 + b * x1

    def interpolation_noise_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None:
        if self.noise_fn is None:
            return None
        return self.noise_fn(t, x0, x1)

    def path_mean(self, t: ArrayLike, mu0: ArrayLike, mu1: ArrayLike) -> Array:
        return self.interpolation_fn(t, jnp.asarray(mu0), jnp.asarray(mu1))

    def path_std(self, t: ArrayLike, std0: ArrayLike, std1: ArrayLike) -> Array:
        a = self.a_t(t)
        b = self.b_t(t)
        return jnp.sqrt((a**2) * jnp.asarray(std0) ** 2 + (b**2) * jnp.asarray(std1) ** 2)


@dataclass
class CosineInterpolationSchedule(InterpolationScheduleProtocol):
    """
    Cosine-eased interpolation: a_t = cos(pi/2 * t), b_t = sin(pi/2 * t).
    """

    noise_fn: Callable[[ArrayLike, Array, Array], Array] | None = None

    def a_t(self, t: ArrayLike) -> Array:
        return jnp.cos(jnp.asarray(t) * jnp.pi / 2.0)

    def b_t(self, t: ArrayLike) -> Array:
        return jnp.sin(jnp.asarray(t) * jnp.pi / 2.0)

    def interpolation_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array:
        a = self.a_t(t)
        b = self.b_t(t)
        return a * x0 + b * x1

    def interpolation_noise_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None:
        if self.noise_fn is None:
            return None
        return self.noise_fn(t, x0, x1)

    def path_mean(self, t: ArrayLike, mu0: ArrayLike, mu1: ArrayLike) -> Array:
        return self.interpolation_fn(t, jnp.asarray(mu0), jnp.asarray(mu1))

    def path_std(self, t: ArrayLike, std0: ArrayLike, std1: ArrayLike) -> Array:
        a = self.a_t(t)
        b = self.b_t(t)
        return jnp.sqrt((a**2) * jnp.asarray(std0) ** 2 + (b**2) * jnp.asarray(std1) ** 2)


@dataclass
class QuadraticInterpolationSchedule(InterpolationScheduleProtocol):
    """
    Quadratic ease-out toward x1: a_t = (1 - t)^2, b_t = 1 - a_t.
    """

    noise_fn: Callable[[ArrayLike, Array, Array], Array] | None = None

    def a_t(self, t: ArrayLike) -> Array:
        return (1.0 - jnp.asarray(t)) ** 2

    def b_t(self, t: ArrayLike) -> Array:
        return 1.0 - self.a_t(t)

    def interpolation_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array:
        a = self.a_t(t)
        b = self.b_t(t)
        return a * x0 + b * x1

    def interpolation_noise_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None:
        if self.noise_fn is None:
            return None
        return self.noise_fn(t, x0, x1)

    def path_mean(self, t: ArrayLike, mu0: ArrayLike, mu1: ArrayLike) -> Array:
        return self.interpolation_fn(t, jnp.asarray(mu0), jnp.asarray(mu1))

    def path_std(self, t: ArrayLike, std0: ArrayLike, std1: ArrayLike) -> Array:
        a = self.a_t(t)
        b = self.b_t(t)
        return jnp.sqrt((a**2) * jnp.asarray(std0) ** 2 + (b**2) * jnp.asarray(std1) ** 2)


@runtime_checkable
class FlowPreconditioningProtocol(Protocol):
    """Preconditioning adapters used by flow matching models."""

    def normalize(
        self,
        schedule: InterpolationScheduleProtocol,
        t: ArrayLike,
        x: Array,
        mu0: ArrayLike,
        mu1: ArrayLike,
        std0: ArrayLike,
        std1: ArrayLike,
    ) -> Tuple[Array, Array, Array]: ...

    def velocity_scale(
        self,
        schedule: InterpolationScheduleProtocol,
        t: ArrayLike,
        std0: ArrayLike,
        std1: ArrayLike,
    ) -> Array: ...

    def decode_velocity(
        self,
        schedule: InterpolationScheduleProtocol,
        t: ArrayLike,
        x: Array,
        mu0: ArrayLike,
        mu1: ArrayLike,
        std0: ArrayLike,
        std1: ArrayLike,
        residual_pred: Array,
    ) -> Array: ...


@dataclass
class GaussianFlowPreconditioning(FlowPreconditioningProtocol):
    """
    Matches the closed-form Gaussian preconditioning in FlowMatcher.
    """

    eps: float = 1e-8

    def normalize(
        self,
        schedule: InterpolationScheduleProtocol,
        t: ArrayLike,
        x: Array,
        mu0: ArrayLike,
        mu1: ArrayLike,
        std0: ArrayLike,
        std1: ArrayLike,
    ) -> Tuple[Array, Array, Array]:
        mean_t = schedule.path_mean(t, mu0, mu1)
        std_t = jnp.maximum(schedule.path_std(t, std0, std1), self.eps)
        x_normed = jax.tree_util.tree_map(lambda xi, m: (xi - m) / std_t, x, mean_t)
        return x_normed, mean_t, std_t

    def velocity_scale(
        self,
        schedule: InterpolationScheduleProtocol,
        t: ArrayLike,
        std0: ArrayLike,
        std1: ArrayLike,
    ) -> Array:
        a = schedule.a_t(t)
        b = schedule.b_t(t)
        denom = (a**2) * jnp.asarray(std0) ** 2 + (b**2) * jnp.asarray(std1) ** 2
        return (b * jnp.asarray(std1) ** 2 - a * jnp.asarray(std0) ** 2) / jnp.maximum(
            denom, self.eps
        )

    def decode_velocity(
        self,
        schedule: InterpolationScheduleProtocol,
        t: ArrayLike,
        x: Array,
        mu0: ArrayLike,
        mu1: ArrayLike,
        std0: ArrayLike,
        std1: ArrayLike,
        residual_pred: Array,
    ) -> Array:
        mean_t = schedule.path_mean(t, mu0, mu1)
        std_t = jnp.maximum(schedule.path_std(t, std0, std1), self.eps)
        scale = self.velocity_scale(schedule, t, std0, std1)
        residual = jax.tree_util.tree_map(lambda r: std_t * r, residual_pred)
        drift = jax.tree_util.tree_map(lambda xi, m: scale * (xi - m), x, mean_t)
        delta_mean = jnp.asarray(mu1) - jnp.asarray(mu0)
        return jax.tree_util.tree_map(lambda res, dr: res + dr + delta_mean, residual, drift)


@runtime_checkable
class FlowTrainingConfigProtocol(Protocol):
    """Training-time sampling of interpolation times."""

    t_min: float
    t_max: float

    def sample_times(self, rng, shape: Tuple[int, ...]) -> Array: ...


@dataclass
class UniformFlowTrainingConfig(FlowTrainingConfigProtocol):
    t_min: float = 0.0
    t_max: float = 1.0

    def sample_times(self, rng, shape: Tuple[int, ...]) -> Array:
        return jax.random.uniform(
            rng, shape=shape + (1,), minval=self.t_min, maxval=self.t_max
        )


@dataclass
class LogitNormalFlowTrainingConfig(FlowTrainingConfigProtocol):
    """Matches the logistic sampling used by LinearFlow noise_schedule."""

    mu: float = 0.0
    scale: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def sample_times(self, rng, shape: Tuple[int, ...]) -> Array:
        raw = jax.random.normal(rng, shape=shape + (1,)) * self.scale + self.mu
        t = jax.nn.sigmoid(raw)
        return jnp.clip(t, self.t_min, self.t_max)


@dataclass
class SigmoidPairFlowTrainingConfig(FlowPairTrainingConfigProtocol):
    """
    Pair sampling mirroring MeanFlowMatcher.noise_schedule.
    """

    percent_rt: float = 0.25
    mu_rt: float = -0.4
    scale_rt: float = 1.0
    mu_t: float = 0.0
    scale_t: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def sample_times_pair(self, rng, shape: Tuple[int, ...]) -> tuple[Array, Array]:
        batch_size = shape[0]
        batch_size_different = int(batch_size * self.percent_rt)
        batch_size_same = batch_size - batch_size_different

        rng_t, rng_r, rng_tr = jax.random.split(rng, 3)

        t1 = jax.nn.sigmoid(
            jax.random.normal(rng_t, (batch_size_different,) + shape[1:] + (1,))
            * self.scale_rt
            - self.mu_rt
        )
        r1 = jax.nn.sigmoid(
            jax.random.normal(rng_r, (batch_size_different,) + shape[1:] + (1,))
            * self.scale_rt
            - self.mu_rt
        )
        r1 = jnp.clip(t1 + r1, a_min=self.t_min, a_max=self.t_max)

        t2 = r2 = jax.nn.sigmoid(
            jax.random.normal(rng_tr, (batch_size_same,) + shape[1:] + (1,))
            * self.scale_t
            - self.mu_t
        )

        t = jnp.concatenate([t1, t2], axis=0)
        r = jnp.concatenate([r1, r2], axis=0)

        t = jnp.clip(t, self.t_min, self.t_max)
        r = jnp.clip(r, self.t_min, self.t_max)
        return t, r


@runtime_checkable
class FlowSolverConfigProtocol(Protocol):
    """Inference-time schedule builder."""

    num_steps: int

    def solve_schedule(
        self, t_min: float, t_max: float, num_steps: int | None = None
    ) -> Array: ...


@runtime_checkable
class FlowPairTrainingConfigProtocol(Protocol):
    """Training-time sampling for (t, r) pairs (for mean flow matching)."""

    t_min: float
    t_max: float

    def sample_times_pair(self, rng, shape: Tuple[int, ...]) -> tuple[Array, Array]: ...


@dataclass
class LinearFlowSolverConfig(FlowSolverConfigProtocol):
    num_steps: int = 50

    def solve_schedule(
        self, t_min: float, t_max: float, num_steps: int | None = None
    ) -> Array:
        steps = self.num_steps if num_steps is None else num_steps
        if steps <= 1:
            return jnp.asarray(t_max)[None]
        return jnp.linspace(t_min, t_max, steps)


@dataclass
class RhoFlowSolverConfig(FlowSolverConfigProtocol):
    """
    Karras-style non-linear spacing on t to cluster steps near t_max.
    """

    num_steps: int = 64
    rho: float = 5.0

    def solve_schedule(
        self, t_min: float, t_max: float, num_steps: int | None = None
    ) -> Array:
        steps = self.num_steps if num_steps is None else num_steps
        if steps <= 1:
            return jnp.asarray(t_max)[None]
        u = jnp.linspace(0.0, 1.0, steps)
        warped = 1.0 - jnp.power(1.0 - u, self.rho)
        return t_min + warped * (t_max - t_min)

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Tuple, runtime_checkable

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array, ArrayLike


def _autodiff_time_gradient(
    fn: Callable[[ArrayLike, Array, Array], Array],
    t: ArrayLike,
    x0: Array,
    x1: Array,
) -> Array:
    grad_fn = jax.jacfwd(lambda time, x_s, x_t: fn(time, x_s, x_t), argnums=0)
    if jnp.ndim(x0) == 0:
        return grad_fn(t, x0, x1)
    if jnp.ndim(t) == 0 or jnp.shape(t)[0] != jnp.shape(x0)[0]:
        return jax.vmap(grad_fn, in_axes=(None, 0, 0))(t, x0, x1)
    return jax.vmap(grad_fn, in_axes=(0, 0, 0))(t, x0, x1)


@runtime_checkable
class InterpolationScheduleProtocol(Protocol):
    """Interpolation path + optional noise."""

    def interpolation_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array: ...

    def interpolation_noise_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None: ...

    def interpolation_velocity_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array: ...

    def interpolation_noise_velocity_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None: ...

    def a_t(self, t: ArrayLike) -> Array: ...
    def b_t(self, t: ArrayLike) -> Array: ...

    def path_mean(self, t: ArrayLike, mu0: ArrayLike, mu1: ArrayLike) -> Array: ...
    def path_std(self, t: ArrayLike, std0: ArrayLike, std1: ArrayLike) -> Array: ...


@dataclass
class GeneralInterpolationSchedule(InterpolationScheduleProtocol):
    """Generic schedule wrapper that derives velocities via autodiff."""

    interp_fn: Callable[[ArrayLike, Array, Array], Array]
    noise_fn: Callable[[ArrayLike, Array, Array], Array] | None = None
    interp_velocity_fn: Callable[[ArrayLike, Array, Array], Array] | None = None
    noise_velocity_fn: Callable[[ArrayLike, Array, Array], Array] | None = None
    a_t_fn: Callable[[ArrayLike], Array] | None = None
    b_t_fn: Callable[[ArrayLike], Array] | None = None
    path_mean_fn: Callable[[ArrayLike, ArrayLike, ArrayLike], Array] | None = None
    path_std_fn: Callable[[ArrayLike, ArrayLike, ArrayLike], Array] | None = None

    def interpolation_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array:
        return self.interp_fn(t, x0, x1)

    def interpolation_noise_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None:
        if self.noise_fn is None:
            return None
        return self.noise_fn(t, x0, x1)

    def interpolation_velocity_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array:
        def grad_fn(time, x_s, x_t):
            if self.interp_velocity_fn is not None:
                return self.interp_velocity_fn(time, x_s, x_t)
            return _autodiff_time_gradient(self.interp_fn, time, x_s, x_t)

        return grad_fn(t, x0, x1)

    def interpolation_noise_velocity_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None:
        if self.noise_fn is None:
            return None
        def grad_fn(time, x_s, x_t):
            if self.noise_velocity_fn is not None:
                return self.noise_velocity_fn(time, x_s, x_t)
            return _autodiff_time_gradient(self.noise_fn, time, x_s, x_t)

        return grad_fn(t, x0, x1)

    def a_t(self, t: ArrayLike) -> Array:
        if self.a_t_fn is None:
            raise NotImplementedError("a_t is not defined for this schedule.")
        return self.a_t_fn(t)

    def b_t(self, t: ArrayLike) -> Array:
        if self.b_t_fn is None:
            raise NotImplementedError("b_t is not defined for this schedule.")
        return self.b_t_fn(t)

    def path_mean(self, t: ArrayLike, mu0: ArrayLike, mu1: ArrayLike) -> Array:
        if self.path_mean_fn is not None:
            return self.path_mean_fn(t, mu0, mu1)
        return self.interpolation_fn(t, jnp.asarray(mu0), jnp.asarray(mu1))

    def path_std(self, t: ArrayLike, std0: ArrayLike, std1: ArrayLike) -> Array:
        if self.path_std_fn is None:
            raise NotImplementedError("path_std is not defined for this schedule.")
        return self.path_std_fn(t, std0, std1)


class AutodiffInterpolationSchedule(GeneralInterpolationSchedule):
    """Backward-compatible alias of GeneralInterpolationSchedule."""


@dataclass
class LinearInterpolationSchedule(InterpolationScheduleProtocol):
    """
    Default linear interpolation schedule with optional additive noise.
    """

    noise_fn: Callable[[ArrayLike, Array, Array], Array] | None = None
    noise_velocity_fn: Callable[[ArrayLike, Array, Array], Array] | None = None

    def a_t(self, t: ArrayLike) -> Array:
        return 1.0 - jnp.asarray(t)

    def b_t(self, t: ArrayLike) -> Array:
        return jnp.asarray(t)

    def da_dt(self, t: ArrayLike) -> Array:
        return -jnp.ones_like(jnp.asarray(t))

    def db_dt(self, t: ArrayLike) -> Array:
        return jnp.ones_like(jnp.asarray(t))

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

    def interpolation_velocity_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array:
        da = self.da_dt(t)
        db = self.db_dt(t)
        return da * x0 + db * x1

    def interpolation_noise_velocity_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None:
        if self.noise_fn is None:
            return None
        if self.noise_velocity_fn is not None:
            return self.noise_velocity_fn(t, x0, x1)
        return _autodiff_time_gradient(self.noise_fn, t, x0, x1)

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
    noise_velocity_fn: Callable[[ArrayLike, Array, Array], Array] | None = None

    def a_t(self, t: ArrayLike) -> Array:
        return jnp.cos(jnp.asarray(t) * jnp.pi / 2.0)

    def b_t(self, t: ArrayLike) -> Array:
        return jnp.sin(jnp.asarray(t) * jnp.pi / 2.0)

    def da_dt(self, t: ArrayLike) -> Array:
        return -(jnp.pi / 2.0) * jnp.sin(jnp.asarray(t) * jnp.pi / 2.0)

    def db_dt(self, t: ArrayLike) -> Array:
        return (jnp.pi / 2.0) * jnp.cos(jnp.asarray(t) * jnp.pi / 2.0)

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

    def interpolation_velocity_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array:
        da = self.da_dt(t)
        db = self.db_dt(t)
        return da * x0 + db * x1

    def interpolation_noise_velocity_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None:
        if self.noise_fn is None:
            return None
        if self.noise_velocity_fn is not None:
            return self.noise_velocity_fn(t, x0, x1)
        return _autodiff_time_gradient(self.noise_fn, t, x0, x1)

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
    noise_velocity_fn: Callable[[ArrayLike, Array, Array], Array] | None = None

    def a_t(self, t: ArrayLike) -> Array:
        return (1.0 - jnp.asarray(t)) ** 2

    def b_t(self, t: ArrayLike) -> Array:
        return 1.0 - self.a_t(t)

    def da_dt(self, t: ArrayLike) -> Array:
        return -2.0 * (1.0 - jnp.asarray(t))

    def db_dt(self, t: ArrayLike) -> Array:
        return -self.da_dt(t)

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

    def interpolation_velocity_fn(self, t: ArrayLike, x0: Array, x1: Array) -> Array:
        da = self.da_dt(t)
        db = self.db_dt(t)
        return da * x0 + db * x1

    def interpolation_noise_velocity_fn(
        self, t: ArrayLike, x0: Array, x1: Array
    ) -> Array | None:
        if self.noise_fn is None:
            return None
        if self.noise_velocity_fn is not None:
            return self.noise_velocity_fn(t, x0, x1)
        return _autodiff_time_gradient(self.noise_fn, t, x0, x1)

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
        x_normed = jax.tree_util.tree_map(lambda xi: (xi - mean_t) / std_t, x)
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
        scale = self.velocity_scale(schedule, t, std0, std1)
        delta_mean = jnp.asarray(mu1) - jnp.asarray(mu0)
        return jax.tree_util.tree_map(
            lambda res, xi: delta_mean + scale * (xi - mean_t + res),
            residual_pred,
            x,
        )


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


@runtime_checkable
class FlowSolverConfigProtocol(Protocol):
    """Inference-time schedule builder."""

    num_steps: int

    def solve_schedule(
        self, t_min: float, t_max: float, num_steps: int | None = None
    ) -> Array: ...


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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Tuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss
from probjax.utils.odeint import odeint
from probjax.utils.odeutil.solvers.exponential import SplitDrift
from probjax.utils.sdeint import sdeint
from probjax.utils.typing import Array, ArrayLike, ModuleLike, PyTree, RngKey


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class NoiseScheduleProtocol(Protocol):
    """
    Forward / physics schedule.

    Required:
      - scale(t), std(t)
      - sigma_eff(t): effective noise (for EDM-style preconditioning)
      - inv_sigma_eff(sigma_eff): analytic/numeric inverse
      - marginal_std, drift, diffusion for SDE.
    """

    def scale(self, t: ArrayLike) -> Array: ...
    def std(self, t: ArrayLike) -> Array: ...

    def sigma_eff(self, t: ArrayLike) -> Array: ...
    def inv_sigma_eff(self, sigma_eff: ArrayLike) -> Array: ...

    def marginal_std(self, t: ArrayLike, std0: ArrayLike) -> Array: ...
    def drift(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]: ...
    def diffusion(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]: ...


@runtime_checkable
class PreconditioningProtocol(Protocol):
    """Preconditioning + loss weights in terms of (scale_fn, std_fn, std0)."""

    def c_in(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array: ...

    def c_out(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array: ...

    def c_t(
        self,
        t: ArrayLike,
        *,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array: ...

    def c_skip(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array | None: ...

    def weight_x0(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array: ...

    def weight_eps(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array: ...

    def weight_v(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array: ...


@runtime_checkable
class TrainingConfigProtocol(Protocol):
    """
    Training-only behavior:
      - which loss target
      - kwargs for loss
      - how to sample t
    """

    loss_type: str
    loss_kwargs: Mapping[str, object]
    t_min: float
    t_max: float

    def sample_times(self, rng: RngKey, shape: Tuple[int, ...]) -> Array: ...


@runtime_checkable
class ScheduleAwareModelProtocol(Protocol):
    """
    What solver configs expect from the model.
    DiffusionDenoiser implements this.
    """

    # schedule (physical)
    def scale_fn(self, t: ArrayLike) -> Array: ...
    def std_fn(self, t: ArrayLike) -> Array: ...
    def sigma_eff(self, t: ArrayLike) -> Array: ...
    def inv_sigma_eff(self, sigma_eff: ArrayLike) -> Array: ...

    # SDE terms
    def drift(self, t: ArrayLike, x: PyTree[Array], *args, **kwargs) -> PyTree[Array]: ...
    def diffusion(self, t: ArrayLike, x: PyTree[Array], *args, **kwargs) -> PyTree[Array]: ...

    # score / prediction heads
    def score(self, t: ArrayLike, x: PyTree[Array], *args, **kwargs) -> PyTree[Array]: ...
    def epsilon(self, t: ArrayLike, x: PyTree[Array], *args, **kwargs) -> PyTree[Array]: ...
    def denoise(self, t: ArrayLike, x: PyTree[Array], *args, **kwargs) -> PyTree[Array]: ...
    def v(self, t: ArrayLike, x: PyTree[Array], *args, **kwargs) -> PyTree[Array]: ...


@runtime_checkable
class SolverConfigProtocol(Protocol):
    """
    Inference-time config:
      - build solve schedule from (t_start, t_end)
      - build ODE drift
      - build SDE drift & diffusion
      - provide sampling helpers using odeint / sdeint
    """

    num_steps: int
    ode_method: str
    sde_method: str

    def solve_schedule(
        self,
        t_start: float,
        t_end: float,
        num_steps: int | None = None,
        model: ScheduleAwareModelProtocol | None = None,
    ) -> Array: ...

    def build_ode_drift(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ) -> Callable[[ArrayLike, PyTree[Array]], PyTree[Array]]: ...

    def build_sde_drift_and_diffusion(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ) -> tuple[
        Callable[[ArrayLike, PyTree[Array]], PyTree[Array]],
        Callable[[ArrayLike, PyTree[Array]], PyTree[Array]],
    ]: ...

    def sample_ode(
        self,
        model: ScheduleAwareModelProtocol,
        x_T: PyTree[Array],
        t_start: float,
        t_end: float,
        num_steps: int | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]: ...

    def sample_sde(
        self,
        model: ScheduleAwareModelProtocol,
        rng: RngKey,
        x_T: PyTree[Array],
        t_start: float,
        t_end: float,
        num_steps: int | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]: ...


# =============================================================================
# Shared helpers
# =============================================================================


def alpha_sigma_from_scale_std(
    scale_fn: Callable[[ArrayLike], Array],
    std_fn: Callable[[ArrayLike], Array],
    t: ArrayLike,
) -> tuple[Array, Array]:
    """
    Derive normalized (alpha, sigma) from generic (scale_fn, std_fn):

        a_raw = scale_fn(t)
        s_raw = std_fn(t)
        norm  = sqrt(a_raw^2 + s_raw^2)
        alpha = a_raw / norm
        sigma = s_raw / norm
    """
    a_raw = scale_fn(t)
    s_raw = std_fn(t)
    norm = jnp.sqrt(a_raw**2 + s_raw**2)
    alpha = a_raw / norm
    sigma = s_raw / norm
    return alpha, sigma


# =============================================================================
# Noise schedules
# =============================================================================


@dataclass
class BaseNoiseSchedule(NoiseScheduleProtocol):
    """
    Base schedule:
      - implement scale(t), std(t)
      - default sigma_eff(t) = std / |scale|
      - default inv_sigma_eff via bisection on [t_min, t_max]
      - generic marginal_std, drift, diffusion via autodiff.
    """

    t_min: float = 0.0
    t_max: float = 1.0
    eps: float = 1e-12
    _num_bisect_steps: int = 32

    # ---- abstract core ----

    def scale(self, t: ArrayLike) -> Array:
        raise NotImplementedError

    def std(self, t: ArrayLike) -> Array:
        raise NotImplementedError

    # ---- sigma_eff & inverse ----

    def sigma_eff(self, t: ArrayLike) -> Array:
        alpha = self.scale(t)
        sigma = self.std(t)
        alpha_safe = jnp.maximum(jnp.abs(alpha), self.eps)
        return sigma / alpha_safe

    def inv_sigma_eff(self, sigma_eff: ArrayLike) -> Array:
        """
        Generic numeric inverse of sigma_eff(t).
        Assumes monotone sigma_eff on [t_min, t_max].
        """
        sigma_target = jnp.asarray(sigma_eff)

        t_lo = jnp.full_like(sigma_target, self.t_min)
        t_hi = jnp.full_like(sigma_target, self.t_max)

        def body_fn(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            sigma_mid = self.sigma_eff(mid)
            cond = sigma_mid < sigma_target
            lo = jnp.where(cond, mid, lo)
            hi = jnp.where(cond, hi, mid)
            return (lo, hi)

        lo, hi = jax.lax.fori_loop(
            0, self._num_bisect_steps, body_fn, (t_lo, t_hi)
        )
        return 0.5 * (lo + hi)

    # ---- SDE helpers (default) ----

    def marginal_std(self, t: ArrayLike, std0: ArrayLike) -> Array:
        scale_t = self.scale(t)
        std_t = self.std(t)
        std0_arr = jnp.asarray(std0)
        return jnp.sqrt(scale_t**2 * std0_arr**2 + std_t**2)

    def drift(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        # drift = (d/dt log scale) * x
        def _sum_scale(tt):
            return jnp.sum(self.scale(tt))

        s = self.scale(t)
        s_dt = jax.grad(_sum_scale)(t)
        return jax.tree_util.tree_map(lambda xi: (s_dt / s) * xi, x)

    def diffusion(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        # diffusion = scale * sqrt(2 * std'(t) * std(t))
        def _sum_std(tt):
            return jnp.sum(self.std(tt))

        s = self.scale(t)
        sig = self.std(t)
        sig_dt = jax.grad(_sum_std)(t)
        diff = s * jnp.sqrt(jnp.maximum(2.0 * sig_dt * sig, 0.0))
        return jax.tree_util.tree_map(lambda xi: jnp.broadcast_to(diff, xi.shape), x)


@dataclass
class EDMNoiseSchedule(BaseNoiseSchedule):
    """
    EDM schedule:
      - interpret t as sigma
      - scale(t) = 1
      - std(t)   = t
    """

    def scale(self, t: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def std(self, t: ArrayLike) -> Array:
        return jnp.atleast_1d(t)

    def sigma_eff(self, t: ArrayLike) -> Array:
        # scale=1 => sigma_eff == std == t
        return jnp.atleast_1d(t)

    def inv_sigma_eff(self, sigma_eff: ArrayLike) -> Array:
        return jnp.asarray(sigma_eff)

    def marginal_std(self, t: ArrayLike, std0: ArrayLike) -> Array:
        return jnp.sqrt(self.std(t) ** 2 + jnp.asarray(std0) ** 2)

    def drift(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        return jax.tree_util.tree_map(lambda xi: jnp.zeros_like(xi), x)

    def diffusion(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        sigma = jnp.sqrt(2.0 * jnp.atleast_1d(t))
        return jax.tree_util.tree_map(lambda xi: jnp.broadcast_to(sigma, xi.shape), x)


@dataclass
class VENoiseSchedule(BaseNoiseSchedule):
    """
    VE-style schedule parameterized by (sigma_min, sigma_max) using the
    geometric (Song et al.) interpolation between the endpoints.
    """

    sigma_min: float = 1e-4
    sigma_max: float = 80.0

    def __post_init__(self) -> None:
        if self.sigma_min <= 0 or self.sigma_max <= 0:
            raise ValueError("sigma_min and sigma_max must be positive for VE.")
        if self.sigma_max <= self.sigma_min:
            raise ValueError("sigma_max must be larger than sigma_min for VE.")
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be greater than t_min for VE.")

    def _u(self, t: ArrayLike) -> Array:
        span = self.t_max - self.t_min
        u = (jnp.asarray(t) - self.t_min) / span
        return jnp.clip(u, 0.0, 1.0)

    def scale(self, t: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def std(self, t: ArrayLike) -> Array:
        u = self._u(t)
        log_ratio = jnp.log(self.sigma_max / self.sigma_min)
        std = self.sigma_min * jnp.exp(u * log_ratio)
        return jnp.atleast_1d(std)

    def sigma_eff(self, t: ArrayLike) -> Array:
        # scale=1
        return self.std(t)

    def inv_sigma_eff(self, sigma_eff: ArrayLike) -> Array:
        sigma = jnp.asarray(sigma_eff)
        sigma = jnp.clip(sigma, self.sigma_min, self.sigma_max)

        ratio = self.sigma_max / self.sigma_min
        log_ratio = jnp.log(ratio)
        u = jnp.log(sigma / self.sigma_min) / jnp.maximum(log_ratio, self.eps)
        u = jnp.clip(u, 0.0, 1.0)

        t = self.t_min + u * (self.t_max - self.t_min)
        return t

    def drift(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        # Standard VE SDE: zero drift in many formulations.
        return jax.tree_util.tree_map(lambda xi: jnp.zeros_like(xi), x)


@dataclass
class VPNoiseSchedule(BaseNoiseSchedule):
    """
    VP schedule with linear beta in [beta_min, beta_max] using the Song-style
    warped time (tau) in [min_tau, 1].
    """

    beta_min: float = 0.1
    beta_max: float = 10.0
    min_tau: float = 1e-5

    def __post_init__(self) -> None:
        if self.beta_min <= 0 or self.beta_max <= 0:
            raise ValueError("beta_min/beta_max must be positive for VP.")
        if self.beta_max <= self.beta_min:
            raise ValueError("beta_max must be larger than beta_min for VP.")
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be greater than t_min for VP.")
        if not (0.0 <= self.min_tau < 1.0):
            raise ValueError("min_tau must be in [0,1) for VP.")

    # ---- helpers ----

    def _u(self, t: ArrayLike) -> Array:
        """Raw normalized time in [0,1]."""
        span = self.t_max - self.t_min
        u = (jnp.asarray(t) - self.t_min) / span
        return jnp.clip(u, 0.0, 1.0)

    def _tau(self, t: ArrayLike) -> Array:
        """Beta parameterization time."""
        u = self._u(t)
        # Avoid tau=0 to keep alpha_bar from exactly 1 and sigma from 0.
        return self.min_tau + (1.0 - self.min_tau) * u

    def _beta(self, t: ArrayLike) -> Array:
        """Instantaneous beta(t)."""
        tau = self._tau(t)
        db = self.beta_max - self.beta_min
        return self.beta_min + db * tau

    def _integral_beta(self, t: ArrayLike) -> Array:
        """Integral of beta from 0 to effective time (tau or t)."""
        db = self.beta_max - self.beta_min
        tau = self._tau(t)
        return self.beta_min * tau + 0.5 * db * tau**2

    # ---- schedule ----

    def scale(self, t: ArrayLike) -> Array:
        I = self._integral_beta(t)
        alpha_bar = jnp.exp(-I)
        alpha_bar = jnp.clip(alpha_bar, self.eps, 1.0)
        return jnp.sqrt(alpha_bar)

    def std(self, t: ArrayLike) -> Array:
        I = self._integral_beta(t)
        alpha_bar = jnp.exp(-I)
        alpha_bar = jnp.clip(alpha_bar, self.eps, 1.0)
        return jnp.sqrt(jnp.maximum(1.0 - alpha_bar, 0.0))

    # ---- sigma_eff ----

    def sigma_eff(self, t: ArrayLike) -> Array:
        # sigma_eff^2 = (1 - alpha_bar)/alpha_bar = e^{I} - 1
        I = self._integral_beta(t)
        se2 = jnp.maximum(jnp.exp(I) - 1.0, 0.0)
        return jnp.sqrt(se2)

    def inv_sigma_eff(self, sigma_eff: ArrayLike) -> Array:
        se = jnp.asarray(sigma_eff)
        se2 = jnp.maximum(se**2, 0.0)
        I = jnp.log1p(se2)  # I = log(1 + sigma_eff^2)
        db = self.beta_max - self.beta_min

        # Solve beta_min * z + 0.5 db z^2 = I for z in [0,1], where z is tau or u.
        a = 0.5 * db
        b = self.beta_min
        disc = jnp.maximum(b**2 + 2.0 * db * I, 0.0)
        z = (-b + jnp.sqrt(disc)) / jnp.maximum(db, self.eps)
        z = jnp.clip(z, 0.0, 1.0)

        # z is tau; map back to u in [0,1]
        u = (z - self.min_tau) / jnp.maximum(1.0 - self.min_tau, self.eps)
        u = jnp.clip(u, 0.0, 1.0)

        t = self.t_min + u * (self.t_max - self.t_min)
        return t

    # ---- SDE: override diffusion to be consistent with VP SDE ----

    def drift(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        """
        VP reverse/forward drift structure uses:
          drift_forward = -0.5 * beta(t) * x
        Our BaseSolver / probability flow will combine this with score.
        """
        beta_t = self._beta(t)
        return jax.tree_util.tree_map(lambda xi: -0.5 * beta_t * xi, x)

    def diffusion(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        """
        VP SDE diffusion:
          g(t) = sqrt(beta(t))
        """
        beta_t = self._beta(t)
        g = jnp.sqrt(jnp.maximum(beta_t, self.eps))
        return jax.tree_util.tree_map(lambda xi: jnp.broadcast_to(g, xi.shape), x)


# =============================================================================
# Preconditioning (EDM-style using sigma_eff conceptually)
# =============================================================================


@dataclass
class EDMPreconditioning(PreconditioningProtocol):
    eps: float = 1e-6  # numerical safety

    def _sigma_eff(
        self,
        t: ArrayLike,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array:
        alpha = scale_fn(t)
        sigma = std_fn(t)
        alpha_safe = jnp.maximum(jnp.abs(alpha), self.eps)
        return sigma / alpha_safe

    def c_in(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array:
        sigma_eff = self._sigma_eff(t, scale_fn, std_fn)
        denom = jnp.sqrt(std0**2 + sigma_eff**2)
        return 1.0 / jnp.maximum(denom, self.eps)

    def c_out(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array:
        sigma_eff = self._sigma_eff(t, scale_fn, std_fn)
        denom = jnp.sqrt(std0**2 + sigma_eff**2)
        return (sigma_eff * std0) / jnp.maximum(denom, self.eps)

    def c_t(
        self,
        t: ArrayLike,
        *,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array:
        sigma_eff = self._sigma_eff(t, scale_fn, std_fn)
        sigma_safe = jnp.maximum(sigma_eff, self.eps)
        return 0.25 * jnp.log(sigma_safe)

    def c_skip(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array | None:
        sigma_eff = self._sigma_eff(t, scale_fn, std_fn)
        denom = std0**2 + sigma_eff**2
        return std0**2 / jnp.maximum(denom, self.eps)

    def weight_x0(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array:
        out_w = self.c_out(t, std0=std0, scale_fn=scale_fn, std_fn=std_fn)
        return 1.0 / jnp.maximum(out_w**2, self.eps)

    def weight_eps(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array:
        sigma_eff = self._sigma_eff(t, scale_fn, std_fn)
        return self.weight_x0(
            t, std0=std0, scale_fn=scale_fn, std_fn=std_fn
        ) * sigma_eff**2

    def weight_v(
        self,
        t: ArrayLike,
        *,
        std0: float,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array:
        alpha_hat, sigma_hat = alpha_sigma_from_scale_std(scale_fn, std_fn, t)
        w_eps = self.weight_eps(t, std0=std0, scale_fn=scale_fn, std_fn=std_fn)
        w_x0 = self.weight_x0(t, std0=std0, scale_fn=scale_fn, std_fn=std_fn)
        return 0.5 * (w_eps * alpha_hat**2 + w_x0 * sigma_hat**2)


# =============================================================================
# Training configs
# =============================================================================


@dataclass
class EDMTrainingConfig(TrainingConfigProtocol):
    """
    EDM-style training where schedule parameter t is sigma itself.
    """

    loss_type: str = "x0"
    loss_kwargs: Mapping[str, object] = field(default_factory=dict)

    lognoise_mean: float = -1.2
    lognoise_scale: float = 1.2
    t_min: float = 2e-4
    t_max: float = 80.0

    def sample_times(self, rng: RngKey, shape: Tuple[int, ...]) -> Array:
        logt = (
            jax.random.normal(rng, shape=shape + (1,))
            * self.lognoise_scale
            + self.lognoise_mean
        )
        t = jnp.exp(logt)
        return jnp.clip(t, self.t_min, self.t_max)


@dataclass
class SigmaEffEDMTrainingConfig(TrainingConfigProtocol):
    """
    General EDM-like training in sigma_eff space:

      - sample sigma_eff ~ log-normal
      - clip to [sigma_eff(t_min), sigma_eff(t_max)]
      - map back to t via inv_sigma_eff
    """

    schedule: NoiseScheduleProtocol

    loss_type: str = "x0"
    loss_kwargs: Mapping[str, object] = field(default_factory=dict)

    logsigma_mean: float = -1.2
    logsigma_std: float = 1.2
    t_min: float = 0.0
    t_max: float = 1.0
    eps: float = 1e-12

    def sample_times(self, rng: RngKey, shape: Tuple[int, ...]) -> Array:
        logsigma = (
            jax.random.normal(rng, shape=shape + (1,))
            * self.logsigma_std
            + self.logsigma_mean
        )
        sigma = jnp.exp(logsigma)

        sigma_min = self.schedule.sigma_eff(self.t_min)
        sigma_max = self.schedule.sigma_eff(self.t_max)
        sigma = jnp.clip(sigma, sigma_min + self.eps, sigma_max - self.eps)

        t = self.schedule.inv_sigma_eff(sigma)
        return t


@dataclass
class UniformTTrainingConfig(TrainingConfigProtocol):
    """
    Uniform in t in [t_min, t_max] while using the physical corruption schedule.
    """

    loss_type: str = "x0"
    loss_kwargs: Mapping[str, object] = field(default_factory=dict)

    t_min: float = 1e-3
    t_max: float = 1.0

    def sample_times(self, rng: RngKey, shape: Tuple[int, ...]) -> Array:
        return jax.random.uniform(
            rng,
            shape=shape + (1,),
            minval=self.t_min,
            maxval=self.t_max,
        )


# =============================================================================
# Base solver config (odeint + sdeint)
# =============================================================================


@dataclass
class BaseSolverConfig(SolverConfigProtocol):
    """
    Default solver config:

      - solve_schedule: linear between t_start and t_end
      - build_ode_drift: probability flow ODE
      - build_sde_drift_and_diffusion: reverse SDE
    """

    num_steps: int = 64
    ode_method: str = "euler"
    sde_method: str = "euler_maruyama"

    def solve_schedule(
        self,
        t_start: float,
        t_end: float,
        num_steps: int | None = None,
        model: ScheduleAwareModelProtocol | None = None,
    ) -> Array:
        steps = self.num_steps if num_steps is None else num_steps
        if steps <= 1:
            return jnp.asarray(t_start)[None]

        if model is None:
            return jnp.linspace(t_start, t_end, steps)

        if not hasattr(model, "sigma_eff") or not hasattr(model, "inv_sigma_eff"):
            return jnp.linspace(t_start, t_end, steps)

        sigma_start = jnp.asarray(model.sigma_eff(t_start)).squeeze()
        sigma_end = jnp.asarray(model.sigma_eff(t_end)).squeeze()
        sigma_eps = 1e-6

        if (
            ~jnp.isfinite(sigma_start)
            or ~jnp.isfinite(sigma_end)
            or jnp.isclose(sigma_start, sigma_end, atol=1e-6)
        ):
            return jnp.linspace(t_start, t_end, steps)

        log_start = jnp.log(jnp.maximum(sigma_start, sigma_eps))
        log_end = jnp.log(jnp.maximum(sigma_end, sigma_eps))
        logs = jnp.linspace(log_start, log_end, steps)
        sigma_targets = jnp.exp(logs)
        sigma_targets = sigma_targets.at[0].set(sigma_start)
        sigma_targets = sigma_targets.at[-1].set(sigma_end)

        ts = jnp.asarray(model.inv_sigma_eff(sigma_targets))
        ts = ts.at[0].set(t_start)
        ts = ts.at[-1].set(t_end)
        return ts

    def build_ode_drift(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ) -> Callable[[ArrayLike, PyTree[Array]], PyTree[Array]]:
        # probability flow ODE
        def ode_drift(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            t = jnp.atleast_1d(t)
            f = model.drift(t, x, *args, **kwargs)
            g = model.diffusion(t, x, *args, **kwargs)
            s = model.score(t, x, *args, **kwargs)
            return jax.tree_util.tree_map(
                lambda fi, gi, si: fi - 0.5 * gi**2 * si,
                f,
                g,
                s,
            )

        return ode_drift

    def build_sde_drift_and_diffusion(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ):
        # reverse SDE: drift = f - g^2 * score, diffusion = g
        def drift(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            f = model.drift(t, x, *args, **kwargs)
            g = model.diffusion(t, x, *args, **kwargs)
            s = model.score(t, x, *args, **kwargs)
            return jax.tree_util.tree_map(
                lambda fi, gi, si: fi - gi**2 * si,
                f,
                g,
                s,
            )

        def diffusion(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            return model.diffusion(t, x, *args, **kwargs)

        return drift, diffusion

    def sample_ode(
        self,
        model: ScheduleAwareModelProtocol,
        x_T: PyTree[Array],
        t_start: float,
        t_end: float,
        num_steps: int | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        ts = self.solve_schedule(t_start, t_end, num_steps, model)
        drift = self.build_ode_drift(model, *args, **kwargs)
        return odeint(
            drift,
            x_T,
            ts,
            collect_trace=collect_trace,
            method=self.ode_method,
        )

    def sample_sde(
        self,
        model: ScheduleAwareModelProtocol,
        rng: RngKey,
        x_T: PyTree[Array],
        t_start: float,
        t_end: float,
        num_steps: int | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        ts = self.solve_schedule(t_start, t_end, num_steps, model)
        drift, diffusion = self.build_sde_drift_and_diffusion(model, *args, **kwargs)
        return sdeint(
            rng,
            drift,
            diffusion,
            x_T,
            ts,
            collect_trace=collect_trace,
            method=self.sde_method,
        )


# =============================================================================
# EDM solver (Karras sigma schedule)
# =============================================================================


@dataclass
class EDMSolverConfig(BaseSolverConfig):
    """
    EDM-style non-linear schedule (Karras rho):

      typically called with (t_start=sigma_max, t_end=sigma_min).
    """

    ode_method : str = "heun"
    sde_method : str = "euler_maruyama"
    rho: float = 7.0

    def solve_schedule(
        self,
        t_start: float,
        t_end: float,
        num_steps: int | None = None,
        model: ScheduleAwareModelProtocol | None = None,
    ) -> Array:
        steps = self.num_steps if num_steps is None else num_steps
        ns = jnp.arange(0, steps, dtype=jnp.float32)
        term1 = t_start ** (1.0 / self.rho)
        term2 = t_end ** (1.0 / self.rho)
        length = (term2 - term1) * ns / jnp.maximum(steps - 1, 1)
        return (term1 + length) ** self.rho


# =============================================================================
# DDIM solver (ODE via odeint, DDIM-style drift)
# =============================================================================


@dataclass
class DDIMSolverConfig(BaseSolverConfig):
    """
    DDIM-style deterministic ODE solver:

      dx/dt = alpha'(t) * x0_hat + sigma'(t) * eps_hat
    """

    eta: float = 0.0  # reserved for stochastic variants

    def build_ode_drift(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ) -> Callable[[ArrayLike, PyTree[Array]], PyTree[Array]]:
        def ode_drift(t: ArrayLike, x_t: PyTree[Array]) -> PyTree[Array]:
            alpha_t = model.scale_fn(t)
            sigma_t = model.std_fn(t)
            sign_alpha = jnp.where(alpha_t >= 0, 1.0, -1.0)
            alpha_safe = jnp.where(
                jnp.abs(alpha_t) < 1e-12, sign_alpha * 1e-12, alpha_t
            )
            eps_hat = model.epsilon(t, x_t, *args, **kwargs)

            x0_hat = jax.tree_util.tree_map(
                lambda x_i, e_i: (jnp.nan_to_num(x_i) - sigma_t * e_i) / alpha_safe,
                x_t,
                eps_hat,
            )

            def _sum_alpha(tt):
                return jnp.sum(model.scale_fn(tt))

            def _sum_sigma(tt):
                return jnp.sum(model.std_fn(tt))

            alpha_p = jax.grad(_sum_alpha)(t)
            sigma_p = jax.grad(_sum_sigma)(t)

            return jax.tree_util.tree_map(
                lambda x0_i, e_i: alpha_p * x0_i + sigma_p * e_i,
                x0_hat,
                eps_hat,
            )

        return ode_drift

    def build_sde_drift_and_diffusion(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ):
        ode_drift = self.build_ode_drift(model, *args, **kwargs)

        if isinstance(ode_drift, SplitDrift):
            def _compose_split(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
                Ax = ode_drift.lin_coeff(t)
                nonlin = ode_drift.nonlin(t, x, *args, **kwargs)
                return jax.tree_util.tree_map(
                    lambda xi, ni: Ax * xi + ni,
                    x,
                    nonlin,
                )

            ode_callable = _compose_split
        else:
            ode_callable = ode_drift

        def drift(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            return ode_callable(t, x)

        def diffusion(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            return jax.tree_util.tree_map(lambda xi: jnp.zeros_like(xi), x)

        return drift, diffusion


# =============================================================================
# v-parameterized ODE solver
# =============================================================================


@dataclass
class VParamODESolverConfig(BaseSolverConfig):
    """
    ODE using v-prediction:

      x_t = alpha(t) x0 + sigma(t) eps
      v   = alpha(t) eps - sigma(t) x0
    """

    ode_method: str = "exp_ab2_scalarL"
    _dpm_methods: tuple[str, ...] = ("exp_ab2_scalarL", "exp_ab3_scalarL")


class _SplitDriftAdapter:
    def __init__(self, split: SplitDrift):
        self.__split_drift__ = split

    def __call__(
        self,
        t: ArrayLike,
        x: PyTree[Array],
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        Ax = self.__split_drift__.lin_coeff(t)
        nonlin = self.__split_drift__.nonlin(t, x, *args, **kwargs)
        return jax.tree_util.tree_map(
            lambda xi, ni: Ax * xi + ni,
            x,
            nonlin,
        )

    def _coeff_fn(
        self,
        model: ScheduleAwareModelProtocol,
    ) -> Callable[[ArrayLike], tuple[Array, Array]]:
        def _sum_norm(tt):
            a_tt = model.scale_fn(tt)
            s_tt = model.std_fn(tt)
            norm_tt = jnp.sqrt(jnp.maximum(a_tt**2 + s_tt**2, 1e-12))
            return jnp.sum(norm_tt)

        def _sum_alpha_hat(tt):
            alpha_tt, _ = alpha_sigma_from_scale_std(
                model.scale_fn, model.std_fn, tt
            )
            return jnp.sum(alpha_tt)

        def _sum_sigma_hat(tt):
            _, sigma_tt = alpha_sigma_from_scale_std(
                model.scale_fn, model.std_fn, tt
            )
            return jnp.sum(sigma_tt)

        def coeffs(t: ArrayLike) -> tuple[Array, Array]:
            a = model.scale_fn(t)
            s = model.std_fn(t)
            denom = jnp.maximum(a**2 + s**2, 1e-12)
            norm = jnp.sqrt(denom)
            alpha_hat, sigma_hat = alpha_sigma_from_scale_std(
                model.scale_fn, model.std_fn, t
            )
            norm_p = jax.grad(_sum_norm)(t)
            alpha_hat_p = jax.grad(_sum_alpha_hat)(t)
            sigma_hat_p = jax.grad(_sum_sigma_hat)(t)
            A_x = norm_p / jnp.maximum(norm, 1e-12)
            A_v = norm * (alpha_hat * sigma_hat_p - sigma_hat * alpha_hat_p)
            return A_x, A_v

        return coeffs

    def _build_split_drift(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ) -> SplitDrift:
        coeffs = self._coeff_fn(model)

        def lin_coeff(t: ArrayLike) -> Array:
            Ax, _ = coeffs(t)
            return jnp.asarray(Ax)

        def nonlin(
            t: ArrayLike,
            x: PyTree[Array],
            *fn_args,
            **fn_kwargs,
        ) -> PyTree[Array]:
            _, Av = coeffs(t)
            v_pred = model.v(t, x, *fn_args, **fn_kwargs)
            return jax.tree_util.tree_map(lambda v_i: Av * v_i, v_pred)

        split = SplitDrift(lin_coeff=lin_coeff, nonlin=nonlin)
        return _SplitDriftAdapter(split)

    def build_ode_drift(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ) -> Callable[[ArrayLike, PyTree[Array]], PyTree[Array]]:
        if self.ode_method in self._dpm_methods:
            return self._build_split_drift(model, *args, **kwargs)

        coeffs = self._coeff_fn(model)

        def ode_drift(t: ArrayLike, x_t: PyTree[Array]) -> PyTree[Array]:
            A_x, A_v = coeffs(t)
            v_pred = model.v(t, x_t, *args, **kwargs)
            return jax.tree_util.tree_map(
                lambda x_i, v_i: A_x * x_i + A_v * v_i,
                x_t,
                v_pred,
            )

        return ode_drift

    def build_sde_drift_and_diffusion(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ):
        ode_drift = self.build_ode_drift(model, *args, **kwargs)

        split_drift = getattr(ode_drift, "__split_drift__", None)

        if split_drift is not None:
            def _compose_split(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
                Ax = split_drift.lin_coeff(t)
                nonlin = split_drift.nonlin(t, x, *args, **kwargs)
                return jax.tree_util.tree_map(
                    lambda xi, ni: Ax * xi + ni,
                    x,
                    nonlin,
                )

            ode_callable = _compose_split
        else:
            ode_callable = ode_drift

        def drift(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            return ode_callable(t, x)

        def diffusion(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            return jax.tree_util.tree_map(lambda xi: jnp.zeros_like(xi), x)

        return drift, diffusion


# =============================================================================
# Core DiffusionDenoiser
# =============================================================================


class DiffusionDenoiser(nnx.Module):
    """
    Composable diffusion denoiser:

      - schedule   : NoiseScheduleProtocol   (physical schedule)
      - precond    : PreconditioningProtocol (defines c_in/out/etc)
      - train_cfg  : TrainingConfigProtocol  (defines t sampling)
      - solver_cfg : SolverConfigProtocol    (defines solve ODE/SDE)
    """

    schedule: NoiseScheduleProtocol
    precond: PreconditioningProtocol
    train_cfg: TrainingConfigProtocol
    solver_cfg: SolverConfigProtocol | None

    def __init__(
        self,
        net: ModuleLike,
        schedule: NoiseScheduleProtocol,
        precond: PreconditioningProtocol,
        train_cfg: TrainingConfigProtocol,
        solver_cfg: SolverConfigProtocol | None = None,
        std0: ArrayLike = 1.0,
        last_layer: Callable[[Array], Array] | None = None,
        rngs: nnx.RngStream | None = None,
    ) -> None:
        if not isinstance(schedule, NoiseScheduleProtocol):
            raise TypeError("schedule must implement NoiseScheduleProtocol")
        if not isinstance(precond, PreconditioningProtocol):
            raise TypeError("precond must implement PreconditioningProtocol")
        if not isinstance(train_cfg, TrainingConfigProtocol):
            raise TypeError("train_cfg must implement TrainingConfigProtocol")
        if (
            solver_cfg is not None
            and not isinstance(solver_cfg, SolverConfigProtocol)
        ):
            raise TypeError("solver_cfg must implement SolverConfigProtocol")

        self.rngs = rngs
        self.net: ModuleLike = net
        self.schedule = schedule
        self.precond = precond
        self.train_cfg = train_cfg
        self.solver_cfg = solver_cfg
        self.std0 = nnx.Variable(std0)
        self.last_layer = last_layer

    def set_solver_cfg(self, solver_cfg: SolverConfigProtocol) -> None:
        if not isinstance(solver_cfg, SolverConfigProtocol):
            raise TypeError("solver_cfg must implement SolverConfigProtocol")
        self.solver_cfg = solver_cfg

    # ---- physical schedule adapters ----

    def scale_fn(self, t: ArrayLike) -> Array:
        return self.schedule.scale(t)

    def std_fn(self, t: ArrayLike) -> Array:
        return self.schedule.std(t)

    def sigma_eff(self, t: ArrayLike) -> Array:
        return self.schedule.sigma_eff(t)

    def inv_sigma_eff(self, sigma_eff: ArrayLike) -> Array:
        return self.schedule.inv_sigma_eff(sigma_eff)

    # ---- preconditioning adapters ----

    def c_in(self, t: ArrayLike) -> Array:
        return self.precond.c_in(
            t,
            std0=float(self.std0.value),
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def c_out(self, t: ArrayLike) -> Array:
        return self.precond.c_out(
            t,
            std0=float(self.std0.value),
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def c_t(self, t: ArrayLike) -> Array:
        return self.precond.c_t(
            t,
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def c_skip(self, t: ArrayLike) -> Array | None:
        return self.precond.c_skip(
            t,
            std0=float(self.std0.value),
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def weight_fn(self, t: ArrayLike) -> Array:
        return self.precond.weight_x0(
            t,
            std0=float(self.std0.value),
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def weight_fn_eps(self, t: ArrayLike) -> Array:
        return self.precond.weight_eps(
            t,
            std0=float(self.std0.value),
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def weight_fn_v(self, t: ArrayLike) -> Array:
        return self.precond.weight_v(
            t,
            std0=float(self.std0.value),
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    # ---- loss ----

    def _build_loss_fn(self):
        loss_type = self.train_cfg.loss_type

        if loss_type == "x0":
            weight_fn = self.weight_fn
            pred_fn = self.denoise
        elif loss_type in ("eps", "epsilon"):
            weight_fn = self.weight_fn_eps
            pred_fn = self.epsilon
            loss_type = "eps"
        elif loss_type == "v":
            weight_fn = self.weight_fn_v
            pred_fn = self.v
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return build_time_dependent_denoising_loss(
            pred_fn,
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
            weight_fn=weight_fn,
            prediction_target=loss_type,
            **dict(self.train_cfg.loss_kwargs),
        )

    @property
    def loss_type(self) -> str:
        return self.train_cfg.loss_type

    @loss_type.setter
    def loss_type(self, value: str) -> None:
        self.train_cfg.loss_type = value  # type: ignore[attr-defined]
        self._build_loss_fn()

    # ---- forward ----

    def __call__(
        self,
        t: ArrayLike,
        x_t: PyTree[Array],
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        noise_embed = self.c_t(t)
        x_embed = jax.tree_util.tree_map(lambda x: self.c_in(t) * x, x_t)
        out = self.net(noise_embed, x_embed, *args, **kwargs)
        if self.last_layer is not None:
            out = jax.tree_util.tree_map(self.last_layer, out)
        return out

    # ---- prediction heads ----

    def denoise(
        self,
        t: ArrayLike,
        x_t: PyTree[Array],
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        model_output = self.__call__(t, x_t, *args, **kwargs)
        c_out = self.c_out(t)
        c_skip = self.c_skip(t)
        if c_skip is None:
            return jax.tree_util.tree_map(lambda o: c_out * o, model_output)
        return jax.tree_util.tree_map(
            lambda x, o: c_skip * jnp.nan_to_num(x) + c_out * o,
            x_t,
            model_output,
        )

    def epsilon(
        self,
        t: ArrayLike,
        x_t: PyTree[Array],
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        sigma_t = self.std_fn(t)
        alpha_t = self.scale_fn(t)
        x0_pred = self.denoise(t, x_t, *args, **kwargs)
        return jax.tree_util.tree_map(
            lambda x, o: (jnp.nan_to_num(x) - alpha_t * o) / sigma_t,
            x_t,
            x0_pred,
        )

    def score(
        self,
        t: ArrayLike,
        x_t: PyTree[Array],
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        eps = self.epsilon(t, x_t, *args, **kwargs)
        sigma_t = self.std_fn(t)
        return jax.tree_util.tree_map(
            lambda e: -jnp.nan_to_num(e) / sigma_t,
            eps,
        )

    def v(
        self,
        t: ArrayLike,
        x_t: PyTree[Array],
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        """
        Normalized v:
          v = alpha_hat(t) * eps - sigma_hat(t) * x0
        based on normalized (alpha_hat, sigma_hat).
        """
        x0_pred = self.denoise(t, x_t, *args, **kwargs)
        alpha_t = self.scale_fn(t)
        sigma_t = self.std_fn(t)
        eps_pred = jax.tree_util.tree_map(
            lambda x, x0_: (jnp.nan_to_num(x) - alpha_t * x0_) / sigma_t,
            x_t,
            x0_pred,
        )
        total_var = jnp.sqrt(jnp.maximum(alpha_t**2 + sigma_t**2, 1e-12))
        alpha_hat = alpha_t / total_var
        sigma_hat = sigma_t / total_var
        return jax.tree_util.tree_map(
            lambda e, x0_: alpha_hat * e - sigma_hat * x0_,
            eps_pred,
            x0_pred,
        )

    # ---- SDE helpers ----

    def marginal_std(self, t: ArrayLike) -> Array:
        return self.schedule.marginal_std(t, self.std0.value)

    def drift(
        self,
        t: ArrayLike,
        x: PyTree[Array],
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        return self.schedule.drift(t, x)

    def diffusion(
        self,
        t: ArrayLike,
        x: PyTree[Array],
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        return self.schedule.diffusion(t, x)

    # ---- training loss ----

    def loss(
        self,
        rng: RngKey,
        data: Array,
        *args,
        **kwargs,
    ) -> Array:
        loss_fn = self._build_loss_fn()
        rng_times, rng_loss = jax.random.split(rng, 2)

        ndims = data.ndim - 2
        time_shape = (data.shape[0],) + (1,) * ndims
        times = self.train_cfg.sample_times(rng_times, time_shape)

        if "axis" not in kwargs:
            kwargs["axis"] = tuple(range(1, data.ndim))
        return loss_fn(times, data, *args, rng=rng_loss, **kwargs)

    # ---- sampling (delegates to solver_cfg) ----

    def sample_ode(
        self,
        eps: PyTree[Array],
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        if t_start is None:
            t_start = self.train_cfg.t_max
        if t_end is None:
            t_end = self.train_cfg.t_min
        if self.solver_cfg is None:
            raise ValueError(
                "solver_cfg is not set. Provide one at init or via set_solver_cfg()."
            )
        return self.solver_cfg.sample_ode(
            self,
            eps,
            t_start=t_start,
            t_end=t_end,
            num_steps=num_steps,
            collect_trace=collect_trace,
            *args,
            **kwargs,
        )

    def sample_sde(
        self,
        rng: RngKey,
        eps: PyTree[Array],
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        if t_start is None:
            t_start = self.train_cfg.t_max
        if t_end is None:
            t_end = self.train_cfg.t_min
        if self.solver_cfg is None:
            raise ValueError(
                "solver_cfg is not set. Provide one at init or via set_solver_cfg()."
            )
        return self.solver_cfg.sample_sde(
            self,
            rng,
            eps,
            t_start=t_start,
            t_end=t_end,
            num_steps=num_steps,
            collect_trace=collect_trace,
            *args,
            **kwargs,
        )


# =============================================================================
# Convenience variants: EDM, VE, VP
# =============================================================================


class EDM(DiffusionDenoiser):
    """
    EDM-style model:
      - EDMNoiseSchedule
      - EDMPreconditioning
      - EDMTrainingConfig (t == sigma)
      - EDMSolverConfig by default
    """

    def __init__(
        self,
        net: ModuleLike,
        *,
        std0: float = 1.0,
        lognoise_mean: float = -1.2,
        lognoise_scale: float = 1.2,
        t_min: float = 2e-4,
        t_max: float = 80.0,
        rho: float = 7.0,
        num_steps: int = 64,
        loss_type: str = "x0",
        loss_kwargs: Mapping[str, object] | None = None,
        last_layer: Callable[[Array], Array] | None = None,
        rngs: nnx.RngStream | None = None,
        solver: SolverConfigProtocol | None = None,
    ) -> None:
        schedule = EDMNoiseSchedule(t_min=t_min, t_max=t_max)
        precond = EDMPreconditioning()
        train_cfg = EDMTrainingConfig(
            loss_type=loss_type,
            loss_kwargs=dict(loss_kwargs or {}),
            lognoise_mean=lognoise_mean,
            lognoise_scale=lognoise_scale,
            t_min=t_min,
            t_max=t_max,
        )
        solver_cfg = solver or EDMSolverConfig(num_steps=num_steps, rho=rho)
        super().__init__(
            net=net,
            schedule=schedule,
            precond=precond,
            train_cfg=train_cfg,
            solver_cfg=solver_cfg,
            std0=std0,
            last_layer=last_layer,
            rngs=rngs,
        )


class VE(DiffusionDenoiser):
    """
    VE variant:
      - VENoiseSchedule(sigma_min, sigma_max)
      - EDMPreconditioning
      - UniformTTrainingConfig
      - BaseSolverConfig by default
    """

    def __init__(
        self,
        net: ModuleLike,
        *,
        std0: float = 1.0,
        sigma_min: float = 1e-4,
        sigma_max: float = 80.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
        num_steps: int = 100,
        loss_type: str = "x0",
        loss_kwargs: Mapping[str, object] | None = None,
        last_layer: Callable[[Array], Array] | None = None,
        rngs: nnx.RngStream | None = None,
        solver: SolverConfigProtocol | None = None,
    ) -> None:
        schedule = VENoiseSchedule(
            t_min=t_min,
            t_max=t_max,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        precond = EDMPreconditioning()
        train_cfg = UniformTTrainingConfig(
            loss_type=loss_type,
            loss_kwargs=dict(loss_kwargs or {}),
            t_min=t_min,
            t_max=t_max,
        )
        solver_cfg = solver or VParamODESolverConfig(num_steps=num_steps)
        super().__init__(
            net=net,
            schedule=schedule,
            precond=precond,
            train_cfg=train_cfg,
            solver_cfg=solver_cfg,
            std0=std0,
            last_layer=last_layer,
            rngs=rngs,
        )


class VP(DiffusionDenoiser):
    """
    VP variant:
      - VPNoiseSchedule(beta_min, beta_max)
      - EDMPreconditioning (σ_eff-based)
      - SigmaEffEDMTrainingConfig (log-normal in σ_eff)
      - BaseSolverConfig by default
    """

    def __init__(
        self,
        net: ModuleLike,
        *,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        std0: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
        min_tau: float = 1e-5,
        num_steps: int = 100,
        logsigma_mean: float = -1.2,
        logsigma_std: float = 1.2,
        loss_type: str = "x0",
        loss_kwargs: Mapping[str, object] | None = None,
        last_layer: Callable[[Array], Array] | None = None,
        rngs: nnx.RngStream | None = None,
        solver: SolverConfigProtocol | None = None,
    ) -> None:
        schedule = VPNoiseSchedule(
            t_min=t_min,
            t_max=t_max,
            beta_min=beta_min,
            beta_max=beta_max,
            min_tau=min_tau,
        )
        precond = EDMPreconditioning()
        train_cfg = SigmaEffEDMTrainingConfig(
            schedule=schedule,
            loss_type=loss_type,
            loss_kwargs=dict(loss_kwargs or {}),
            logsigma_mean=logsigma_mean,
            logsigma_std=logsigma_std,
            t_min=t_min,
            t_max=t_max,
        )
        solver_cfg = solver or BaseSolverConfig(num_steps=num_steps)
        super().__init__(
            net=net,
            schedule=schedule,
            precond=precond,
            train_cfg=train_cfg,
            solver_cfg=solver_cfg,
            std0=std0,
            last_layer=last_layer,
            rngs=rngs,
        )

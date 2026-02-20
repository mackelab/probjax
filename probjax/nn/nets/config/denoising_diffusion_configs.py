from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Tuple, runtime_checkable

import jax
import jax.numpy as jnp

from probjax.utils.odeint import odeint
from probjax.utils.functions import split_drift
from probjax.utils.sdeint import sdeint
from probjax.utils.typing import Array, ArrayLike, PyTree, RngKey

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
        std0: ArrayLike,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array: ...

    def c_out(
        self,
        t: ArrayLike,
        *,
        std0: ArrayLike,
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
        std0: ArrayLike,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array | None: ...

    def weight_x0(
        self,
        t: ArrayLike,
        *,
        std0: ArrayLike,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array: ...

    def weight_eps(
        self,
        t: ArrayLike,
        *,
        std0: ArrayLike,
        scale_fn: Callable[[ArrayLike], Array],
        std_fn: Callable[[ArrayLike], Array],
    ) -> Array: ...

    def weight_v(
        self,
        t: ArrayLike,
        *,
        std0: ArrayLike,
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
    def drift(
        self, t: ArrayLike, x: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]: ...
    def diffusion(
        self, t: ArrayLike, x: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]: ...

    # score / prediction heads
    def score(
        self, t: ArrayLike, x: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]: ...
    def epsilon(
        self, t: ArrayLike, x: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]: ...
    def denoise(
        self, t: ArrayLike, x: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]: ...
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
        t_min: float,
        t_max: float,
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
        t_min: float,
        t_max: float,
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
        t_min: float,
        t_max: float,
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

        lo, hi = jax.lax.fori_loop(0, self._num_bisect_steps, body_fn, (t_lo, t_hi))
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


@dataclass
class CosineNoiseSchedule(BaseNoiseSchedule):
    """
    Cosine schedule (Nichol & Dhariwal 2021).

    Introduces a beta schedule where alpha_bar follows a cosine squared law
    controlled by offset parameter s.
    """

    s: float = 0.008

    def __post_init__(self) -> None:
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be greater than t_min for cosine schedule.")
        if self.s < 0.0:
            raise ValueError("offset s must be non-negative for cosine schedule.")

    def _u(self, t: ArrayLike) -> Array:
        span = self.t_max - self.t_min
        u = (jnp.asarray(t) - self.t_min) / span
        return jnp.clip(u, 0.0, 1.0)

    def _theta(self, t: ArrayLike) -> Array:
        u = self._u(t)
        return (self.s + u) / (1.0 + self.s) * (jnp.pi / 2.0)

    def _theta0(self) -> Array:
        return (self.s / (1.0 + self.s)) * (jnp.pi / 2.0)

    def _cos_norm(self) -> Array:
        return jnp.maximum(jnp.cos(self._theta0()), self.eps)

    def _alpha_bar(self, t: ArrayLike) -> Array:
        theta = self._theta(t)
        cos_theta = jnp.cos(theta)
        denom = self._cos_norm()
        alpha_bar = (cos_theta / denom) ** 2
        return jnp.clip(alpha_bar, self.eps, 1.0 - self.eps)

    def scale(self, t: ArrayLike) -> Array:
        return jnp.sqrt(self._alpha_bar(t))

    def std(self, t: ArrayLike) -> Array:
        alpha_bar = self._alpha_bar(t)
        return jnp.sqrt(jnp.maximum(1.0 - alpha_bar, self.eps))

    def sigma_eff(self, t: ArrayLike) -> Array:
        alpha_bar = self._alpha_bar(t)
        return jnp.sqrt(jnp.maximum(1.0 / alpha_bar - 1.0, 0.0))

    def inv_sigma_eff(self, sigma_eff: ArrayLike) -> Array:
        sigma = jnp.asarray(sigma_eff)
        alpha_bar = 1.0 / jnp.maximum(1.0 + sigma**2, self.eps)
        cos_theta = jnp.sqrt(alpha_bar) * self._cos_norm()
        cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
        theta = jnp.arccos(cos_theta)
        frac = (2.0 * theta / jnp.pi) * (1.0 + self.s) - self.s
        u = jnp.clip(frac, 0.0, 1.0)
        return self.t_min + u * (self.t_max - self.t_min)

    def _beta(self, t: ArrayLike) -> Array:
        span = jnp.maximum(self.t_max - self.t_min, self.eps)
        theta = self._theta(t)
        dtheta_dt = (jnp.pi / 2.0) / ((1.0 + self.s) * span)
        beta_t = 2.0 * jnp.tan(theta) * dtheta_dt
        return jnp.maximum(beta_t, self.eps)

    def drift(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        beta_t = self._beta(t)
        return jax.tree_util.tree_map(lambda xi: -0.5 * beta_t * xi, x)

    def diffusion(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
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
        return (
            self.weight_x0(t, std0=std0, scale_fn=scale_fn, std_fn=std_fn)
            * sigma_eff**2
        )

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
            jax.random.normal(rng, shape=shape + (1,)) * self.lognoise_scale
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
            jax.random.normal(rng, shape=shape + (1,)) * self.logsigma_std
            + self.logsigma_mean
        )
        sigma = jnp.exp(logsigma)

        sigma_min = self.schedule.sigma_eff(self.t_min)
        sigma_max = self.schedule.sigma_eff(self.t_max)
        sigma = jnp.clip(sigma, sigma_min + self.eps, sigma_max - self.eps)

        t = self.schedule.inv_sigma_eff(sigma)
        return t


@dataclass
class LogSNRTrainingConfig(TrainingConfigProtocol):
    """
    Training config that samples times uniformly in log-SNR space.

    Equivalent to distributing log(α/σ) uniformly which mirrors the log-SNR
    spacing many solver schedules (e.g. VPRED/v-ODE) use at inference time.
    """

    schedule: NoiseScheduleProtocol

    loss_type: str = "x0"
    loss_kwargs: Mapping[str, object] = field(default_factory=dict)

    t_min: float = 0.0
    t_max: float = 1.0
    eps: float = 1e-12

    def sample_times(self, rng: RngKey, shape: Tuple[int, ...]) -> Array:
        sigma_start = jnp.asarray(self.schedule.sigma_eff(self.t_min))
        sigma_end = jnp.asarray(self.schedule.sigma_eff(self.t_max))
        sigma_min = jnp.minimum(sigma_start, sigma_end)
        sigma_max = jnp.maximum(sigma_start, sigma_end)

        snr_max = 1.0 / jnp.maximum(sigma_min, self.eps)
        snr_min = 1.0 / jnp.maximum(sigma_max, self.eps)

        log_snr_min = jnp.log(jnp.maximum(snr_min, self.eps))
        log_snr_max = jnp.log(jnp.maximum(snr_max, self.eps))

        log_snr = jax.random.uniform(
            rng,
            shape=shape + (1,),
            minval=jnp.minimum(log_snr_min, log_snr_max),
            maxval=jnp.maximum(log_snr_min, log_snr_max),
        )
        snr = jnp.exp(log_snr)
        sigma = 1.0 / jnp.maximum(snr, self.eps)
        sigma = jnp.clip(sigma, sigma_min + self.eps, sigma_max - self.eps)
        return self.schedule.inv_sigma_eff(sigma)


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

    schedule: NoiseScheduleProtocol
    num_steps: int = 64
    ode_method: str = "euler"
    sde_method: str = "euler_maruyama"

    def set_schedule(self, schedule: NoiseScheduleProtocol) -> None:
        self.schedule = schedule

    def solve_schedule(
        self,
        t_min: float,
        t_max: float,
        num_steps: int | None = None,
    ) -> Array:
        steps = self.num_steps if num_steps is None else num_steps
        if steps <= 1:
            return jnp.asarray(t_max)[None]

        schedule = self.schedule
        if schedule is None:
            raise ValueError(
                "BaseSolverConfig requires a schedule. Call set_schedule first."
            )

        sigma_start = jnp.asarray(schedule.sigma_eff(t_max)).squeeze()
        sigma_end = jnp.asarray(schedule.sigma_eff(t_min)).squeeze()
        sigma_eps = 1e-6

        log_start = jnp.log(jnp.maximum(sigma_start, sigma_eps))
        log_end = jnp.log(jnp.maximum(sigma_end, sigma_eps))
        logs = jnp.linspace(log_start, log_end, steps)
        sigma_targets = jnp.exp(logs)
        sigma_targets = sigma_targets.at[0].set(sigma_start)
        sigma_targets = sigma_targets.at[-1].set(sigma_end)

        ts = jnp.asarray(schedule.inv_sigma_eff(sigma_targets))
        ts = ts.at[0].set(t_max)
        ts = ts.at[-1].set(t_min)
        return ts

    def build_ode_drift(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ) -> split_drift:
        # probability flow ODE

        def _sum_scale(tt):
            return jnp.sum(model.scale_fn(tt))

        def linear_coeff(t: ArrayLike) -> Array:
            t_arr = jnp.atleast_1d(t)
            scale = jnp.asarray(model.scale_fn(t_arr))
            scale = jnp.where(jnp.abs(scale) < 1e-12, 1e-12, scale)
            scale_grad = jax.grad(_sum_scale)(t_arr)
            return scale_grad / scale

        def nonlin(t: ArrayLike, x: PyTree[Array]):
            t = jnp.atleast_1d(t)
            g = model.diffusion(t, x)
            s = model.score(t, x, *args, **kwargs)
            return jax.tree_util.tree_map(
                lambda gi, si: -0.5 * gi**2 * si,
                g,
                s,
            )

        return split_drift(lin_coeff=linear_coeff, nonlin=nonlin)

    def build_sde_drift_and_diffusion(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ):
        # reverse SDE: drift = f - g^2 * score, diffusion = g
        def drift(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            f = model.drift(t, x)
            g = model.diffusion(t, x)
            s = model.score(t, x, *args, **kwargs)
            return jax.tree_util.tree_map(
                lambda fi, gi, si: fi - gi**2 * si,
                f,
                g,
                s,
            )

        def diffusion(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            return model.diffusion(t, x)

        return drift, diffusion

    def sample_ode(
        self,
        model: ScheduleAwareModelProtocol,
        x_T: PyTree[Array],
        t_min: float,
        t_max: float,
        num_steps: int | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        ts = self.solve_schedule(t_max=t_max, t_min=t_min, num_steps=num_steps)
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
        t_min: float,
        t_max: float,
        num_steps: int | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        ts = self.solve_schedule(t_max=t_max, t_min=t_min, num_steps=num_steps)
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

    ode_method: str = "heun"
    sde_method: str = "euler_maruyama"
    rho: float = 7.0

    def solve_schedule(
        self,
        t_min: float,
        t_max: float,
        num_steps: int | None = None,
    ) -> Array:
        steps = self.num_steps if num_steps is None else num_steps
        if steps <= 1:
            return jnp.asarray(t_max)[None]

        schedule = self.schedule
        sigma_eps = 1e-6
        sigma_start = jnp.asarray(schedule.sigma_eff(t_max)).squeeze()
        sigma_end = jnp.asarray(schedule.sigma_eff(t_min)).squeeze()

        sigma_start_root = jnp.maximum(sigma_start, sigma_eps) ** (1.0 / self.rho)
        sigma_end_root = jnp.maximum(sigma_end, sigma_eps) ** (1.0 / self.rho)
        ns = jnp.arange(0, steps, dtype=jnp.float32)
        length = (sigma_end_root - sigma_start_root) * ns / jnp.maximum(steps - 1, 1)
        sigma_targets = (sigma_start_root + length) ** self.rho
        sigma_targets = sigma_targets.at[0].set(sigma_start)
        sigma_targets = sigma_targets.at[-1].set(sigma_end)
        ts = jnp.asarray(schedule.inv_sigma_eff(sigma_targets))
        ts = ts.at[0].set(t_max)
        ts = ts.at[-1].set(t_min)
        return ts


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
    ) -> split_drift:
        def nonlin(t: ArrayLike, x_t: PyTree[Array]):
            t = jnp.atleast_1d(t)
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

        def lin_coeff(t: ArrayLike) -> Array:
            return jnp.zeros_like(jnp.asarray(t))

        return split_drift(lin_coeff=lin_coeff, nonlin=nonlin)

    def build_sde_drift_and_diffusion(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ):
        ode_drift = self.build_ode_drift(model, *args, **kwargs)
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
class VSolverConfig(BaseSolverConfig):
    """
    ODE using v-prediction:

      x_t = alpha(t) x0 + sigma(t) eps
      v   = alpha(t) eps - sigma(t) x0
    """

    ode_method: str = "exp_ab2_scalarL"

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
            alpha_tt, _ = alpha_sigma_from_scale_std(model.scale_fn, model.std_fn, tt)
            return jnp.sum(alpha_tt)

        def _sum_sigma_hat(tt):
            _, sigma_tt = alpha_sigma_from_scale_std(model.scale_fn, model.std_fn, tt)
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

    def build_ode_drift(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ) -> split_drift:
        coeffs = self._coeff_fn(model)

        def lin_coeff(t: ArrayLike) -> Array:
            Ax, _ = coeffs(t)
            return jnp.asarray(Ax)

        def nonlin(
            t: ArrayLike,
            x: PyTree[Array],
        ) -> PyTree[Array]:
            t = jnp.atleast_1d(t)
            _, Av = coeffs(t)
            v_pred = model.v(t, x, *args, **kwargs)
            return jax.tree_util.tree_map(lambda v_i: Av * v_i, v_pred)

        return split_drift(lin_coeff=lin_coeff, nonlin=nonlin)

    def build_sde_drift_and_diffusion(
        self,
        model: ScheduleAwareModelProtocol,
        *args,
        **kwargs,
    ):
        split = self.build_ode_drift(model, *args, **kwargs)

        def drift(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            return split(t, x, *args, **kwargs)

        def diffusion(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            return jax.tree_util.tree_map(lambda xi: jnp.zeros_like(xi), x)

        return drift, diffusion

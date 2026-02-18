from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from probjax.nn.loss_fn.multinomial_diffusion import (
    build_time_dependent_multinomial_diffusion_loss,
)
from probjax.utils.typing import Array, ArrayLike, ModuleLike, RngKey


def _require_float_time(t: ArrayLike, *, name: str = "t") -> Array:
    t_arr = jnp.asarray(t)
    if not jnp.issubdtype(t_arr.dtype, jnp.floating):
        raise TypeError(
            f"{name} must be floating-point continuous time, got dtype={t_arr.dtype}."
        )
    return t_arr.astype(jnp.float32)


def _default_base_probs(num_classes: int) -> Array:
    return jnp.ones((num_classes,), dtype=jnp.float32) / float(num_classes)


def _ensure_simplex(base_probs: ArrayLike, eps: float = 1e-12) -> Array:
    probs = jnp.asarray(base_probs, dtype=jnp.float32)
    if probs.ndim != 1:
        raise ValueError("base_probs must be a 1D probability vector.")
    probs = jnp.clip(probs, eps, None)
    probs = probs / jnp.sum(probs)
    return probs


def _one_hot(x: Array, num_classes: int) -> Array:
    return jax.nn.one_hot(x, num_classes, dtype=jnp.float32)


def _sample_categorical(rng: RngKey, probs: Array) -> Array:
    logits = jnp.log(jnp.maximum(probs, 1e-12))
    return jax.random.categorical(rng, logits, axis=-1)


def _sample_unit_stratified(rng: RngKey, n: int) -> Array:
    if n <= 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    rng_u, rng_perm = jax.random.split(rng, 2)
    u = (jnp.arange(n, dtype=jnp.float32) + jax.random.uniform(rng_u, (n,))) / float(n)
    return jax.random.permutation(rng_perm, u)


def _sample_unit_interval(
    rng: RngKey,
    shape: tuple[int, ...],
    *,
    stratified: bool = False,
    antithetic: bool = False,
) -> Array:
    n = int(np.prod(shape))
    if n <= 0:
        return jnp.zeros(shape, dtype=jnp.float32)

    if antithetic:
        n_half = (n + 1) // 2
        rng_base, rng_perm = jax.random.split(rng, 2)
        if stratified:
            u_base = _sample_unit_stratified(rng_base, n_half)
        else:
            u_base = jax.random.uniform(rng_base, (n_half,), dtype=jnp.float32)
        u = jnp.concatenate([u_base, 1.0 - u_base], axis=0)[:n]
        u = jax.random.permutation(rng_perm, u)
    else:
        if stratified:
            u = _sample_unit_stratified(rng, n)
        else:
            u = jax.random.uniform(rng, (n,), dtype=jnp.float32)

    return jnp.clip(u, 0.0, 1.0 - 1e-7).reshape(shape)


@runtime_checkable
class CategoricalScheduleProtocol(Protocol):
    num_steps: int
    num_classes: int
    base_probs: Array
    t_min: float
    t_max: float

    def sample_timesteps(self, rng: RngKey, shape: tuple[int, ...]) -> Array: ...
    def beta(self, t: ArrayLike) -> Array: ...
    def alpha_bar(self, t: ArrayLike) -> Array: ...
    def q_xt_given_x0_probs(self, x0: Array, t: ArrayLike) -> Array: ...
    def q_sample(self, rng: RngKey, x0: Array, t: ArrayLike) -> Array: ...
    def posterior_mixture_probs(
        self,
        x_t: Array,
        x0_probs: Array,
        t: ArrayLike,
        t_prev: ArrayLike,
    ) -> Array: ...
    def prior_sample(self, rng: RngKey, shape: tuple[int, ...]) -> Array: ...


@runtime_checkable
class CategoricalPreconditioningProtocol(Protocol):
    def c_in(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array: ...
    def c_out(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array: ...
    def c_skip(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array: ...
    def c_t(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array: ...
    def weight_ce(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array: ...


@runtime_checkable
class CategoricalTrainingConfigProtocol(Protocol):
    t_min: float
    t_max: float
    loss_kwargs: Mapping[str, object]

    def sample_times(self, rng: RngKey, shape: tuple[int, ...]) -> Array: ...


@dataclass
class MultinomialDiffusionSchedule(CategoricalScheduleProtocol):
    """
    Forward categorical diffusion with:
      Q_t = (1 - beta_t) I + beta_t 1 pi^T
    """

    num_steps: int
    num_classes: int
    base_probs: ArrayLike
    kind: str
    params: Mapping[str, float] = field(default_factory=dict)
    t_min: float = 0.0
    t_max: float = 1.0
    eps: float = 1e-12

    def __post_init__(self):
        if int(self.num_steps) <= 0:
            raise ValueError("num_steps must be a positive integer.")
        if int(self.num_classes) <= 1:
            raise ValueError("num_classes must be >= 2.")
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be greater than t_min.")
        if self.kind not in {"linear", "cosine", "sigmoid", "logsnr"}:
            raise ValueError(
                "kind must be one of {'linear', 'cosine', 'sigmoid', 'logsnr'}."
            )

        self.num_steps = int(self.num_steps)
        self.num_classes = int(self.num_classes)
        self.base_probs = _ensure_simplex(self.base_probs, eps=self.eps)
        self.params = {k: float(v) for k, v in dict(self.params).items()}

        u_grid = jnp.linspace(0.0, 1.0, 257, dtype=jnp.float32)
        alpha_grid = np.asarray(self._alpha_bar_unit(u_grid))
        if np.any(~np.isfinite(alpha_grid)):
            raise ValueError("alpha_bar function produced non-finite values.")
        if np.any(alpha_grid <= 0.0) or np.any(alpha_grid > 1.0 + 1e-6):
            raise ValueError("alpha_bar(t) must stay in (0, 1].")
        if np.any(np.diff(alpha_grid) > 1e-4):
            raise ValueError("alpha_bar(t) must be non-increasing.")

        beta_grid = np.asarray(self._beta_unit(u_grid[:-1]))
        if np.any(~np.isfinite(beta_grid)):
            raise ValueError("beta(t) function produced non-finite values.")
        if np.any(beta_grid < 0.0):
            raise ValueError("beta(t) must be non-negative.")

    def _param(self, name: str, default: Optional[float] = None) -> float:
        if name in self.params:
            return float(self.params[name])
        if default is not None:
            return float(default)
        raise ValueError(f"Missing required schedule parameter '{name}' for kind='{self.kind}'.")

    def _alpha_bar_unit(self, u: Array) -> Array:
        u = jnp.clip(jnp.asarray(u, dtype=jnp.float32), 0.0, 1.0)

        if self.kind == "linear":
            beta_start = self._param("beta_start")
            beta_end = self._param("beta_end")
            integral = beta_start * u + 0.5 * (beta_end - beta_start) * (u**2)
            alpha = jnp.exp(-integral)
        elif self.kind == "cosine":
            s = self._param("s", 0.008)
            f = ((u + s) / (1.0 + s)) * (jnp.pi * 0.5)
            f0 = (s / (1.0 + s)) * (jnp.pi * 0.5)
            alpha = (jnp.cos(f) / jnp.maximum(jnp.cos(f0), self.eps)) ** 2
        elif self.kind == "sigmoid":
            beta_start = self._param("beta_start")
            beta_end = self._param("beta_end")
            start = self._param("start")
            end = self._param("end")
            span = end - start
            if abs(span) < self.eps:
                sig = jax.nn.sigmoid(start)
                integral = (beta_start + (beta_end - beta_start) * sig) * u
            else:
                z = start + span * u
                integral_sig = (jax.nn.softplus(z) - jax.nn.softplus(start)) / span
                integral = beta_start * u + (beta_end - beta_start) * integral_sig
            alpha = jnp.exp(-integral)
        elif self.kind == "logsnr":
            logsnr_max = self._param("logsnr_max")
            logsnr_min = self._param("logsnr_min")
            logsnr = logsnr_max + (logsnr_min - logsnr_max) * u
            alpha = jax.nn.sigmoid(logsnr)
        else:
            raise ValueError(f"Unsupported schedule kind: {self.kind}")

        return jnp.clip(alpha, self.eps, 1.0)

    def _beta_unit(self, u: Array) -> Array:
        u = jnp.clip(jnp.asarray(u, dtype=jnp.float32), 0.0, 1.0)

        if self.kind == "linear":
            beta_start = self._param("beta_start")
            beta_end = self._param("beta_end")
            beta = beta_start + (beta_end - beta_start) * u
        elif self.kind == "cosine":
            s = self._param("s", 0.008)
            max_beta = self._param("max_beta", 50.0)
            f = ((u + s) / (1.0 + s)) * (jnp.pi * 0.5)
            beta = (jnp.pi / (1.0 + s)) * jnp.tan(f)
            beta = jnp.clip(beta, 0.0, max_beta)
        elif self.kind == "sigmoid":
            beta_start = self._param("beta_start")
            beta_end = self._param("beta_end")
            start = self._param("start")
            end = self._param("end")
            beta = beta_start + (beta_end - beta_start) * jax.nn.sigmoid(
                start + (end - start) * u
            )
        elif self.kind == "logsnr":
            logsnr_max = self._param("logsnr_max")
            logsnr_min = self._param("logsnr_min")
            max_beta = self._param("max_beta", 50.0)
            alpha = self._alpha_bar_unit(u)
            beta = (logsnr_max - logsnr_min) * (1.0 - alpha)
            beta = jnp.clip(beta, 0.0, max_beta)
        else:
            raise ValueError(f"Unsupported schedule kind: {self.kind}")

        return jnp.maximum(beta, 0.0)

    @classmethod
    def from_linear(
        cls,
        num_steps: int,
        num_classes: int,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        base_probs: Optional[ArrayLike] = None,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> "MultinomialDiffusionSchedule":
        pi = _default_base_probs(num_classes) if base_probs is None else base_probs
        return cls(
            num_steps=num_steps,
            num_classes=num_classes,
            base_probs=pi,
            kind="linear",
            params={"beta_start": beta_start, "beta_end": beta_end},
            t_min=t_min,
            t_max=t_max,
        )

    @classmethod
    def from_cosine(
        cls,
        num_steps: int,
        num_classes: int,
        *,
        s: float = 0.008,
        max_beta: float = 0.999,
        base_probs: Optional[ArrayLike] = None,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> "MultinomialDiffusionSchedule":
        pi = _default_base_probs(num_classes) if base_probs is None else base_probs
        return cls(
            num_steps=num_steps,
            num_classes=num_classes,
            base_probs=pi,
            kind="cosine",
            params={"s": s, "max_beta": max_beta},
            t_min=t_min,
            t_max=t_max,
        )

    @classmethod
    def from_sigmoid(
        cls,
        num_steps: int,
        num_classes: int,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        start: float = -3.0,
        end: float = 3.0,
        base_probs: Optional[ArrayLike] = None,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> "MultinomialDiffusionSchedule":
        pi = _default_base_probs(num_classes) if base_probs is None else base_probs
        return cls(
            num_steps=num_steps,
            num_classes=num_classes,
            base_probs=pi,
            kind="sigmoid",
            params={
                "beta_start": beta_start,
                "beta_end": beta_end,
                "start": start,
                "end": end,
            },
            t_min=t_min,
            t_max=t_max,
        )

    @classmethod
    def from_logsnr(
        cls,
        num_steps: int,
        num_classes: int,
        *,
        logsnr_max: float = 9.0,
        logsnr_min: float = -6.0,
        max_beta: float = 20.0,
        base_probs: Optional[ArrayLike] = None,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> "MultinomialDiffusionSchedule":
        pi = _default_base_probs(num_classes) if base_probs is None else base_probs
        return cls(
            num_steps=num_steps,
            num_classes=num_classes,
            base_probs=pi,
            kind="logsnr",
            params={
                "logsnr_max": logsnr_max,
                "logsnr_min": logsnr_min,
                "max_beta": max_beta,
            },
            t_min=t_min,
            t_max=t_max,
        )

    def sample_timesteps(self, rng: RngKey, shape: tuple[int, ...]) -> Array:
        # Continuous t in [t_min, t_max].
        return jax.random.uniform(
            rng,
            shape=shape,
            minval=self.t_min,
            maxval=self.t_max,
            dtype=jnp.float32,
        )

    def _to_unit_time(self, t: ArrayLike) -> Array:
        t_cont = self._to_continuous_time(t)
        span = jnp.maximum(self.t_max - self.t_min, self.eps)
        u = (t_cont - self.t_min) / span
        return jnp.clip(u, 0.0, 1.0)

    def _to_continuous_time(self, t: ArrayLike) -> Array:
        t_arr = _require_float_time(t)
        t_cont = t_arr.astype(jnp.float32)
        return jnp.clip(t_cont, self.t_min, self.t_max)

    def _broadcast_time_like(self, t: ArrayLike, ref: Array) -> Array:
        t_cont = self._to_continuous_time(t)
        if t_cont.shape != ref.shape:
            t_cont = jnp.broadcast_to(t_cont, ref.shape)
        return t_cont

    def beta(self, t: ArrayLike) -> Array:
        u = self._to_unit_time(t)
        span = jnp.maximum(self.t_max - self.t_min, self.eps)
        return self._beta_unit(u) / span

    def alpha_bar(self, t: ArrayLike) -> Array:
        u = self._to_unit_time(t)
        return self._alpha_bar_unit(u)

    def beta_between(self, t_prev: ArrayLike, t: ArrayLike) -> Array:
        """
        Effective beta for the interval [t_prev, t] under alpha_bar parameterization.
        """
        t_prev_cont = self._to_continuous_time(t_prev)
        t_cont = self._to_continuous_time(t)
        alpha_prev = self.alpha_bar(t_prev_cont)
        alpha_t = self.alpha_bar(t_cont)
        alpha_ratio = alpha_t / jnp.maximum(alpha_prev, self.eps)
        return jnp.clip(1.0 - alpha_ratio, 0.0, 1.0 - self.eps)

    def q_xt_given_x0_probs(self, x0: Array, t: ArrayLike) -> Array:
        x0_oh = _one_hot(jnp.asarray(x0, dtype=jnp.int32), self.num_classes)
        alpha_bar_t = jnp.asarray(self.alpha_bar(t), dtype=jnp.float32)[..., None]
        pi = self.base_probs.reshape((1,) * (x0_oh.ndim - 1) + (-1,))
        return alpha_bar_t * x0_oh + (1.0 - alpha_bar_t) * pi

    def q_sample(self, rng: RngKey, x0: Array, t: ArrayLike) -> Array:
        probs = self.q_xt_given_x0_probs(x0, t)
        return _sample_categorical(rng, probs)

    def posterior_mixture_probs(
        self,
        x_t: Array,
        x0_probs: Array,
        t: ArrayLike,
        t_prev: ArrayLike,
    ) -> Array:
        """
        p(x_{t_prev}|x_t) by mixing exact q(x_{t_prev}|x_t,x0) with p_theta(x0|x_t).

        Uses the closed-form O(K) evaluation for kernels of the form:
          Q = (1 - beta_eff) I + beta_eff 1 pi^T
        """
        x_t_arr = jnp.asarray(x_t, dtype=jnp.int32)
        x0_arr = jnp.asarray(x0_probs, dtype=jnp.float32)
        if x0_arr.shape != x_t_arr.shape + (self.num_classes,):
            x0_arr = jnp.broadcast_to(x0_arr, x_t_arr.shape + (self.num_classes,))

        t_cont = self._broadcast_time_like(t, x_t_arr)
        t_prev_cont = self._broadcast_time_like(t_prev, x_t_arr)

        x_t_flat = x_t_arr.reshape(-1)
        x0_flat = x0_arr.reshape(-1, self.num_classes)
        t_flat = t_cont.reshape(-1)
        t_prev_flat = t_prev_cont.reshape(-1)

        alpha_t = self.alpha_bar(t_flat)[:, None]
        alpha_prev = self.alpha_bar(t_prev_flat)[:, None]
        beta_eff = self.beta_between(t_prev_flat, t_flat)[:, None]

        pi = self.base_probs
        pi_j = pi[x_t_flat][:, None]
        onehot_j = jax.nn.one_hot(x_t_flat, self.num_classes, dtype=jnp.float32)

        # z_i = q(x_t=j | x0=i) = alpha_t * 1[i=j] + (1-alpha_t) * pi_j
        z = (1.0 - alpha_t) * pi_j + alpha_t * onehot_j
        c = x0_flat / jnp.maximum(z, self.eps)
        c_sum = jnp.sum(c, axis=-1, keepdims=True)

        # S_k = sum_i c_i * q(x_{t_prev}=k | x0=i)
        s = alpha_prev * c + (1.0 - alpha_prev) * (pi[None, :] * c_sum)

        # q(x_t=j | x_{t_prev}=k)
        b = beta_eff * pi_j + (1.0 - beta_eff) * onehot_j

        out = b * s
        out = out.reshape(x_t_arr.shape + (self.num_classes,))
        out = out / jnp.maximum(jnp.sum(out, axis=-1, keepdims=True), self.eps)
        return out

    def prior_sample(self, rng: RngKey, shape: tuple[int, ...]) -> Array:
        probs = jnp.broadcast_to(self.base_probs, shape + (self.num_classes,))
        return _sample_categorical(rng, probs)


@dataclass
class CategoricalEDMPreconditioning(CategoricalPreconditioningProtocol):
    """EDM-like preconditioning over alpha_bar / sigma_eff for categorical data."""

    eps: float = 1e-6
    min_weight: float = 0.05
    max_weight: float = 50.0

    def _sigma_eff(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array:
        alpha_bar_t = jnp.asarray(schedule.alpha_bar(t), dtype=jnp.float32)
        return jnp.sqrt((1.0 - alpha_bar_t) / jnp.maximum(alpha_bar_t, self.eps))

    def c_in(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array:
        sigma = self._sigma_eff(t, schedule)
        return 1.0 / jnp.sqrt(1.0 + sigma**2)

    def c_out(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array:
        sigma = self._sigma_eff(t, schedule)
        return sigma / jnp.sqrt(1.0 + sigma**2)

    def c_skip(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array:
        sigma = self._sigma_eff(t, schedule)
        return 1.0 / (1.0 + sigma**2)

    def c_t(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array:
        sigma = self._sigma_eff(t, schedule)
        return 0.25 * jnp.log(jnp.maximum(sigma, self.eps))

    def weight_ce(self, t: ArrayLike, schedule: CategoricalScheduleProtocol) -> Array:
        c_out = self.c_out(t, schedule)
        w = 1.0 / jnp.maximum(c_out**2, self.eps)
        return jnp.clip(w, self.min_weight, self.max_weight)


@dataclass
class UniformContinuousTimeTrainingConfig(CategoricalTrainingConfigProtocol):
    num_steps: int
    t_min: float = 0.0
    t_max: float = 1.0
    stratified: bool = False
    antithetic: bool = False
    loss_kwargs: Mapping[str, object] = field(default_factory=dict)

    def sample_times(self, rng: RngKey, shape: tuple[int, ...]) -> Array:
        if self.t_max <= self.t_min:
            raise ValueError("t_max must be greater than t_min.")
        u = _sample_unit_interval(
            rng,
            shape,
            stratified=self.stratified,
            antithetic=self.antithetic,
        )
        return self.t_min + u * (self.t_max - self.t_min)


@dataclass
class ImportanceContinuousTimeTrainingConfig(CategoricalTrainingConfigProtocol):
    schedule: CategoricalScheduleProtocol
    power: float = 1.0
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    stratified: bool = False
    antithetic: bool = False
    loss_kwargs: Mapping[str, object] = field(default_factory=dict)

    def _time_bounds(self) -> tuple[float, float]:
        t_min = self.schedule.t_min if self.t_min is None else self.t_min
        t_max = self.schedule.t_max if self.t_max is None else self.t_max
        if t_max <= t_min:
            raise ValueError("t_max must be greater than t_min.")
        return t_min, t_max

    def sample_times(self, rng: RngKey, shape: tuple[int, ...]) -> Array:
        # Sample continuous times with interval mass proportional to beta(t)^power.
        n_bins = max(int(self.schedule.num_steps), 2)
        u = _sample_unit_interval(
            rng,
            shape,
            stratified=self.stratified,
            antithetic=self.antithetic,
        ).reshape(-1)
        t_min, t_max = self._time_bounds()
        t_edges = jnp.linspace(t_min, t_max, n_bins + 1, dtype=jnp.float32)
        t_mid = 0.5 * (t_edges[:-1] + t_edges[1:])

        beta_mid = jnp.maximum(self.schedule.beta(t_mid), 1e-8) ** self.power
        probs = beta_mid / jnp.maximum(jnp.sum(beta_mid), 1e-12)
        cdf = jnp.cumsum(probs)

        bin_idx = jnp.searchsorted(cdf, u, side="right")
        bin_idx = jnp.clip(bin_idx, 0, n_bins - 1)

        cdf_prev = jnp.where(bin_idx == 0, 0.0, cdf[bin_idx - 1])
        cdf_curr = cdf[bin_idx]
        u_local = (u - cdf_prev) / jnp.maximum(cdf_curr - cdf_prev, 1e-12)
        u_local = jnp.clip(u_local, 0.0, 1.0 - 1e-7)

        t_left = t_edges[bin_idx]
        t_right = t_edges[bin_idx + 1]
        t = t_left + u_local * (t_right - t_left)
        return t.reshape(shape)


class MultinomialDiffusion(nnx.Module):
    """
    Discrete diffusion model with denoising_diffusion_model-like composition:
      - schedule
      - preconditioning
      - training config
    """

    def __init__(
        self,
        net: ModuleLike,
        schedule: CategoricalScheduleProtocol,
        *,
        preconditioning: Optional[CategoricalPreconditioningProtocol] = None,
        train_cfg: Optional[CategoricalTrainingConfigProtocol] = None,
        use_loss_weighting: bool = False,
        rao_blackwellize_xt: bool = False,
        rao_blackwellize_xt_num_samples: int = 4,
        rao_blackwellize_xt_num_features: Optional[int] = None,
        rngs: nnx.RngStream | None = None,
        sharding: jax.sharding.Mesh | None = None,
        eps: float = 1e-12,
    ):
        if not isinstance(schedule, CategoricalScheduleProtocol):
            raise TypeError("schedule must implement CategoricalScheduleProtocol")

        self.net = net
        self.schedule = schedule
        self.precond = preconditioning or CategoricalEDMPreconditioning()
        # Backward-compatible alias.
        self.preconditioning = self.precond
        self.train_cfg = train_cfg or UniformContinuousTimeTrainingConfig(
            num_steps=schedule.num_steps,
            t_min=schedule.t_min,
            t_max=schedule.t_max,
        )
        self.use_loss_weighting = bool(use_loss_weighting)
        self.rao_blackwellize_xt = bool(rao_blackwellize_xt)
        self.rao_blackwellize_xt_num_samples = int(rao_blackwellize_xt_num_samples)
        if self.rao_blackwellize_xt_num_samples < 1:
            raise ValueError("rao_blackwellize_xt_num_samples must be >= 1.")
        self.rao_blackwellize_xt_num_features = (
            None
            if rao_blackwellize_xt_num_features is None
            else int(rao_blackwellize_xt_num_features)
        )
        if (
            self.rao_blackwellize_xt_num_features is not None
            and self.rao_blackwellize_xt_num_features < 1
        ):
            raise ValueError("rao_blackwellize_xt_num_features must be >= 1.")
        self.eps = eps
        self.rngs = rngs
        self._mesh = sharding

    @property
    def num_classes(self) -> int:
        return self.schedule.num_classes

    @property
    def num_steps(self) -> int:
        return self.schedule.num_steps

    def _to_continuous_time(self, t: ArrayLike) -> Array:
        t_array = _require_float_time(t)
        t_cont = t_array.astype(jnp.float32)
        return jnp.clip(t_cont, self.schedule.t_min, self.schedule.t_max)

    def c_in(self, t: ArrayLike) -> Array:
        t_cont = self._to_continuous_time(t)
        return self.precond.c_in(t_cont, self.schedule)

    def c_out(self, t: ArrayLike) -> Array:
        t_cont = self._to_continuous_time(t)
        return self.precond.c_out(t_cont, self.schedule)

    def c_skip(self, t: ArrayLike) -> Array:
        t_cont = self._to_continuous_time(t)
        return self.precond.c_skip(t_cont, self.schedule)

    def c_t(self, t: ArrayLike) -> Array:
        t_cont = self._to_continuous_time(t)
        return self.precond.c_t(t_cont, self.schedule)

    def weight_fn(self, t: ArrayLike) -> Array:
        t_cont = self._to_continuous_time(t)
        return self.precond.weight_ce(t_cont, self.schedule)

    def _as_onehot(self, x_t: Array) -> Array:
        return _one_hot(jnp.asarray(x_t, dtype=jnp.int32), self.num_classes)

    def _base_probs_for(self, x_onehot: Array) -> Array:
        return self.schedule.base_probs.reshape((1,) * (x_onehot.ndim - 1) + (-1,))

    def _net_input(self, x_t: Array) -> Array:
        x_onehot = self._as_onehot(x_t)
        pi = self._base_probs_for(x_onehot)
        return x_onehot - pi

    def _net_forward(
        self,
        t: ArrayLike,
        x_features: Array,
        *args,
        **kwargs,
    ) -> Array:
        t_cont = self._to_continuous_time(t)
        t_embed = self.c_t(t_cont)
        x_embed = self.c_in(t_cont)[..., None] * x_features
        return self.net(t_embed, x_embed, *args, **kwargs)

    def __call__(
        self,
        t: ArrayLike,
        x_t: Array,
        *args,
        **kwargs,
    ) -> Array:
        x_features = self._net_input(x_t)
        return self._net_forward(t, x_features, *args, **kwargs)

    def _predict_x0_logits_from_probs(
        self,
        t: ArrayLike,
        x_t_probs: Array,
        *args,
        **kwargs,
    ) -> Array:
        x_t_probs = jnp.asarray(x_t_probs, dtype=jnp.float32)
        pi = self._base_probs_for(x_t_probs)
        x_features = x_t_probs - pi

        net_out = self._net_forward(t, x_features, *args, **kwargs)

        c_skip = self.c_skip(t)[..., None]
        c_out = self.c_out(t)[..., None]
        skip_probs = c_skip * x_t_probs + (1.0 - c_skip) * pi
        skip_logits = jnp.log(jnp.maximum(skip_probs, self.eps))

        return skip_logits + c_out * net_out

    def predict_x0_logits(
        self,
        t: ArrayLike,
        x_t: Array,
        *args,
        **kwargs,
    ) -> Array:
        x_t_probs = self._as_onehot(x_t)
        return self._predict_x0_logits_from_probs(t, x_t_probs, *args, **kwargs)

    def denoise_logits(
        self,
        t: ArrayLike,
        x_t: Array,
        *args,
        **kwargs,
    ) -> Array:
        return self.predict_x0_logits(t, x_t, *args, **kwargs)

    def predict_x0_probs(
        self,
        t: ArrayLike,
        x_t: Array,
        *args,
        **kwargs,
    ) -> Array:
        logits = self.predict_x0_logits(t, x_t, *args, **kwargs)
        return jax.nn.softmax(logits, axis=-1)

    def denoise_probs(
        self,
        t: ArrayLike,
        x_t: Array,
        *args,
        **kwargs,
    ) -> Array:
        return self.predict_x0_probs(t, x_t, *args, **kwargs)

    def score_reparameterized(
        self,
        rng: RngKey,
        t: ArrayLike,
        x_t: Array,
        *args,
        temperature: float = 1.0,
        num_samples: int = 1,
        **kwargs,
    ) -> Array:
        """
        Relaxed discrete score: gradient wrt relaxed x_t simplex variable.

        Uses Gumbel-Softmax reparameterization and computes
          grad_{x_t_relaxed} E[ <x_t_relaxed, log p_theta(x0|x_t_relaxed,t)> ].
        """
        x_onehot = self._as_onehot(x_t)
        log_x = jnp.log(jnp.maximum(x_onehot, self.eps))

        def one_score(sample_key):
            u = jax.random.uniform(
                sample_key,
                x_onehot.shape,
                minval=self.eps,
                maxval=1.0 - self.eps,
            )
            g = -jnp.log(-jnp.log(u))
            x_relaxed = jax.nn.softmax(
                (log_x + g) / jnp.maximum(temperature, self.eps),
                axis=-1,
            )

            def objective(x_in):
                logits = self._predict_x0_logits_from_probs(t, x_in, *args, **kwargs)
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                return jnp.sum(x_in * log_probs)

            return jax.grad(objective)(x_relaxed)

        keys = jax.random.split(rng, num_samples)
        grads = jax.vmap(one_score)(keys)
        return jnp.mean(grads, axis=0)

    def score(
        self,
        rng: RngKey,
        t: ArrayLike,
        x_t: Array,
        *args,
        temperature: float = 1.0,
        num_samples: int = 1,
        **kwargs,
    ) -> Array:
        return self.score_reparameterized(
            rng,
            t,
            x_t,
            *args,
            temperature=temperature,
            num_samples=num_samples,
            **kwargs,
        )

    def q_sample(self, rng: RngKey, x0: Array, t: ArrayLike) -> Array:
        return self.schedule.q_sample(rng, x0, self._to_continuous_time(t))

    def p_probs(
        self,
        t: ArrayLike,
        x_t: Array,
        *args,
        t_prev: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Array:
        t_cont = self._to_continuous_time(t)
        if t_prev is None:
            raise ValueError("p_probs requires t_prev for continuous-time reverse transitions.")
        t_prev_cont = self._to_continuous_time(t_prev)
        x0_probs = self.predict_x0_probs(t_cont, x_t, *args, **kwargs)
        return self.schedule.posterior_mixture_probs(
            x_t,
            x0_probs,
            t=t_cont,
            t_prev=t_prev_cont,
        )

    def p_sample(
        self,
        rng: RngKey,
        t: ArrayLike,
        x_t: Array,
        *args,
        t_prev: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Array:
        probs = self.p_probs(t, x_t, *args, t_prev=t_prev, **kwargs)
        return _sample_categorical(rng, probs)

    def _build_loss_fn(self):
        return build_time_dependent_multinomial_diffusion_loss(
            self.denoise_logits,
            q_sample_fn=self.q_sample,
            sample_timesteps_fn=self.train_cfg.sample_times,
            q_xt_given_x0_probs_fn=self.schedule.q_xt_given_x0_probs,
            alpha_bar_fn=self.schedule.alpha_bar,
            base_probs=self.schedule.base_probs,
            rao_blackwellize_xt=self.rao_blackwellize_xt,
            rao_blackwellize_xt_num_samples=self.rao_blackwellize_xt_num_samples,
            rao_blackwellize_xt_num_features=self.rao_blackwellize_xt_num_features,
            num_classes=self.num_classes,
            weight_fn=self.weight_fn if self.use_loss_weighting else None,
        )

    def loss(
        self,
        rng: RngKey,
        x0: Array,
        *args,
        t: Optional[Array] = None,
        loss_mask: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Array:
        loss_fn = self._build_loss_fn()
        model_kwargs = dict(self.train_cfg.loss_kwargs)
        model_kwargs.update(kwargs)
        return loss_fn(
            x0,
            *args,
            rng=rng,
            t=t,
            loss_mask=loss_mask,
            **model_kwargs,
        )

    def sample(
        self,
        rng: RngKey,
        shape: tuple[int, ...],
        *args,
        num_sample_steps: Optional[int] = None,
        **kwargs,
    ) -> Array:
        rng_prior, rng_loop = jax.random.split(rng, 2)
        x_init = self.schedule.prior_sample(rng_prior, shape)

        steps = self.num_steps if num_sample_steps is None else int(num_sample_steps)
        if steps <= 0:
            raise ValueError("num_sample_steps must be a positive integer.")

        times = jnp.linspace(
            self.schedule.t_max,
            self.schedule.t_min,
            steps + 1,
            dtype=jnp.float32,
        )
        step_keys = jax.random.split(rng_loop, steps)

        def scan_body(x_curr, inp):
            step_key, t_curr, t_prev = inp
            x_prev = self.p_sample(
                step_key,
                t_curr,
                x_curr,
                *args,
                t_prev=t_prev,
                **kwargs,
            )
            return x_prev, None

        x_final, _ = jax.lax.scan(
            scan_body,
            x_init,
            (step_keys, times[:-1], times[1:]),
        )
        return x_final


class MultinomialCosineDM(MultinomialDiffusion):
    """Recommended preset: cosine schedule + EDM-like preconditioning.

    Defaults are tuned for lower-variance continuous-time training.
    """

    def __init__(
        self,
        net: ModuleLike,
        num_classes: int,
        *,
        num_steps: int = 1000,
        t_min: float = 0.0,
        t_max: float = 1.0,
        s: float = 0.008,
        base_probs: Optional[ArrayLike] = None,
        train_cfg: Optional[CategoricalTrainingConfigProtocol] = None,
        preconditioning: Optional[CategoricalPreconditioningProtocol] = None,
        use_loss_weighting: bool = False,
        rao_blackwellize_xt: bool = False,
        rao_blackwellize_xt_num_samples: int = 4,
        rao_blackwellize_xt_num_features: Optional[int] = 1,
        rngs: nnx.RngStream | None = None,
        sharding: jax.sharding.Mesh | None = None,
    ):
        schedule = MultinomialDiffusionSchedule.from_cosine(
            num_steps=num_steps,
            num_classes=num_classes,
            s=s,
            base_probs=base_probs,
            t_min=t_min,
            t_max=t_max,
        )
        precond = preconditioning or CategoricalEDMPreconditioning()
        cfg = train_cfg or ImportanceContinuousTimeTrainingConfig(
            schedule=schedule,
            power=0.5,
            stratified=True,
            antithetic=True,
        )
        super().__init__(
            net,
            schedule,
            preconditioning=precond,
            train_cfg=cfg,
            use_loss_weighting=use_loss_weighting,
            rao_blackwellize_xt=rao_blackwellize_xt,
            rao_blackwellize_xt_num_samples=rao_blackwellize_xt_num_samples,
            rao_blackwellize_xt_num_features=rao_blackwellize_xt_num_features,
            rngs=rngs,
            sharding=sharding,
        )


class MultinomialLogSNRDM(MultinomialDiffusion):
    """Recommended preset: log-SNR schedule + EDM-like preconditioning.

    Defaults are tuned for stable discrete training:
      - moderated log-SNR range
      - lower-variance importance time sampling
    """

    def __init__(
        self,
        net: ModuleLike,
        num_classes: int,
        *,
        num_steps: int = 1000,
        t_min: float = 0.0,
        t_max: float = 1.0,
        logsnr_max: float = 9.0,
        logsnr_min: float = -6.0,
        max_beta: float = 20.0,
        base_probs: Optional[ArrayLike] = None,
        train_cfg: Optional[CategoricalTrainingConfigProtocol] = None,
        preconditioning: Optional[CategoricalPreconditioningProtocol] = None,
        use_loss_weighting: bool = False,
        rao_blackwellize_xt: bool = False,
        rao_blackwellize_xt_num_samples: int = 4,
        rao_blackwellize_xt_num_features: Optional[int] = 1,
        rngs: nnx.RngStream | None = None,
        sharding: jax.sharding.Mesh | None = None,
    ):
        schedule = MultinomialDiffusionSchedule.from_logsnr(
            num_steps=num_steps,
            num_classes=num_classes,
            logsnr_max=logsnr_max,
            logsnr_min=logsnr_min,
            max_beta=max_beta,
            base_probs=base_probs,
            t_min=t_min,
            t_max=t_max,
        )
        precond = preconditioning or CategoricalEDMPreconditioning()
        cfg = train_cfg or ImportanceContinuousTimeTrainingConfig(
            schedule=schedule,
            power=0.5,
            stratified=True,
            antithetic=True,
        )
        super().__init__(
            net,
            schedule,
            preconditioning=precond,
            train_cfg=cfg,
            use_loss_weighting=use_loss_weighting,
            rao_blackwellize_xt=rao_blackwellize_xt,
            rao_blackwellize_xt_num_samples=rao_blackwellize_xt_num_samples,
            rao_blackwellize_xt_num_features=rao_blackwellize_xt_num_features,
            rngs=rngs,
            sharding=sharding,
        )

__all__ = [
    "CategoricalScheduleProtocol",
    "CategoricalPreconditioningProtocol",
    "CategoricalTrainingConfigProtocol",
    "MultinomialDiffusionSchedule",
    "CategoricalEDMPreconditioning",
    "UniformContinuousTimeTrainingConfig",
    "ImportanceContinuousTimeTrainingConfig",
    "MultinomialDiffusion",
    "MultinomialCosineDM",
    "MultinomialLogSNRDM",
]

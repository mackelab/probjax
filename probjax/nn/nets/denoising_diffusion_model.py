from __future__ import annotations

from typing import Callable, Mapping

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss
from probjax.nn.sharding import mesh_context
from probjax.nn.nets.denoising_diffusion_configs import (
    BaseSolverConfig,
    CosineNoiseSchedule,
    EDMNoiseSchedule,
    EDMPreconditioning,
    EDMSolverConfig,
    EDMTrainingConfig,
    LogSNRTrainingConfig,
    NoiseScheduleProtocol,
    PreconditioningProtocol,
    SigmaEffEDMTrainingConfig,
    SolverConfigProtocol,
    TrainingConfigProtocol,
    UniformTTrainingConfig,
    VENoiseSchedule,
    VPNoiseSchedule,
    VSolverConfig,
)
from probjax.utils.typing import Array, ArrayLike, ModuleLike, PyTree, RngKey


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
        sharding: jax.sharding.Mesh | None = None,
        rngs: nnx.RngStream | None = None,
    ) -> None:
        if not isinstance(schedule, NoiseScheduleProtocol):
            raise TypeError("schedule must implement NoiseScheduleProtocol")
        if not isinstance(precond, PreconditioningProtocol):
            raise TypeError("precond must implement PreconditioningProtocol")
        if not isinstance(train_cfg, TrainingConfigProtocol):
            raise TypeError("train_cfg must implement TrainingConfigProtocol")
        if solver_cfg is not None and not isinstance(solver_cfg, SolverConfigProtocol):
            raise TypeError("solver_cfg must implement SolverConfigProtocol")

        self.rngs = rngs
        self.net: ModuleLike = net
        self.schedule = schedule
        self.precond = precond
        self.train_cfg = train_cfg
        self.solver_cfg = solver_cfg
        self._mesh = sharding
        if self.solver_cfg is not None and hasattr(self.solver_cfg, "set_schedule"):
            self.solver_cfg.set_schedule(self.schedule)
        with mesh_context(self._mesh):
            self.std0 = nnx.Variable(std0)
        self.last_layer = last_layer

    def set_solver_cfg(self, solver_cfg: SolverConfigProtocol) -> None:
        if not isinstance(solver_cfg, SolverConfigProtocol):
            raise TypeError("solver_cfg must implement SolverConfigProtocol")
        self.solver_cfg = solver_cfg
        if hasattr(self.solver_cfg, "set_schedule"):
            self.solver_cfg.set_schedule(self.schedule)

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
            std0=self.std0.value,
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def c_out(self, t: ArrayLike) -> Array:
        return self.precond.c_out(
            t,
            std0=self.std0.value,
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
            std0=self.std0.value,
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def weight_fn(self, t: ArrayLike) -> Array:
        return self.precond.weight_x0(
            t,
            std0=self.std0.value,
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def weight_fn_eps(self, t: ArrayLike) -> Array:
        return self.precond.weight_eps(
            t,
            std0=self.std0.value,
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
        )

    def weight_fn_v(self, t: ArrayLike) -> Array:
        return self.precond.weight_v(
            t,
            std0=self.std0.value,
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
        with mesh_context(self._mesh):
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
        with mesh_context(self._mesh):
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
        num_steps: int | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        if t_max is None:
            t_max = self.train_cfg.t_max
        if t_min is None:
            t_min = self.train_cfg.t_min
        if self.solver_cfg is None:
            raise ValueError(
                "solver_cfg is not set. Provide one at init or via set_solver_cfg()."
            )
        return self.solver_cfg.sample_ode(
            self,
            eps,
            t_max=t_max,
            t_min=t_min,
            num_steps=num_steps,
            collect_trace=collect_trace,
            *args,
            **kwargs,
        )

    def sample_sde(
        self,
        rng: RngKey,
        eps: PyTree[Array],
        num_steps: int | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
        collect_trace: bool = False,
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        if t_max is None:
            t_max = self.train_cfg.t_max
        if t_min is None:
            t_min = self.train_cfg.t_min
        if self.solver_cfg is None:
            raise ValueError(
                "solver_cfg is not set. Provide one at init or via set_solver_cfg()."
            )
        return self.solver_cfg.sample_sde(
            self,
            rng,
            eps,
            t_min=t_min,
            t_max=t_max,
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
        sharding: jax.sharding.Mesh | None = None,
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
        solver_cfg = solver or EDMSolverConfig(
            schedule=schedule,
            num_steps=num_steps,
            rho=rho,
        )
        super().__init__(
            net=net,
            schedule=schedule,
            precond=precond,
            train_cfg=train_cfg,
            solver_cfg=solver_cfg,
            std0=std0,
            last_layer=last_layer,
            rngs=rngs,
            sharding=sharding,
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
        sharding: jax.sharding.Mesh | None = None,
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
        solver_cfg = solver or VSolverConfig(
            schedule=schedule,
            num_steps=num_steps,
        )
        super().__init__(
            net=net,
            schedule=schedule,
            precond=precond,
            train_cfg=train_cfg,
            solver_cfg=solver_cfg,
            std0=std0,
            last_layer=last_layer,
            rngs=rngs,
            sharding=sharding,
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
        sharding: jax.sharding.Mesh | None = None,
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
        solver_cfg = solver or BaseSolverConfig(
            schedule=schedule,
            num_steps=num_steps,
        )
        super().__init__(
            net=net,
            schedule=schedule,
            precond=precond,
            train_cfg=train_cfg,
            solver_cfg=solver_cfg,
            std0=std0,
            last_layer=last_layer,
            rngs=rngs,
            sharding=sharding,
        )


class CosineDM(DiffusionDenoiser):
    """
    Cosine schedule variant:
      - CosineNoiseSchedule
      - EDMPreconditioning
      - LogSNRTrainingConfig (log-SNR sampling)
      - BaseSolverConfig by default
    """

    def __init__(
        self,
        net: ModuleLike,
        *,
        std0: float = 1.0,
        t_min: float = 1e-3,
        t_max: float = 1.0,
        s: float = 0.008,
        num_steps: int = 64,
        loss_type: str = "x0",
        loss_kwargs: Mapping[str, object] | None = None,
        last_layer: Callable[[Array], Array] | None = None,
        sharding: jax.sharding.Mesh | None = None,
        rngs: nnx.RngStream | None = None,
        solver: SolverConfigProtocol | None = None,
    ) -> None:
        schedule = CosineNoiseSchedule(t_min=t_min, t_max=t_max, s=s)
        precond = EDMPreconditioning()
        train_cfg = LogSNRTrainingConfig(
            schedule=schedule,
            loss_type=loss_type,
            loss_kwargs=dict(loss_kwargs or {}),
            t_min=t_min,
            t_max=t_max,
        )
        solver_cfg = solver or BaseSolverConfig(
            schedule=schedule,
            num_steps=num_steps,
        )
        super().__init__(
            net=net,
            schedule=schedule,
            precond=precond,
            train_cfg=train_cfg,
            solver_cfg=solver_cfg,
            std0=std0,
            last_layer=last_layer,
            rngs=rngs,
            sharding=sharding,
        )

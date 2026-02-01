from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import jax.tree_util
from flax import nnx

from probjax.nn.loss_fn.mean_flow_matching import build_mean_flow_matching_loss

from probjax.nn.nets.flow_matching_configs import (
    FlowPreconditioningProtocol,
    FlowSolverConfigProtocol,
    GaussianFlowPreconditioning,
    InterpolationScheduleProtocol,
    LinearFlowSolverConfig,
    LinearInterpolationSchedule,
)
from probjax.nn.nets.mean_flow_matching_configs import (
    FlowPairTrainingConfigProtocol,
    SigmoidPairFlowTrainingConfig,
)
from probjax.utils.typing import Array, ArrayLike, ModuleLike, RngKey


class MeanFlowMatcher(nnx.Module):
    def __init__(
        self,
        net: ModuleLike,
        schedule: InterpolationScheduleProtocol,
        preconditioning: FlowPreconditioningProtocol,
        train_cfg: FlowPairTrainingConfigProtocol,
        solver_cfg: FlowSolverConfigProtocol | None = None,
        mu0: ArrayLike = 0,
        std0: ArrayLike = 1,
        mu1: ArrayLike = 0.0,
        std1: ArrayLike = 1.0,
        rngs: nnx.RngStream | None = None,
        loss_kwargs: Mapping[str, object] | None = None,
        sharding: jax.sharding.Mesh | None = None,
    ):
        self.net: ModuleLike = net
        self._mesh = sharding
        self.mu0 = nnx.Variable(mu0)
        self.std0 = nnx.Variable(std0)
        self.mu1 = nnx.Variable(mu1)
        self.std1 = nnx.Variable(std1)
        self.rngs = rngs
        self._loss_kwargs: dict[str, object] = dict(loss_kwargs or {})

        if not isinstance(schedule, InterpolationScheduleProtocol):
            raise TypeError("schedule must implement InterpolationScheduleProtocol")
        if not isinstance(preconditioning, FlowPreconditioningProtocol):
            raise TypeError("preconditioning must implement FlowPreconditioningProtocol")
        if not isinstance(train_cfg, FlowPairTrainingConfigProtocol):
            raise TypeError("train_cfg must implement FlowPairTrainingConfigProtocol")
        if solver_cfg is not None and not isinstance(solver_cfg, FlowSolverConfigProtocol):
            raise TypeError("solver_cfg must implement FlowSolverConfigProtocol")

        self.schedule = schedule
        self.preconditioning = preconditioning
        self.train_cfg = train_cfg
        self.solver_cfg = solver_cfg

    def __call__(
        self, t: ArrayLike, x: Array, r: ArrayLike | None = None, *args, **kwargs
    ) -> Array:
        """
        Args:
            t: Current time t.
            x: Data at time t.
            r: Time at which to predict the mean (r > t)

        Returns:
            Predicted velocity to time r.
        """
        mu0 = self.mu0.value
        std0 = self.std0.value
        mu1 = self.mu1.value
        std1 = self.std1.value

        r: ArrayLike = t if r is None else jnp.clip(r, a_min=t, a_max=1.0)

        eps = getattr(self.preconditioning, "eps", 1e-8)
        approx_mu_t = self.schedule.path_mean(t, mu0, mu1)
        approx_std_t = jnp.maximum(self.schedule.path_std(t, std0, std1), eps)

        x_normed = jax.tree_util.tree_map(
            lambda x: (x - approx_mu_t) / approx_std_t, x
        )
        std_t = approx_std_t
        std_r = jnp.maximum(self.schedule.path_std(r, std0, std1), eps)
        a_t = self.schedule.a_t(t)
        b_t = self.schedule.b_t(t)
        denom = (a_t**2) * std0**2 + (b_t**2) * std1**2
        scale = (b_t * std1**2 - a_t * std0**2) / jnp.maximum(denom, eps)

        def g(h):
            return 1.0 + jnp.tanh(h / 0.1)

        geo_std = jnp.sqrt(std_r * std_t)
        scale_residual = geo_std * g(r - t)

        pred_mu1 = self.net(t, x_normed, *args, r=r, **kwargs)

        return mu1 - mu0 + scale * (x - approx_mu_t) + scale_residual * pred_mu1

    def loss(
        self,
        rng: RngKey,
        data: Array,
        *args,
        adaptive_weight_p: float = 0.3,
        adaptive_weight_eps: float = 1e-3,
        **kwargs,
    ) -> Array:
        loss_fn = build_mean_flow_matching_loss(
            self,
            schedule=self.schedule,
            weight_fn=None,
            **self._loss_kwargs,
        )

        rng_source, rng_times = jax.random.split(rng, 2)
        ndims = data.ndim - 2
        times_t, times_r = self.train_cfg.sample_times_pair(
            rng_times, (data.shape[0],) + (1,) * ndims
        )

        x0 = (
            jax.random.normal(rng_source, shape=data.shape) * self.std0.value
            + self.mu0.value
        )
        loss = loss_fn(
            times_r,
            times_t,
            x0,
            data,
            *args,
            adaptive_weight_p=adaptive_weight_p,
            adaptive_weight_eps=adaptive_weight_eps,
            **kwargs,
        )
        return loss

    def solve_schedule(
        self,
        t_min: float = 0.0,
        t_max: float = 1.0,
        num_steps: int | None = None,
    ) -> Array:
        return self.solver_cfg.solve_schedule(t_min=t_min, t_max=t_max, num_steps=num_steps)


class LinearMeanFlow(MeanFlowMatcher):
    def __init__(
        self,
        net: ModuleLike,
        mu0: ArrayLike = 0,
        std0: ArrayLike = 1,
        mu1: ArrayLike = 0.0,
        std1: ArrayLike = 1.0,
        rngs: nnx.RngStream | None = None,
        loss_kwargs: Mapping[str, object] | None = None,
        schedule: InterpolationScheduleProtocol | None = None,
        preconditioning: FlowPreconditioningProtocol | None = None,
        train_cfg: FlowPairTrainingConfigProtocol | None = None,
        solver_cfg: FlowSolverConfigProtocol | None = None,
        sharding: jax.sharding.Mesh | None = None,
    ):
        schedule = schedule or LinearInterpolationSchedule()
        preconditioning = preconditioning or GaussianFlowPreconditioning()
        train_cfg = train_cfg or SigmoidPairFlowTrainingConfig()
        solver_cfg = solver_cfg or LinearFlowSolverConfig()
        super().__init__(
            net,
            schedule=schedule,
            preconditioning=preconditioning,
            train_cfg=train_cfg,
            solver_cfg=solver_cfg,
            mu0=mu0,
            std0=std0,
            mu1=mu1,
            std1=std1,
            rngs=rngs,
            loss_kwargs=loss_kwargs,
            sharding=sharding,
        )

    def solve_schedule(self, num_steps: int = 50) -> Array:
        ts = jnp.linspace(0, 1, num_steps)
        return ts

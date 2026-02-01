from typing import Mapping, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util
from flax import nnx

from probjax.nn.loss_fn.flow_matching import build_flow_matching_loss
from probjax.nn.sharding import mesh_context
from probjax.nn.nets.flow_matching_configs import (
    CosineInterpolationSchedule,
    FlowPreconditioningProtocol,
    FlowSolverConfigProtocol,
    FlowTrainingConfigProtocol,
    GaussianFlowPreconditioning,
    InterpolationScheduleProtocol,
    LinearFlowSolverConfig,
    LinearInterpolationSchedule,
    LogitNormalFlowTrainingConfig,
    QuadraticInterpolationSchedule,
    UniformFlowTrainingConfig,
)
from probjax.utils.typing import Array, ArrayLike, ModuleLike, PyTree, RngKey


class FlowMatcher(nnx.Module):
    """
    Composable flow matcher:

      - schedule   : InterpolationScheduleProtocol
      - precond    : FlowPreconditioningProtocol
      - train_cfg  : FlowTrainingConfigProtocol
      - solver_cfg : FlowSolverConfigProtocol | None
    """

    schedule: InterpolationScheduleProtocol
    preconditioning: FlowPreconditioningProtocol
    train_cfg: FlowTrainingConfigProtocol
    solver_cfg: FlowSolverConfigProtocol | None

    def __init__(
        self,
        net: ModuleLike,
        schedule: InterpolationScheduleProtocol,
        preconditioning: FlowPreconditioningProtocol,
        train_cfg: FlowTrainingConfigProtocol,
        solver_cfg: FlowSolverConfigProtocol | None = None,
        mu0: ArrayLike = 0.0,
        std0: ArrayLike = 1.0,
        mu1: ArrayLike = 0.0,
        std1: ArrayLike = 1.0,
        loss_kwargs: Mapping[str, object] | None = None,
        sharding: jax.sharding.Mesh | None = None,
        rngs: nnx.RngStream | None = None,
    ):
        if not isinstance(schedule, InterpolationScheduleProtocol):
            raise TypeError("schedule must implement InterpolationScheduleProtocol")
        if not isinstance(preconditioning, FlowPreconditioningProtocol):
            raise TypeError("preconditioning must implement FlowPreconditioningProtocol")
        if not isinstance(train_cfg, FlowTrainingConfigProtocol):
            raise TypeError("train_cfg must implement FlowTrainingConfigProtocol")
        if solver_cfg is not None and not isinstance(solver_cfg, FlowSolverConfigProtocol):
            raise TypeError("solver_cfg must implement FlowSolverConfigProtocol")

        self.rngs = rngs
        self.net: ModuleLike = net
        self.schedule = schedule
        self.preconditioning = preconditioning
        self.train_cfg = train_cfg
        self.solver_cfg = solver_cfg
        self._mesh = sharding

        with mesh_context(self._mesh):
            self.mu0 = nnx.Variable(mu0)
            self.std0 = nnx.Variable(std0)
            self.mu1 = nnx.Variable(mu1)
            self.std1 = nnx.Variable(std1)

        self._loss_kwargs: dict[str, object] = dict(loss_kwargs or {})

    def set_solver_cfg(self, solver_cfg: FlowSolverConfigProtocol) -> None:
        if not isinstance(solver_cfg, FlowSolverConfigProtocol):
            raise TypeError("solver_cfg must implement FlowSolverConfigProtocol")
        self.solver_cfg = solver_cfg

    def __call__(
        self, t: ArrayLike, x: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]:
        """Forward pass of the model - v-prediction.

        We do a Gaussian closed-form preconditioning scheme. We know that
        p0(x) = N(x; mu0, std0**2) and let's assume that p1(x) = N(x; mu1, std1**2).
        Then the optimal v-prediction target is tractable and given by:

        E[x1-x0|xt] = (mu1 - mu0) + s(t) * (xt - mu_t)

        Where s(t) = d/dt log sigma(t) and with sigma(t) = sqrt{t**2 * std1**2 + (1 - t) ** 2 * std0**2}
        we have that s(t) = (t * std1**2) / ((1 - t) ** 2 * std0**2 + t**2 * std1**2)

        We can plug in all the values for this but predict mu_t by the model.

        """
        with mesh_context(self._mesh):
            # With preconditioning
            mu0 = self.mu0.value
            std0 = self.std0.value
            mu1 = self.mu1.value
            std1 = self.std1.value

            x_normed, approx_mut, approx_stdt = self.preconditioning.normalize(
                self.schedule, t, x, mu0, mu1, std0, std1
            )
            residual_pred = self.net(t, x_normed, *args, **kwargs)
            residual_correction = jax.tree_util.tree_map(
                lambda r: approx_stdt * r, residual_pred
            )
            return self.preconditioning.decode_velocity(
                self.schedule,
                t,
                x,
                mu0,
                mu1,
                std0,
                std1,
                residual_correction,
            )

    def score(self, t: ArrayLike, x: PyTree[Array], *args, **kwargs) -> PyTree[Array]:
        """Score function for the model."""
        raise NotImplementedError(
            "Implemented only for specific implementation of this base class"
        )

    def denoise(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        """Denoise the input x at time t."""
        raise NotImplementedError(
            "Implemented only for specific implementation of this base class"
        )

    def loss(
        self,
        rng: RngKey,
        data: Array,
        *args,
        **kwargs,
    ) -> Array:
        with mesh_context(self._mesh):
            loss_fn = build_flow_matching_loss(
                self,
                schedule=self.schedule,
                weight_fn=None,
                interpolation_grad_fn=None,
                interpolation_noise_grad_fn=None,
                **self._loss_kwargs,
            )

            rng_source, rng_times = jax.random.split(rng, 2)

            # Generate noise for x0
            x0 = (
                jax.random.normal(rng_source, shape=data.shape) * self.std0.value
                + self.mu0.value
            )

            # Get shape from data for time scheduling
            data_shape = data.shape
            ndims = data.ndim - 2
            times = self.train_cfg.sample_times(
                rng_times, (data_shape[0],) + (1,) * ndims
            )

            loss = loss_fn(times, x0, data, *args, **kwargs)
            return loss

    def solve_schedule(
        self,
        t_min: float = 0.0,
        t_max: float = 1.0,
        num_steps: int | None = None,
    ) -> Array:
        return self.solver_cfg.solve_schedule(t_min=t_min, t_max=t_max, num_steps=num_steps)


class LinearFlow(FlowMatcher):
    def __init__(
        self,
        net: ModuleLike,
        mu0: ArrayLike = 0.0,
        std0: ArrayLike = 1.0,
        mu1: ArrayLike = 0.0,
        std1: ArrayLike = 1.0,
        rngs: nnx.RngStream | None = None,
        loss_kwargs: Mapping[str, object] | None = None,
        train_cfg: FlowTrainingConfigProtocol | None = None,
        solver_cfg: FlowSolverConfigProtocol | None = None,
        schedule: InterpolationScheduleProtocol | None = None,
        preconditioning: FlowPreconditioningProtocol | None = None,
        sharding: jax.sharding.Mesh | None = None,
    ):
        schedule = schedule or LinearInterpolationSchedule()
        preconditioning = preconditioning or GaussianFlowPreconditioning()
        train_cfg = train_cfg or LogitNormalFlowTrainingConfig(mu=0.7, scale=1.0)
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

    def denoise(self, t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
        # x0 is noise
        # x1 is data
        # xt = (1 - t) * x0 + t * x1
        # x0 = (xt - t * x1) / (1 - t)
        # E[x1-x0|xt] = E[x1 - (xt - t * x1) / (1 - t)|xt]
        # = (xt - E[x1|xt])/(1 - t)
        # So we can recover the denoiser by
        # E[x1|xt] = xt - (1-t)*E[x1-x0|xt]
        v = self.__call__(t, x)

        def denoise_leaf(leaf_x, leaf_v):
            return leaf_x - (1 - t) * leaf_v

        return jax.tree_util.tree_map(denoise_leaf, x, v)

    def score(
        self, t: ArrayLike, x: PyTree[Array], max_t: float = 1 - 1e-3
    ) -> PyTree[Array]:
        # We can recover a "denoiser" so we can use it to get the score
        # using Tweedie's formula
        # Pertubration kernel is given by
        # p(xt|x1) = N(xt, t*x1, (1-t)*std0**2)
        # score = (t * E[x1|xt] + (1-t)*mu0 - xt) / (1-t)**2*std0**2
        # = (t * (xt - (1-t)*v) + (1-t)*mu0 - xt) / (1-t)**2*std0**2
        # = (-(1-t) * xt - t*(1-t)*v + (1-t)*mu0) / (1-t)**2*std0**2
        # = (- t*v + mu0 - xt) / (1-t)*std0**2

        mu0 = self.mu0.value
        std0 = self.std0.value
        t = jnp.clip(t, 0, max_t)
        v = self.__call__(t, x)

        def score_leaf(leaf_x, leaf_v):
            return (-t * leaf_v + mu0 - leaf_x) / ((1 - t) * std0**2)

        return jax.tree_util.tree_map(score_leaf, x, v)

    def noise_schedule(
        self, rng: RngKey, shape: Tuple[int, ...], mu: float = 0.0, scale: float = 1.0
    ) -> Array:
        return jax.nn.sigmoid(jax.random.normal(rng, shape=shape + (1,)) * scale + mu)

    def solve_schedule(self, num_steps: int = 50) -> Array:
        ts = jnp.linspace(0, 1, num_steps)
        return ts

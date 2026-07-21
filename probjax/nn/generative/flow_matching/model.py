from functools import partial
from typing import Mapping, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util
from flax import nnx

from probjax.nn.generative.base import GenerativeModel
from probjax.nn.generative.flow_matching.config import (
    FlowPreconditioningProtocol,
    FlowSolverConfigProtocol,
    FlowTrainingConfigProtocol,
    GaussianFlowPreconditioning,
    InterpolationScheduleProtocol,
    LinearFlowSolverConfig,
    LinearInterpolationSchedule,
    LogitNormalFlowTrainingConfig,
)
from probjax.nn.generative.sampling import (
    _ExportedSampler,
    make_ode_sample_fn,
    sample_normal,
)
from probjax.nn.losses.flow_matching import build_flow_matching_loss
from probjax.utils.functions import generic_drift
from probjax.utils.typing import Array, ArrayLike, ModuleLike, PyTree, RngKey


def _flow_ode_drift(model, context):
    return model._build_ode_drift(context)


class FlowMatcher(GenerativeModel):
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
        rngs: nnx.RngStream | None = None,
    ):
        if not isinstance(schedule, InterpolationScheduleProtocol):
            raise TypeError("schedule must implement InterpolationScheduleProtocol")
        if not isinstance(preconditioning, FlowPreconditioningProtocol):
            raise TypeError(
                "preconditioning must implement FlowPreconditioningProtocol"
            )
        if not isinstance(train_cfg, FlowTrainingConfigProtocol):
            raise TypeError("train_cfg must implement FlowTrainingConfigProtocol")
        if solver_cfg is not None and not isinstance(
            solver_cfg, FlowSolverConfigProtocol
        ):
            raise TypeError("solver_cfg must implement FlowSolverConfigProtocol")

        self.rngs = rngs
        self.net: ModuleLike = net
        self.schedule = schedule
        self.preconditioning = preconditioning
        self.train_cfg = train_cfg
        self.solver_cfg = solver_cfg

        self.mu0 = nnx.Variable(mu0)
        self.std0 = nnx.Variable(std0)
        self.mu1 = nnx.Variable(mu1)
        self.std1 = nnx.Variable(std1)

        self._loss_kwargs: dict[str, object] = dict(loss_kwargs or {})

    def set_solver_cfg(self, solver_cfg: FlowSolverConfigProtocol) -> None:
        if not isinstance(solver_cfg, FlowSolverConfigProtocol):
            raise TypeError("solver_cfg must implement FlowSolverConfigProtocol")
        self.solver_cfg = solver_cfg
        self._clear_distribution_cache()

    def __call__(
        self,
        t: ArrayLike,
        x: PyTree[Array],
        *args,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> PyTree[Array]:
        """Forward pass of the model - v-prediction.

        Uses a Gaussian closed-form preconditioning scheme. For endpoints
        p0(x) = N(x; mu0, std0**2) and p1(x) = N(x; mu1, std1**2), the
        marginal velocity field under the interpolation schedule is

        E[x1-x0|xt] = (mu1 - mu0) + s(t) * (xt - mu_t)

        where mu_t and sigma_t are the schedule's interpolated mean and
        standard deviation, and s(t) = d/dt log sigma(t). For the linear
        schedule this gives

        s(t) = (t * std1**2 - (1 - t) * std0**2)
               / ((1 - t)**2 * std0**2 + t**2 * std1**2).

        The network operates on the normalized input ``x_normed = (x - mu_t)
        / sigma_t`` and predicts a displacement correction ``v_out`` in the
        same normalized space. The preconditioner maps it back to data space
        as ``sigma_t * v_out`` and adds it inside the velocity-scaled term:

        v(t, x) = (mu1 - mu0) + s(t) * ((x - mu_t) + sigma_t * v_out).

        ``mu_t`` and ``sigma_t`` are computed analytically from the schedule;
        the network never predicts them.
        """
        # With preconditioning
        mu0 = self.mu0.get_value()
        std0 = self.std0.get_value()
        mu1 = self.mu1.get_value()
        std1 = self.std1.get_value()

        x_normed, approx_mut, approx_stdt = self.preconditioning.normalize(
            self.schedule, t, x, mu0, mu1, std0, std1
        )
        v_out = self.net(t, x_normed, *args, rng=rng, **kwargs)
        v_out_data = jax.tree_util.tree_map(lambda v: approx_stdt * v, v_out)
        return self.preconditioning.decode_velocity(
            self.schedule,
            t,
            x,
            mu0,
            mu1,
            std0,
            std1,
            v_out_data,
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

    def _build_loss_fn(self):
        return build_flow_matching_loss(
            self,
            schedule=self.schedule,
            **self._loss_kwargs,
        )

    def loss(
        self,
        rng: RngKey,
        data: Array,
        *args,
        **kwargs,
    ) -> Array:
        loss_fn = self._build_loss_fn()

        rng_source, rng_times = jax.random.split(rng, 2)

        # Generate noise for x0
        x0 = (
            jax.random.normal(rng_source, shape=data.shape) * self.std0.get_value()
            + self.mu0.get_value()
        )

        # Get shape from data for time scheduling
        data_shape = data.shape
        ndims = data.ndim - 2
        times = self.train_cfg.sample_times(rng_times, (data_shape[0],) + (1,) * ndims)

        loss = loss_fn(times, x0, data, *args, **kwargs)
        return loss

    def solve_schedule(
        self,
        t_min: float = 0.0,
        t_max: float = 1.0,
        num_steps: int | None = None,
    ) -> Array:
        if self.solver_cfg is None:
            raise ValueError(
                "solver_cfg is not set. Provide one at init or via set_solver_cfg()."
            )
        return self.solver_cfg.solve_schedule(
            t_min=t_min, t_max=t_max, num_steps=num_steps
        )

    def _sample_base(self, rng, sample_shape, spec):
        return sample_normal(
            rng,
            sample_shape,
            spec,
            loc=self.mu0.get_value(),
            scale=self.std0.get_value(),
        )

    def _build_ode_drift(self, context=None):
        def drift(t, value):
            if context is None:
                return self(t, value)
            return self(t, value, context=context)

        return generic_drift(drift)

    def _distribution_sampler(
        self,
        event_spec,
        *,
        num_steps: int | None = None,
        method: str = "rk4",
        t_min: float = 0.0,
        t_max: float = 1.0,
        collect_trace: bool = False,
        dtype=jnp.float32,
        context_spec=None,
    ) -> _ExportedSampler:
        """Build and cache an ODE sampler with symbolic batch size.

        ``event_spec`` may be a plain shape tuple or a pytree of shapes /
        ``jax.ShapeDtypeStruct`` for structured data. ``context_spec`` uses
        the same format and enables a required, batch-aligned context input.
        """
        if self.solver_cfg is None:
            raise ValueError(
                "solver_cfg is not set. Provide one at init or via set_solver_cfg()."
            )
        steps = self.solver_cfg.num_steps if num_steps is None else int(num_steps)
        ts = self.solve_schedule(t_min=t_min, t_max=t_max, num_steps=steps)
        make_sample_fn = partial(
            make_ode_sample_fn,
            ts=ts,
            prototype=self._build_ode_drift(),
            build_drift=_flow_ode_drift,
            method=method,
            collect_trace=collect_trace,
        )

        return self._build_exported_sampler(
            ("flow", steps, method, t_min, t_max, collect_trace),
            event_spec,
            make_sample_fn,
            dtype=dtype,
            context_spec=context_spec,
            trace=collect_trace,
        )


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
    ):
        schedule = schedule or LinearInterpolationSchedule()
        preconditioning = preconditioning or GaussianFlowPreconditioning()
        train_cfg = train_cfg or LogitNormalFlowTrainingConfig(mu=0.7, scale=1.0)
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

        mu0 = self.mu0.get_value()
        std0 = self.std0.get_value()
        t = jnp.clip(t, 0, max_t)
        v = self.__call__(t, x)

        def score_leaf(leaf_x, leaf_v):
            return (-t * leaf_v + mu0 - leaf_x) / ((1 - t) * std0**2)

        return jax.tree_util.tree_map(score_leaf, x, v)

    def noise_schedule(
        self, rng: RngKey, shape: Tuple[int, ...], mu: float = 0.0, scale: float = 1.0
    ) -> Array:
        return jax.nn.sigmoid(jax.random.normal(rng, shape=shape + (1,)) * scale + mu)

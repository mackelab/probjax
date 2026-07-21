from functools import partial
from typing import Mapping

import jax
import jax.numpy as jnp
import jax.tree_util
from flax import nnx

from probjax.nn.generative.base import GenerativeModel
from probjax.nn.generative.flow_matching.config import (
    FlowPreconditioningProtocol,
    FlowSolverConfigProtocol,
    GaussianFlowPreconditioning,
    InterpolationScheduleProtocol,
    LinearFlowSolverConfig,
    LinearInterpolationSchedule,
)
from probjax.nn.generative.mean_flow.config import (
    FlowPairTrainingConfigProtocol,
    SigmoidPairFlowTrainingConfig,
)
from probjax.nn.generative.sampling import (
    _ExportedSampler,
    make_scan_sample_fn,
    sample_normal,
)
from probjax.nn.losses.mean_flow import build_mean_flow_matching_loss
from probjax.utils.typing import Array, ArrayLike, ModuleLike, PyTree, RngKey


def _mean_flow_step(model, value, times, context):
    t, r = times
    if context is None:
        velocity = model(t, value, r=r)
    else:
        velocity = model(t, value, r=r, context=context)
    return jax.tree.map(
        lambda current, update: current + (r - t) * update,
        value,
        velocity,
    )


class MeanFlowMatcher(GenerativeModel):
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
    ):
        self.net: ModuleLike = net
        self.mu0 = nnx.Variable(mu0)
        self.std0 = nnx.Variable(std0)
        self.mu1 = nnx.Variable(mu1)
        self.std1 = nnx.Variable(std1)
        self.rngs = rngs
        self._loss_kwargs: dict[str, object] = dict(loss_kwargs or {})

        if not isinstance(schedule, InterpolationScheduleProtocol):
            raise TypeError("schedule must implement InterpolationScheduleProtocol")
        if not isinstance(preconditioning, FlowPreconditioningProtocol):
            raise TypeError(
                "preconditioning must implement FlowPreconditioningProtocol"
            )
        if not isinstance(train_cfg, FlowPairTrainingConfigProtocol):
            raise TypeError("train_cfg must implement FlowPairTrainingConfigProtocol")
        if solver_cfg is not None and not isinstance(
            solver_cfg, FlowSolverConfigProtocol
        ):
            raise TypeError("solver_cfg must implement FlowSolverConfigProtocol")

        self.schedule = schedule
        self.preconditioning = preconditioning
        self.train_cfg = train_cfg
        self.solver_cfg = solver_cfg

    def set_solver_cfg(self, solver_cfg: FlowSolverConfigProtocol) -> None:
        if not isinstance(solver_cfg, FlowSolverConfigProtocol):
            raise TypeError("solver_cfg must implement FlowSolverConfigProtocol")
        self.solver_cfg = solver_cfg
        self._clear_distribution_cache()

    def __call__(
        self,
        t: ArrayLike,
        x: PyTree[Array],
        r: ArrayLike | None = None,
        *args,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> PyTree[Array]:
        """
        Predicted velocity to time r.

        Uses a Gaussian closed-form preconditioning scheme. The marginal
        mean-velocity field is

        v(t, x) = (mu1 - mu0) + s(t) * (x - mu_t)

        where s(t) = d/dt log sigma(t). The network operates on the
        normalized input ``x_normed = (x - mu_t) / sigma_t`` and predicts a
        displacement correction ``v_out`` in the same normalized space. The
        preconditioner maps it back to data space as ``sigma_t * v_out`` and
        adds it inside the velocity-scaled term:

        v(t, x, r) = (mu1 - mu0) + s(t) * ((x - mu_t) + sigma_t * v_out).

        ``mu_t`` and ``sigma_t`` are computed analytically from the schedule;
        the network never predicts them.

        Args:
            t: Current time t.
            x: Data at time t.
            r: Time at which to predict the mean (r > t).

        Returns:
            Predicted velocity to time r.
        """
        mu0 = self.mu0.get_value()
        std0 = self.std0.get_value()
        mu1 = self.mu1.get_value()
        std1 = self.std1.get_value()

        r: ArrayLike = t if r is None else jnp.clip(r, min=t, max=1.0)

        eps = getattr(self.preconditioning, "eps", 1e-8)
        approx_mu_t = self.schedule.path_mean(t, mu0, mu1)
        approx_std_t = jnp.maximum(self.schedule.path_std(t, std0, std1), eps)

        x_normed = jax.tree_util.tree_map(lambda x: (x - approx_mu_t) / approx_std_t, x)
        std_t = approx_std_t
        a_t = self.schedule.a_t(t)
        b_t = self.schedule.b_t(t)
        denom = (a_t**2) * std0**2 + (b_t**2) * std1**2
        scale = (b_t * std1**2 - a_t * std0**2) / jnp.maximum(denom, eps)

        v_out = self.net(t, x_normed, *args, r=r, rng=rng, **kwargs)
        v_out_data = jax.tree_util.tree_map(lambda v: std_t * v, v_out)

        return jax.tree.map(
            lambda value, update: mu1 - mu0 + scale * (value - approx_mu_t + update),
            x,
            v_out_data,
        )

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
            jax.random.normal(rng_source, shape=data.shape) * self.std0.get_value()
            + self.mu0.get_value()
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

    def _distribution_sampler(
        self,
        event_spec,
        *,
        num_steps: int | None = None,
        dtype=jnp.float32,
        context_spec=None,
    ) -> _ExportedSampler:
        """Build and cache a mean-flow sampler with symbolic batch size.

        ``event_spec`` may be a plain shape tuple or a pytree of shapes /
        ``jax.ShapeDtypeStruct`` for structured data. ``context_spec`` uses
        the same format and enables a required, batch-aligned context input.
        """
        if self.solver_cfg is None:
            raise ValueError(
                "solver_cfg is not set. Provide one at init or via set_solver_cfg()."
            )
        steps = self.solver_cfg.num_steps if num_steps is None else int(num_steps)
        ts = self.solve_schedule(num_steps=steps)
        make_sample_fn = partial(
            make_scan_sample_fn,
            xs=(ts[:-1], ts[1:]),
            step=_mean_flow_step,
        )

        return self._build_exported_sampler(
            ("mean-flow", steps),
            event_spec,
            make_sample_fn,
            dtype=dtype,
            context_spec=context_spec,
        )


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
        )

    def solve_schedule(self, num_steps: int = 50) -> Array:
        ts = jnp.linspace(0, 1, num_steps)
        return ts

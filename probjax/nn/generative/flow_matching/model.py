from typing import Mapping, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util
from flax import nnx

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
    BuiltSampler,
    cached_sampler,
    clear_sampler_cache,
    export_sampler,
)
from probjax.nn.losses.flow_matching import build_flow_matching_loss
from probjax.stats.fit import FitMixin
from probjax.utils.functions import generic_drift
from probjax.utils.odeint import odeint
from probjax.utils.typing import Array, ArrayLike, ModuleLike, PyTree, RngKey


class FlowMatcher(nnx.Module, FitMixin):
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
        clear_sampler_cache(self)

    def __call__(
        self,
        t: ArrayLike,
        x: PyTree[Array],
        *args,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> PyTree[Array]:
        """Forward pass of the model - v-prediction.

        We do a Gaussian closed-form preconditioning scheme. We know that
        p0(x) = N(x; mu0, std0**2) and let's assume that p1(x) = N(x; mu1, std1**2).
        Then the optimal v-prediction target is tractable and given by:

        E[x1-x0|xt] = (mu1 - mu0) + s(t) * (xt - mu_t)

        Where s(t) = d/dt log sigma(t), with sigma(t) defined by the
        interpolation of the endpoint variances.
        we have that s(t) = (t * std1**2) / ((1 - t) ** 2 * std0**2 + t**2 * std1**2)

        We can plug in all the values for this but predict mu_t by the model.

        """
        # With preconditioning
        mu0 = self.mu0.get_value()
        std0 = self.std0.get_value()
        mu1 = self.mu1.get_value()
        std1 = self.std1.get_value()

        x_normed, approx_mut, approx_stdt = self.preconditioning.normalize(
            self.schedule, t, x, mu0, mu1, std0, std1
        )
        residual_pred = self.net(t, x_normed, *args, rng=rng, **kwargs)
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

    def build_sampler(
        self,
        event_shape: tuple[int, ...],
        *,
        num_steps: int | None = None,
        method: str = "rk4",
        t_min: float = 0.0,
        t_max: float = 1.0,
        dtype=jnp.float32,
    ) -> BuiltSampler:
        """Build and cache an ODE sampler with symbolic batch size."""
        if self.solver_cfg is None:
            raise ValueError(
                "solver_cfg is not set. Provide one at init or via set_solver_cfg()."
            )
        event_shape = tuple(int(size) for size in event_shape)
        dtype = jnp.dtype(dtype)
        steps = self.solver_cfg.num_steps if num_steps is None else int(num_steps)
        key = ("flow", event_shape, dtype.str, steps, method, t_min, t_max)

        def build():
            graphdef, state = nnx.split(self)
            ts = self.solve_schedule(t_min=t_min, t_max=t_max, num_steps=steps)

            def drift_fn(t, x, current_state):
                model = nnx.merge(graphdef, current_state)
                return model(t, x)

            drift = generic_drift(drift_fn)

            def sample_fn(current_state, eps):
                return odeint(
                    drift,
                    eps,
                    ts,
                    current_state,
                    method=method,
                    dtype=dtype,
                    collect_trace=False,
                )

            exported = export_sampler(state, event_shape, dtype, sample_fn)
            return BuiltSampler(
                self,
                graphdef,
                exported,
                event_shape,
                dtype,
                base="flow",
            )

        return cached_sampler(self, key, build)

    def sample(
        self,
        eps: Array,
        *,
        num_steps: int | None = None,
        method: str = "rk4",
    ) -> Array:
        """Generate samples by integrating the velocity ODE from ``eps``.

        Solves ``dx/dt = self(t, x)`` along the schedule returned by
        :meth:`solve_schedule`, starting at ``x(t_0) = eps``, and returns
        the terminal state.

        Args:
            eps: Starting noise of shape ``batch_shape + event_shape``,
                typically drawn from the base distribution
                ``N(mu0, std0**2)``.
            num_steps: Grid resolution forwarded to ``solve_schedule``.
            method: ODE method for :func:`probjax.utils.odeint`
                (``"rk4"`` default; ``"tsit5"``/``"dopri5"`` for adaptive
                integration when paired with a ``step_size_adaptor``).
        """
        from probjax.utils.odeint import odeint

        kwargs = {} if num_steps is None else {"num_steps": num_steps}
        ts = self.solve_schedule(**kwargs)
        traj = odeint(lambda t, x: self(t, x), eps, ts, method=method)
        return traj[-1]

    def as_distribution(
        self,
        event_shape: tuple,
        *,
        num_steps: int | None = None,
        method: str = "rk4",
    ):
        """Expose this flow matcher as a :class:`probjax.stats.base.DistributionAPI`.

        ``rvs`` draws ``eps ~ N(mu0, std0**2)`` and integrates the velocity
        ODE; ``logpdf`` raises (use Hutchinson + ODE for change-of-
        variables — pass ``logpdf_fn`` to :class:`LearnedDistribution`
        explicitly if you need it).
        """
        from probjax.nn.distribution import LearnedDistribution

        event_shape = tuple(int(d) for d in event_shape)
        sampler = self.build_sampler(
            event_shape,
            num_steps=num_steps,
            method=method,
        )

        def sampler_fn(rng, batch_shape):
            return sampler(rng, tuple(batch_shape))

        return LearnedDistribution(
            event_shape=event_shape,
            sampler_fn=sampler_fn,
            name=f"{type(self).__name__}",
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

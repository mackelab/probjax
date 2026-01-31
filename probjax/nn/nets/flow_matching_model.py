from typing import Callable, Mapping, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util
from flax import nnx

from probjax.nn.loss_fn.flow_matching import (
    build_flow_matching_loss,
    build_mean_flow_matching_loss,
    build_flow_matching_loss_from_schedule,
    build_mean_flow_matching_loss_from_schedule,
)
from probjax.nn.nets.flow_matching_configs import (
    CosineInterpolationSchedule,
    FlowPreconditioningProtocol,
    FlowPairTrainingConfigProtocol,
    FlowSolverConfigProtocol,
    FlowTrainingConfigProtocol,
    GaussianFlowPreconditioning,
    InterpolationScheduleProtocol,
    LinearFlowSolverConfig,
    LinearInterpolationSchedule,
    LogitNormalFlowTrainingConfig,
    QuadraticInterpolationSchedule,
    SigmoidPairFlowTrainingConfig,
    UniformFlowTrainingConfig,
)
from probjax.utils.typing import Array, ArrayLike, ModuleLike, PyTree, RngKey


class FlowMatcher(nnx.Module):
    r"""This serves as a base class for diffusion denoising models.

    Specifically this implementation will minimize a denoising network
    with loss of form:
    $$
    \mathcal{L} = \mathbb{E}_{t,x,x_t} \left[w_t \left\| x -x_t \right\|^2 \right]
    $$
    where $x_t$ is the noisy version of $x$ at time $t$. Althought equivalent in
    target this is somewhat different from using denoising score matching.

    """

    # Can be overwritten
    interpolation_fn: Callable
    interpolation_std_fn: Callable
    drift: Callable
    diffusion: Callable

    def __init__(
        self,
        net: ModuleLike,
        mu0: ArrayLike = 0.0,
        std0: ArrayLike = 1.0,
        mu1: ArrayLike = 0.0,
        std1: ArrayLike = 1.0,
        interpolation_fn: Callable | None = None,
        interpolation_std_fn: Callable | None = None,
        loss_kwargs: Mapping[str, object] | None = None,
        rngs: nnx.RngStream | None = None,
        interp_schedule: InterpolationScheduleProtocol | None = None,
        preconditioning: FlowPreconditioningProtocol | None = None,
        train_cfg: FlowTrainingConfigProtocol | None = None,
        solver_cfg: FlowSolverConfigProtocol | None = None,
    ):
        """Base class for flow matching models.


        Args:
            net (nnx.Module): Neural network to be trained.
            mu0 (Array): Initial mean.
            std0 (Array): Initial standard deviation.
            mu1 (Array): Final mean guess (e.g. data mean).
            std1 (Array): Final standard deviation guess (e.g. data std).
            interpolation_fn (Optional[Callable], optional): Interpolation function
                for the mean. Defaults to None.
            interpolation_std_fn (Optional[Callable], optional): Interpolation noise
                which can be none (i.e. no noise). Defaults to None.
            rngs (_type_, optional): _description_. Defaults to None.
        """
        self.rngs = rngs
        self.net: ModuleLike = net
        self.interp_schedule = (
            interp_schedule if interp_schedule is not None else LinearInterpolationSchedule()
        )
        if not isinstance(self.interp_schedule, InterpolationScheduleProtocol):
            raise TypeError("interp_schedule must implement InterpolationScheduleProtocol")
        self.preconditioning = (
            preconditioning if preconditioning is not None else GaussianFlowPreconditioning()
        )
        if not isinstance(self.preconditioning, FlowPreconditioningProtocol):
            raise TypeError("preconditioning must implement FlowPreconditioningProtocol")
        self.interpolation_fn = (
            interpolation_fn if interpolation_fn is not None else self.interp_schedule.interpolation_fn
        )

        if interpolation_std_fn is not None:
            self.interpolation_std_fn = interpolation_std_fn
        else:
            self.interpolation_std_fn = self.interp_schedule.interpolation_noise_fn

        self.mu0 = nnx.Variable(mu0)
        self.std0 = nnx.Variable(std0)
        self.mu1 = nnx.Variable(mu1)
        self.std1 = nnx.Variable(std1)

        default_train_cfg = LogitNormalFlowTrainingConfig(mu=0.7, scale=1.0)
        self.train_cfg = train_cfg or default_train_cfg
        if not isinstance(self.train_cfg, FlowTrainingConfigProtocol):
            raise TypeError("train_cfg must implement FlowTrainingConfigProtocol")
        self.solver_cfg = solver_cfg or LinearFlowSolverConfig()
        if not isinstance(self.solver_cfg, FlowSolverConfigProtocol):
            raise TypeError("solver_cfg must implement FlowSolverConfigProtocol")

        self._loss_kwargs: dict[str, object] = dict(loss_kwargs or {})

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
        # With preconditioning
        mu0 = self.mu0.value
        std0 = self.std0.value
        mu1 = self.mu1.value
        std1 = self.std1.value

        x_normed, approx_mut, approx_stdt = self.preconditioning.normalize(
            self.interp_schedule, t, x, mu0, mu1, std0, std1
        )
        residual_pred = self.net(t, x_normed, *args, **kwargs)
        residual_correction = jax.tree_util.tree_map(
            lambda r: approx_stdt * r, residual_pred
        )
        return self.preconditioning.decode_velocity(
            self.interp_schedule,
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
        loss_fn = build_flow_matching_loss_from_schedule(
            self,
            self.interp_schedule,
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
        times = self.train_cfg.sample_times(rng_times, (data_shape[0],) + (1,) * ndims)

        loss = loss_fn(times, x0, data, *args, **kwargs)
        return loss

    def solve_schedule(
        self,
        t_min: float = 0.0,
        t_max: float = 1.0,
        num_steps: int | None = None,
    ) -> Array:
        return self.solver_cfg.solve_schedule(t_min=t_min, t_max=t_max, num_steps=num_steps)


class MeanFlowMatcher(nnx.Module):
    def __init__(
        self,
        net: ModuleLike,
        interpolation_fn: Callable | None = None,
        interpolation_std_fn: Callable | None = None,
        interpolation_grad_fn: Callable | None = None,
        interpolation_noise_grad_fn: Callable | None = None,
        mu0: ArrayLike = 0,
        std0: ArrayLike = 1,
        mu1: ArrayLike = 0.0,
        std1: ArrayLike = 1.0,
        rngs: nnx.RngStream | None = None,
        loss_kwargs: Mapping[str, object] | None = None,
        interp_schedule: InterpolationScheduleProtocol | None = None,
        preconditioning: FlowPreconditioningProtocol | None = None,
        train_cfg: FlowPairTrainingConfigProtocol | None = None,
        solver_cfg: FlowSolverConfigProtocol | None = None,
    ):
        self.net: ModuleLike = net
        self.mu0 = nnx.Variable(mu0)
        self.std0 = nnx.Variable(std0)
        self.mu1 = nnx.Variable(mu1)
        self.std1 = nnx.Variable(std1)
        self.rngs = rngs
        self._loss_kwargs: dict[str, object] = dict(loss_kwargs or {})

        self.interp_schedule = (
            interp_schedule if interp_schedule is not None else LinearInterpolationSchedule()
        )
        if not isinstance(self.interp_schedule, InterpolationScheduleProtocol):
            raise TypeError("interp_schedule must implement InterpolationScheduleProtocol")

        self.preconditioning = (
            preconditioning if preconditioning is not None else GaussianFlowPreconditioning()
        )
        if not isinstance(self.preconditioning, FlowPreconditioningProtocol):
            raise TypeError("preconditioning must implement FlowPreconditioningProtocol")

        self.interpolation_fn = (
            interpolation_fn if interpolation_fn is not None else self.interp_schedule.interpolation_fn
        )
        self.interpolation_std_fn = (
            interpolation_std_fn if interpolation_std_fn is not None else self.interp_schedule.interpolation_noise_fn
        )
        self.interpolation_grad_fn = interpolation_grad_fn
        self.interpolation_noise_grad_fn = interpolation_noise_grad_fn

        self.train_cfg = train_cfg or SigmoidPairFlowTrainingConfig()
        if not isinstance(self.train_cfg, FlowPairTrainingConfigProtocol):
            raise TypeError("train_cfg must implement FlowPairTrainingConfigProtocol")
        self.solver_cfg = solver_cfg or LinearFlowSolverConfig()
        if not isinstance(self.solver_cfg, FlowSolverConfigProtocol):
            raise TypeError("solver_cfg must implement FlowSolverConfigProtocol")

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
        approx_mu_t = self.interp_schedule.path_mean(t, mu0, mu1)
        approx_std_t = jnp.maximum(
            self.interp_schedule.path_std(t, std0, std1), eps
        )

        x_normed = jax.tree_util.tree_map(lambda x: (x - approx_mu_t) / approx_std_t, x)
        std_t = approx_std_t
        std_r = jnp.maximum(self.interp_schedule.path_std(r, std0, std1), eps)
        a_t = self.interp_schedule.a_t(t)
        b_t = self.interp_schedule.b_t(t)
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
        loss_fn = build_mean_flow_matching_loss_from_schedule(
            self,
            self.interp_schedule,
            weight_fn=None,
            interpolation_grad_fn=self.interpolation_grad_fn,
            interpolation_noise_grad_fn=self.interpolation_noise_grad_fn,
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
    ):
        super().__init__(
            net,
            mu0=mu0,
            std0=std0,
            mu1=mu1,
            std1=std1,
            rngs=rngs,
            interpolation_fn=lambda t, x0, x1: (1 - t) * x0 + t * x1,
            loss_kwargs=loss_kwargs,
            train_cfg=train_cfg or LogitNormalFlowTrainingConfig(),
            solver_cfg=solver_cfg,
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
    ):
        interpolation_fn = lambda t, x0, x1: (1 - t) * x0 + t * x1
        super().__init__(
            net,
            interpolation_fn,
            None,
            None,
            None,
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

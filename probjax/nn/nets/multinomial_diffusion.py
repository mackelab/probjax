from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.loss_fn.multinomial_diffusion import (
    build_time_dependent_multinomial_diffusion_loss,
)
from probjax.nn.utils import module_accepts_rng
from probjax.nn.nets.config.multinomial_diffusion_configs import (
    CategoricalEDMPreconditioning,
    CategoricalPreconditioningProtocol,
    CategoricalScheduleProtocol,
    CategoricalTrainingConfigProtocol,
    ImportanceContinuousTimeTrainingConfig,
    MultinomialDiffusionSchedule,
    UniformContinuousTimeTrainingConfig,
    _one_hot,
    _require_float_time,
    _sample_categorical,
)
from probjax.nn.sharding import ShardingCfg, resolve_sharding_mesh
from probjax.utils.typing import Array, ArrayLike, ModuleLike, RngKey


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
        sharding_cfg: ShardingCfg | None = None,
        eps: float = 1e-12,
    ):
        if not isinstance(schedule, CategoricalScheduleProtocol):
            raise TypeError("schedule must implement CategoricalScheduleProtocol")

        self.net = net
        self._net_accepts_rng = module_accepts_rng(self.net)
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
        self._mesh = resolve_sharding_mesh(sharding_cfg)

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
        rng: jax.Array | None = None,
        **kwargs,
    ) -> Array:
        t_cont = self._to_continuous_time(t)
        t_embed = self.c_t(t_cont)
        x_embed = self.c_in(t_cont)[..., None] * x_features
        if self._net_accepts_rng:
            return self.net(t_embed, x_embed, *args, rng=rng, **kwargs)
        return self.net(t_embed, x_embed, *args, **kwargs)

    def __call__(
        self,
        t: ArrayLike,
        x_t: Array,
        *args,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> Array:
        x_features = self._net_input(x_t)
        return self._net_forward(t, x_features, *args, rng=rng, **kwargs)

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
            raise ValueError(
                "p_probs requires t_prev for continuous-time reverse transitions."
            )
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
        sharding_cfg: ShardingCfg | None = None,
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
            sharding_cfg=sharding_cfg,
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
        sharding_cfg: ShardingCfg | None = None,
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
            sharding_cfg=sharding_cfg,
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

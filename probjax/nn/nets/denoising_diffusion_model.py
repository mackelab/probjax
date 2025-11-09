from typing import Callable, Mapping, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss
from probjax.utils.odeint import odeint
from probjax.utils.sdeint import sdeint
from probjax.utils.typing import Array, ArrayLike, ModuleLike, PyTree, RngKey


class DiffusionDenoiser(nnx.Module):
    r"""This serves as a base class for diffusion denoising models.

    Specifically this implementation will minimize a denoising network
    with loss of form:
    $$
    \mathcal{L} = \mathbb{E}_{t,x,x_t} \left[w_t \left\| x -x_t \right\|^2 \right]
    $$
    where $x_t$ is the noisy version of $x$ at time $t$. Although equivalent in
    target this is somewhat different from using denoising score matching.

    """

    # Can be overwritten
    scale_fn: Callable[[ArrayLike], Array]
    std_fn: Callable[[ArrayLike], Array]

    def __init__(
        self,
        net: ModuleLike,
        std0: ArrayLike = 1.0,
        scale_fn: Callable[[ArrayLike], Array] | None = None,
        std_fn: Callable[[ArrayLike], Array] | None = None,
        last_layer: Callable[[Array], Array] | None = None,
        loss_type: str = "x0",
        loss_kwargs: Mapping[str, object] | None = None,
        rngs: nnx.RngStream | None = None,
    ) -> None:
        """
        Initialize the diffusion denoiser.

        Args:
            net: Neural network to train.
            std0: Initial standard deviation.
            scale_fn: Scale schedule function.
            std_fn: Std schedule function.
            last_layer: Optional last transformation layer.
            rngs: Random number streams.
            loss_type: Type of prediction target: "x0", "epsilon", or "v".
        """
        self.rngs = rngs
        self.net: ModuleLike = net
        if scale_fn is not None:
            self.scale_fn = scale_fn
        if std_fn is not None:
            self.std_fn = std_fn
        self.std0 = nnx.Variable(std0)
        self.last_layer = last_layer

        self._loss_kwargs: dict[str, object] = dict(loss_kwargs or {})
        self._loss_type = loss_type
        self._build_loss_fn(loss_type)

    def _build_loss_fn(
        self, loss_type: str, loss_kwargs: Mapping[str, object] | None = None
    ) -> Callable:
        if loss_kwargs is not None:
            self._loss_kwargs = dict(loss_kwargs)

        if loss_type == "x0":
            weight_fn = self.weight_fn
            pred_fn = self.denoise
        elif loss_type == "eps":
            weight_fn = self.weight_fn_eps
            pred_fn = self.epsilon
        elif loss_type == "v":
            weight_fn = self.weight_fn_v
            pred_fn = self.v
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        loss_fn = build_time_dependent_denoising_loss(
            pred_fn,
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
            weight_fn=weight_fn,
            prediction_target=loss_type,
            **self._loss_kwargs,
        )
        return loss_fn

    @property
    def loss_type(self) -> str:
        return self._loss_type

    @loss_type.setter
    def loss_type(
        self, value: str, loss_kwargs: Mapping[str, object] | None = None
    ) -> None:
        self._loss_type = value
        self._build_loss_fn(value, loss_kwargs)

    def c_in(self, t: ArrayLike) -> Array:
        """Compute input scaling factor."""
        return jnp.asarray(1.0)

    def c_out(self, t: ArrayLike) -> Array:
        """Compute output scaling factor."""
        return jnp.asarray(1.0)

    def c_t(self, t: ArrayLike) -> Array:
        """Compute noise embedding scaling factor."""
        return self.std_fn(t)

    def c_skip(self, t: ArrayLike) -> Array | None:
        """Compute skip connection scaling factor or None."""
        return None

    def weight_fn(self, t: ArrayLike) -> Array:
        """Compute weighting function for loss."""
        return jnp.asarray(1.0)

    def weight_fn_eps(self, t: ArrayLike) -> Array:
        """Compute weighting function for loss."""
        return self.weight_fn(t) * self.std_fn(t) ** 2

    def weight_fn_v(self, t: ArrayLike) -> Array:
        """Compute weighting function for loss."""
        alpha_t = self.scale_fn(t)
        sigma_t = self.std_fn(t)
        total_variance = jnp.sqrt(alpha_t**2 + sigma_t**2)
        a_t = alpha_t / total_variance
        s_t = sigma_t / total_variance

        a_t_weight = self.weight_fn_eps(t) * a_t**2
        s_t_weight = self.weight_fn(t) * s_t**2
        return 0.5 * (a_t_weight + s_t_weight)  # This normally comes down to just 1.0

    def __call__(
        self, t: ArrayLike, x_t: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]:
        """
        Forward pass of the model.

        Args:
            t: Time steps.
            x: Input data (typically x_t, the noisy data).

        Returns:
            Model output, conditioned on prediction_type.
            If "x0", it's the Karras-preconditioned prediction of x0.
            If "epsilon", it's the direct prediction of epsilon.
            If "v", it's the direct prediction of v.
        """
        noise_embed = self.c_t(t)
        x_embed = jax.tree_util.tree_map(lambda x: self.c_in(t) * x, x_t)

        out = self.net(noise_embed, x_embed, *args, **kwargs)

        if self.last_layer is not None:
            out = jax.tree_util.tree_map(self.last_layer, out)

        return out

    def denoise(
        self, t: ArrayLike, x_t: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]:
        """Predict denoised x0 from noisy x_t at time t."""
        model_output = self.__call__(t, x_t, *args, **kwargs)
        # Karras-preconditioned x0 prediction
        c_out = self.c_out(t)
        c_skip = self.c_skip(t)

        return jax.tree_util.tree_map(
            lambda x, o: c_skip * jnp.nan_to_num(x) + c_out * o,
            x_t,
            model_output,
        )

    def epsilon(
        self, t: ArrayLike, x_t: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]:
        """Predict noise epsilon from noisy x_t at time t."""
        sigma_t = self.std_fn(t)
        x0_pred = self.denoise(t, x_t, *args, **kwargs)

        return jax.tree_util.tree_map(
            lambda x, o: (jnp.nan_to_num(x) - o) / sigma_t,
            x_t,
            x0_pred,
        )

    def score(self, t: ArrayLike, x_t: PyTree[Array], *args, **kwargs) -> PyTree[Array]:
        """Compute score (nabla_x_t log p(x_t|x0)) from noisy x_t at time t.
        Defined as -epsilon_pred / sigma_t.
        """
        epsilon_pred = self.epsilon(t, x_t, *args, **kwargs)
        sigma_t = self.std_fn(t)
        return jax.tree_util.tree_map(
            lambda x: -jnp.nan_to_num(x) / sigma_t,
            epsilon_pred,
        )

    def v(self, t: ArrayLike, x_t: PyTree[Array], *args, **kwargs) -> PyTree[Array]:
        """Predict v (velocity or related quantity) from noisy x_t at time t.
        Here, v is defined as v_target = alpha_t * epsilon - sigma_t * x0.
        """
        # Get denoised prediction (x0)
        x0_pred = self.denoise(t, x_t, *args, **kwargs)

        # Get noise prediction (epsilon)
        # Compute from x0_pred and x_t
        epsilon_pred = jax.tree_util.tree_map(
            lambda x, x0: (jnp.nan_to_num(x) - x0) / self.std_fn(t), x_t, x0_pred
        )

        # Calculate v using the formula: alpha_t * epsilon - sigma_t * x0
        # alpha_t is the signal scaling factor related to scale_fn
        # and sigma_t is the noise standard deviation from std_fn
        alpha_t = self.scale_fn(t)
        sigma_t = self.std_fn(t)

        # Normalize by total variance to maintain consistency between different noise levels
        total_variance = jnp.sqrt(alpha_t**2 + sigma_t**2)
        normalized_alpha = alpha_t / total_variance
        normalized_sigma = sigma_t / total_variance

        # Apply the formula using tree_map for each operation
        v = jax.tree_util.tree_map(
            lambda eps, x0: normalized_alpha * eps - normalized_sigma * x0,
            epsilon_pred,
            x0_pred,
        )

        return v

    def marginal_std(self, t: ArrayLike) -> Array:
        """Compute marginal standard deviation."""
        return jnp.sqrt(
            self.scale_fn(t) ** 2 * (self.std_fn(t) ** 2 + self.std0.value**2)
        )

    def drift(self, t: ArrayLike, x: PyTree[Array], *args, **kwargs) -> PyTree[Array]:
        """Compute SDE drift term."""
        scale = self.scale_fn(t)
        scale_dt = jax.grad(lambda t: jnp.sum(self.scale_fn(t)))(t)
        return jax.tree_util.tree_map(
            lambda x_i: (scale_dt / scale) * x_i,
            x,
        )

    def diffusion(
        self, t: ArrayLike, x: PyTree[Array], *args, **kwargs
    ) -> PyTree[Array]:
        """Compute SDE diffusion term."""
        scale = self.scale_fn(t)
        std = self.std_fn(t)
        std_dt = jax.grad(lambda t: jnp.sum(self.std_fn(t)))(t)
        return jax.tree_util.tree_map(
            lambda x_i: scale * jnp.sqrt(2 * std_dt * std),
            x,
        )

    def noise_schedule(self, rng: RngKey, shape: Tuple[int, ...]) -> Array:
        """Compute noise levels for given shape and RNG."""
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
        """Compute diffusion denoising loss."""
        loss_fn = self._build_loss_fn(self._loss_type)
        rng_times, rng_loss = jax.random.split(rng, 2)
        ndims = data.ndim - 2
        times = self.noise_schedule(rng_times, (data.shape[0],) + (1,) * ndims)

        if "axis" not in kwargs:
            kwargs["axis"] = tuple(range(1, data.ndim))
        loss = loss_fn(times, data, *args, rng=rng_loss, **kwargs)
        return loss

    def solve_schedule(self, num_steps: int) -> Array:
        return jnp.linspace(0.0, 1.0, num_steps)[::-1]

    def build_ode_drift(self, *args, **kwargs) -> Callable:
        def ode_drift(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            f = self.drift(t, x, *args, **kwargs)
            g = self.diffusion(t, x, *args, **kwargs)
            score = self.score(t, x, *args, **kwargs)
            return jax.tree_util.tree_map(
                lambda fi, gi, si: fi - 0.5 * gi**2 * si,
                f,
                g,
                score,
            )

        return ode_drift

    def build_sde_drift(self, *args, **kwargs) -> Callable:
        def sde_drift(t: ArrayLike, x: PyTree[Array]) -> PyTree[Array]:
            f = self.drift(t, x, *args, **kwargs)
            g = self.diffusion(t, x, *args, **kwargs)
            score = self.score(t, x, *args, **kwargs)
            return jax.tree_util.tree_map(
                lambda fi, gi, si: fi - gi**2 * si,
                f,
                g,
                score,
            )

        return sde_drift

    def sample_ode(
        self,
        eps: PyTree[Array],
        num_steps: int,
        collect_trace: bool = False,
        method: str = "heun",
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        """Sample from the diffusion model using ODE solver."""
        ts = self.solve_schedule(num_steps)
        drift = self.build_ode_drift(*args, **kwargs)
        x_T = odeint(
            drift,
            eps,
            ts,
            collect_trace=collect_trace,
            method=method,
        )
        return x_T

    def sample_sde(
        self,
        rng: RngKey,
        eps: PyTree[Array],
        num_steps: int,
        collect_trace: bool = False,
        method: str = "euler_maruyama",
        *args,
        **kwargs,
    ) -> PyTree[Array]:
        """Sample from the diffusion model using SDE solver."""
        ts = self.solve_schedule(num_steps)
        drift = self.build_sde_drift(*args, **kwargs)
        diffusion = self.diffusion
        x_T = sdeint(
            rng,
            drift,
            diffusion,
            eps,
            ts,
            collect_trace=collect_trace,
            method=method,
        )
        return x_T


class EDM(DiffusionDenoiser):
    scale_fn = lambda _, t: jnp.array([1.0])
    std_fn = lambda _, t: jnp.atleast_1d(t)
    drift: lambda _, t, x: jnp.array([0.0])
    diffusion: lambda _, t, x: jnp.atleast_1d(jnp.sqrt(2 * t))
    lognoise_mean: float = -1.2
    lognoise_scale: float = 1.2
    min_noise: float = 0.0001
    max_noise: float = 80.0

    def c_in(self, t: ArrayLike) -> Array:
        total_std = jnp.sqrt(self.std0.value**2 + self.std_fn(t) ** 2)
        return 1.0 / total_std

    def c_out(self, t: ArrayLike) -> Array:
        std = self.std_fn(t)
        return std * self.std0.value / jnp.sqrt(self.std0.value**2 + std**2)

    def c_t(self, t: ArrayLike) -> Array:
        return 0.25 * jnp.log(self.std_fn(t))

    def c_skip(self, t: ArrayLike) -> Array:
        return self.std0.value / (self.std0.value**2 + self.std_fn(t) ** 2)

    def weight_fn(self, t: ArrayLike) -> Array:
        out_weight = self.c_out(t)
        return 1.0 / out_weight**2

    def noise_schedule(
        self,
        rng: RngKey,
        shape: Tuple[int, ...],
        *,
        min_noise: float | None = None,
        max_noise: float | None = None,
        lognoise_mean: float | None = None,
        lognoise_scale: float | None = None,
    ) -> Array:
        min_noise = self.min_noise if min_noise is None else min_noise
        max_noise = self.max_noise if max_noise is None else max_noise
        lognoise_mean = self.lognoise_mean if lognoise_mean is None else lognoise_mean
        lognoise_scale = (
            self.lognoise_scale if lognoise_scale is None else lognoise_scale
        )
        logt = (
            jax.random.normal(rng, shape=shape + (1,)) * lognoise_scale + lognoise_mean
        )
        return jnp.clip(jnp.exp(logt) + min_noise, min_noise, max_noise)

    def solve_schedule(
        self,
        num_steps: int | None = None,
        *,
        rho: int | None = None,
        min_noise: float | None = None,
        max_noise: float | None = None,
    ) -> Array:
        num_steps = 100 if num_steps is None else num_steps
        rho = 7 if rho is None else rho
        min_noise = self.min_noise if min_noise is None else min_noise
        max_noise = self.max_noise if max_noise is None else max_noise
        ns = jnp.arange(0, num_steps, dtype=jnp.float32)
        term1 = max_noise ** (1 / rho)
        length = (min_noise ** (1 / rho) - term1) * ns / (num_steps - 1)
        return (term1 + length) ** rho


class VE(EDM):
    """Variance Exploding (VE) SDE variant."""

    scale_fn = lambda _, t: jnp.array([1.0])
    std_fn = lambda _, t: jnp.atleast_1d(jnp.sqrt(t))
    drift: lambda _, t, x: jnp.array([0.0])
    diffusion: lambda _, t, x: jnp.atleast_1d((t**-0.5 * 2 * t**0.5) ** 0.5)

    def noise_schedule(
        self,
        rng: RngKey,
        shape: Tuple[int, ...],
        *,
        min_noise: float | None = None,
        max_noise: float | None = None,
        lognoise_mean: float | None = None,
        lognoise_scale: float | None = None,
    ) -> Array:
        """Compute noise schedule for VE."""
        min_noise = self.min_noise if min_noise is None else min_noise
        max_noise = self.max_noise if max_noise is None else max_noise
        lognoise_mean = self.lognoise_mean if lognoise_mean is None else lognoise_mean
        lognoise_scale = (
            self.lognoise_scale if lognoise_scale is None else lognoise_scale
        )
        logt = (
            jax.random.normal(rng, shape=shape + (1,)) * lognoise_scale + lognoise_mean
        )
        return jnp.clip(jnp.exp(logt) + min_noise, min_noise, max_noise)

    def solve_schedule(
        self,
        num_steps: int | None = None,
        rho: int | None = None,
        *,
        min_noise: float | None = None,
        max_noise: float | None = None,
    ) -> Array:
        """Compute solving schedule for VE."""
        num_steps = 100 if num_steps is None else num_steps
        rho = 7 if rho is None else rho
        min_noise = self.min_noise if min_noise is None else min_noise
        max_noise = self.max_noise if max_noise is None else max_noise
        ns = jnp.arange(0, num_steps, dtype=jnp.float32)
        term1 = max_noise ** (1 / rho)
        length = (min_noise ** (1 / rho) - term1) * ns / (num_steps - 1)
        return (term1 + length) ** rho


class VP(EDM):
    """Variance Preserving (VP) SDE variant."""

    min_noise: float = 0.002
    max_noise: float = 1.0
    lognoise_mean: float = -0.5
    lognoise_scale: float = 1.2

    def __init__(
        self,
        net: ModuleLike,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        std0: ArrayLike = 1.0,
        rngs: nnx.RngStream | None = None,
        loss_type: str = "x0",
    ) -> None:
        """VP variant with beta range for SDE."""
        super().__init__(net, std0=std0, rngs=rngs, loss_type=loss_type)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def std_fn(self, t: ArrayLike) -> Array:
        """Compute standard deviation for VP SDE."""
        beta_min = self.beta_min
        beta_max = self.beta_max
        dbeta = beta_max - beta_min
        integral = 0.5 * dbeta * t**2 + beta_min * t
        term = jnp.exp(integral) - 1
        return jnp.atleast_1d(jnp.sqrt(term))

    def scale_fn(self, t: ArrayLike) -> Array:
        """Compute scale function for VP SDE."""
        beta_min = self.beta_min
        beta_max = self.beta_max
        dbeta = beta_max - beta_min
        integral = 0.5 * dbeta * t**2 + beta_min * t
        term = jnp.exp(integral)
        return 1 / jnp.atleast_1d(jnp.sqrt(term))

    def noise_schedule(
        self,
        rng: RngKey,
        shape: Tuple[int, ...],
        *,
        min_noise: float | None = None,
        max_noise: float | None = None,
        lognoise_mean: float | None = None,
        lognoise_scale: float | None = None,
    ) -> Array:
        """Compute noise schedule for VP."""
        min_noise = self.min_noise if min_noise is None else min_noise
        max_noise = self.max_noise if max_noise is None else max_noise
        lognoise_mean = self.lognoise_mean if lognoise_mean is None else lognoise_mean
        lognoise_scale = (
            self.lognoise_scale if lognoise_scale is None else lognoise_scale
        )
        logt = (
            jax.random.normal(rng, shape=shape + (1,)) * lognoise_scale + lognoise_mean
        )
        return jax.nn.sigmoid(logt) * (max_noise - min_noise) + min_noise

    def solve_schedule(
        self,
        num_steps: int | None = None,
        *,
        min_noise: float | None = None,
    ) -> Array:
        """Compute solving schedule for VP."""
        num_steps = 100 if num_steps is None else num_steps
        min_noise = self.min_noise if min_noise is None else min_noise
        ts = jnp.linspace(min_noise, 1.0 + 1 / num_steps, num_steps)[::-1]
        return ts

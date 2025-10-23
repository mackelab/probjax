from typing import Callable, Optional, Tuple, Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.random import PRNGKey
from jax.typing import ArrayLike
from jaxtyping import PyTree

from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss


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
        net: nnx.Module,
        std0: ArrayLike = 1.0,
        scale_fn: Optional[Callable[[ArrayLike], Array]] = None,
        std_fn: Optional[Callable[[ArrayLike], Array]] = None,
        last_layer: Optional[Callable[[ArrayLike], ArrayLike]] = None,
        loss_type: str = "x0",
        loss_kwargs=None,
        rngs: nnx.RngStream = None,
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
        self.net = net
        if scale_fn is not None:
            self.scale_fn = scale_fn
        if std_fn is not None:
            self.std_fn = std_fn
        self.std0 = nnx.Variable(std0)
        self.last_layer = last_layer

        self._loss_type = loss_type

    def _build_loss_fn(self, loss_type: str, loss_kwargs=None):
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

        self._loss_fn = build_time_dependent_denoising_loss(
            pred_fn,
            scale_fn=self.scale_fn,
            std_fn=self.std_fn,
            weight_fn=weight_fn,
            prediction_target=loss_type,
            **(loss_kwargs or {}),
        )

    @property
    def loss_type(self):
        return self._loss_type

    @loss_type.setter
    def loss_type(self, value: str, loss_kwargs=None):
        self._loss_type = value
        self._build_loss_fn(value, loss_kwargs)

    def c_in(self, t: ArrayLike) -> float:
        """Compute input scaling factor."""
        return 1.0

    def c_out(self, t: ArrayLike) -> float:
        """Compute output scaling factor."""
        return 1.0

    def c_t(self, t: ArrayLike) -> Array:
        """Compute noise embedding scaling factor."""
        return self.std_fn(t)

    def c_skip(self, t: ArrayLike) -> Optional[float]:
        """Compute skip connection scaling factor or None."""
        return None

    def weight_fn(self, t: ArrayLike) -> float:
        """Compute weighting function for loss."""
        return 1.0

    def weight_fn_eps(self, t: ArrayLike) -> float:
        """Compute weighting function for loss."""
        return self.weight_fn(t) * self.std_fn(t) ** 2

    def weight_fn_v(self, t: ArrayLike) -> float:
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
        self, t: ArrayLike, x_t: PyTree[ArrayLike], *args, **kwargs
    ) -> PyTree[ArrayLike]:
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
        self, t: ArrayLike, x_t: PyTree[ArrayLike], *args, **kwargs
    ) -> PyTree[ArrayLike]:
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
        self, t: ArrayLike, x_t: PyTree[ArrayLike], *args, **kwargs
    ) -> PyTree[ArrayLike]:
        """Predict noise epsilon from noisy x_t at time t."""
        sigma_t = self.std_fn(t)
        x0_pred = self.denoise(t, x_t, *args, **kwargs)

        return jax.tree_util.tree_map(
            lambda x, o: (jnp.nan_to_num(x) - o) / sigma_t,
            x_t,
            x0_pred,
        )

    def score(
        self, t: ArrayLike, x_t: PyTree[ArrayLike], *args, **kwargs
    ) -> PyTree[ArrayLike]:
        """Compute score (nabla_x_t log p(x_t|x0)) from noisy x_t at time t.
        Defined as -epsilon_pred / sigma_t.
        """
        epsilon_pred = self.epsilon(t, x_t, *args, **kwargs)
        sigma_t = self.std_fn(t)
        return jax.tree_util.tree_map(
            lambda x: -jnp.nan_to_num(x) / sigma_t,
            epsilon_pred,
        )

    def v(
        self, t: ArrayLike, x_t: PyTree[ArrayLike], *args, **kwargs
    ) -> PyTree[ArrayLike]:
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

    def marginal_std(self, t: ArrayLike) -> ArrayLike:
        """Compute marginal standard deviation."""
        return jnp.sqrt(
            self.scale_fn(t) ** 2 * (self.std_fn(t) ** 2 + self.std0.value**2)
        )

    def drift(
        self, t: ArrayLike, x: PyTree[ArrayLike], *args, **kwargs
    ) -> PyTree[ArrayLike]:
        """Compute SDE drift term."""
        scale = self.scale_fn(t)
        scale_dt = jax.grad(lambda t: jnp.sum(self.scale_fn(t)))(t)
        return jax.tree_util.tree_map(
            lambda x_i: (scale_dt / scale) * x_i,
            x,
        )

    def diffusion(
        self, t: ArrayLike, x: PyTree[ArrayLike], *args, **kwargs
    ) -> PyTree[ArrayLike]:
        """Compute SDE diffusion term."""
        scale = self.scale_fn(t)
        std = self.std_fn(t)
        std_dt = jax.grad(lambda t: jnp.sum(self.std_fn(t)))(t)
        return jax.tree_util.tree_map(
            lambda x_i: scale * jnp.sqrt(2 * std_dt * std),
            x,
        )

    def noise_schedule(self, rng: PRNGKey, shape: Tuple[int, ...]) -> Array:
        """Compute noise levels for given shape and RNG."""
        raise NotImplementedError(
            "Implemented only for specific implementation of this base class"
        )

    def loss(
        self,
        rng: PRNGKey,
        data: ArrayLike,
        *args,
        **kwargs,
    ) -> Array:
        """Compute diffusion denoising loss."""
        self._build_loss_fn(self._loss_type)
        rng_times, rng_loss = jax.random.split(rng, 2)
        ndims = data.ndim - 2
        times = self.noise_schedule(rng_times, (data.shape[0],) + (1,) * ndims)

        if 'axis' not in kwargs:
            axis = tuple(range(1, data.ndim))
            kwargs['axis'] = axis
        loss = self._loss_fn(times, data, *args, rng=rng_loss,**kwargs)
        return loss


class EDM(DiffusionDenoiser):
    scale_fn = lambda _, t: jnp.array([1.0])
    std_fn = lambda _, t: jnp.atleast_1d(t)
    drift: lambda _, t, x: jnp.array([0.0])
    diffusion: lambda _, t, x: jnp.atleast_1d(jnp.sqrt(2 * t))
    lognoise_mean: float = -1.2
    lognoise_scale: float = 1.2
    min_noise: float = 0.0001
    max_noise: float = 80.0

    def c_in(self, t: ArrayLike) -> float:
        total_std = jnp.sqrt(self.std0.value**2 + self.std_fn(t) ** 2)
        return 1.0 / total_std

    def c_out(self, t: ArrayLike) -> float:
        std = self.std_fn(t)
        return std * self.std0.value / jnp.sqrt(self.std0.value**2 + std**2)

    def c_t(self, t: ArrayLike) -> Array:
        return 0.25 * jnp.log(self.std_fn(t))

    def c_skip(self, t: ArrayLike) -> Optional[float]:
        return self.std0.value / (self.std0.value**2 + self.std_fn(t) ** 2)

    def weight_fn(self, t: ArrayLike) -> float:
        out_weight = self.c_out(t)
        return 1.0 / out_weight**2

    def noise_schedule(self, rng: PRNGKey, shape: Tuple[int, ...]) -> Array:
        logt = (
            jax.random.normal(rng, shape=shape + (1,)) * self.lognoise_scale
            + self.lognoise_mean
        )
        return jnp.clip(jnp.exp(logt) + self.min_noise, self.min_noise, self.max_noise)

    def solve_schedule(self, num_steps: int = 100, rho: int = 7) -> Array:
        ns = jnp.arange(0, num_steps, dtype=jnp.float32)
        term1 = self.max_noise ** (1 / rho)
        length = (self.min_noise ** (1 / rho) - term1) * ns / (num_steps - 1)
        return (term1 + length) ** rho


class VE(EDM):
    """Variance Exploding (VE) SDE variant."""

    scale_fn = lambda _, t: jnp.array([1.0])
    std_fn = lambda _, t: jnp.atleast_1d(jnp.sqrt(t))
    drift: lambda _, t, x: jnp.array([0.0])
    diffusion: lambda _, t, x: jnp.atleast_1d((t**-0.5 * 2 * t**0.5) ** 0.5)

    def noise_schedule(self, rng: PRNGKey, shape: Tuple[int, ...]) -> Array:
        """Compute noise schedule for VE."""
        logt = (
            jax.random.normal(rng, shape=shape + (1,)) * self.lognoise_scale
            + self.lognoise_mean
        )
        return jnp.clip(jnp.exp(logt) + self.min_noise, self.min_noise, self.max_noise)

    def solve_schedule(self, num_steps: int = 100, rho: int = 7) -> Array:
        """Compute solving schedule for VE."""
        ns = jnp.arange(0, num_steps, dtype=jnp.float32)
        term1 = self.max_noise ** (1 / rho)
        length = (self.min_noise ** (1 / rho) - term1) * ns / (num_steps - 1)
        return (term1 + length) ** rho


class VP(EDM):
    """Variance Preserving (VP) SDE variant."""

    min_noise: float = 0.002
    max_noise: float = 1.0
    lognoise_mean: float = -0.5
    lognoise_scale: float = 1.2

    def __init__(
        self,
        net: nnx.Module,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        std0: ArrayLike = 1.0,
        rngs: nnx.RngStream = None,
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

    def noise_schedule(self, rng: PRNGKey, shape: Tuple[int, ...]) -> Array:
        """Compute noise schedule for VP."""
        logt = (
            jax.random.normal(rng, shape=shape + (1,)) * self.lognoise_scale
            + self.lognoise_mean
        )
        return jax.nn.sigmoid(logt) * (self.max_noise - self.min_noise) + self.min_noise

    def solve_schedule(self, num_steps: int = 100, rho: int = 7) -> Array:
        """Compute solving schedule for VP."""
        ts = jnp.linspace(self.min_noise, 1.0 + 1 / num_steps, num_steps)[::-1]
        return ts

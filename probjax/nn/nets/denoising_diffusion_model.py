from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.random import PRNGKey
from jax.typing import ArrayLike

from probjax.nn.loss_fn.denoising import build_time_dependent_denoising_loss


class DiffusionDenoiser(nnx.Module, experimental_pytree=True):
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
        loss_kwargs: Optional[dict] = None,
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
            loss_kwargs: Additional loss function arguments.
            rngs: Random number streams.
        """
        self.rngs = rngs
        self.net = net
        if scale_fn is not None:
            self.scale_fn = scale_fn
        if std_fn is not None:
            self.std_fn = std_fn
        self.std0 = nnx.Variable(std0)
        self.last_layer = last_layer

        self._loss = build_time_dependent_denoising_loss(
            self,
            mean_fn=lambda t, x: x,
            std_fn=lambda t, x: self.std_fn(t),
            weight_fn=self.weight_fn,
            **(loss_kwargs or {}),
        )

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

    def __call__(self, t: ArrayLike, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """
        Forward pass of the model.

        Args:
            t: Time steps.
            x: Input data.

        Returns:
            Processed output.
        """
        # With preconditioning
        noise_embed = self.c_t(t)
        x_normed = jax.tree_util.tree_map(lambda x: x * self.c_in(t), x)

        x_pred = self.net(noise_embed, x_normed, *args, **kwargs)

        scale_out = self.c_out(t)
        scale_skip = self.c_skip(t)

        out = jax.tree_util.tree_map(lambda x: x * scale_out, x_pred)
        if scale_skip is not None:
            out = jax.tree_util.tree_map(lambda x, o: x * scale_skip + o, x, out)

        if self.last_layer:
            out = jax.tree_util.tree_map(self.last_layer, out)

        return out

    def score(self, t: ArrayLike, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Compute Tweedie-based score."""
        x = x / self.scale_fn(t)
        mean = self.__call__(t, x, *args, **kwargs)

        # Score by Tweedie's formula
        # mean = x + std**2 * score
        # score = (mean - x) / std**2
        std = self.std_fn(t)
        score = (mean - x) / (std**2 * self.scale_fn(t))

        return score

    def marginal_std(self, t: ArrayLike) -> ArrayLike:
        """Compute marginal standard deviation."""
        return jnp.sqrt(
            self.scale_fn(t) ** 2 * (self.std_fn(t) ** 2 + self.std0.value**2)
        )

    def drift(self, t: ArrayLike, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Compute SDE drift term."""
        scale = self.scale_fn(t)
        scale_dt = jax.grad(lambda t: jnp.sum(self.scale_fn(t)))(t)
        return (scale_dt / scale) * x

    def diffusion(self, t: ArrayLike, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        """Compute SDE diffusion term."""
        scale = self.scale_fn(t)
        std = self.std_fn(t)
        std_dt = jax.grad(lambda t: jnp.sum(self.std_fn(t)))(t)
        return scale * jnp.sqrt(2 * std_dt * std)

    def noise_schedule(self, rng: PRNGKey, shape: Tuple[int, ...]) -> Array:
        """Compute noise levels for given shape and RNG."""
        raise NotImplementedError(
            "Implemented only for specific implementation of this base class"
        )

    def loss(
        self,
        params: Any,
        rng: PRNGKey,
        data: ArrayLike,
        *args,
        **kwargs,
    ) -> Array:
        """Compute diffusion denoising loss."""
        rng_times, rng_loss = jax.random.split(rng, 2)
        ndims = data.ndim - 2
        times = self.noise_schedule(rng_times, (data.shape[0],) + (1,) * ndims)

        axis = tuple(range(1, data.ndim))
        loss = self._loss(params, times, data, *args, rng=rng_loss, axis=axis, **kwargs)
        return loss


class EDM(DiffusionDenoiser):
    scale_fn = lambda _, t: jnp.array([1.0])
    std_fn = lambda _, t: jnp.atleast_1d(t)
    drift: lambda _, t, x: jnp.array([0.0])
    diffusion: lambda _, t, x: jnp.atleast_1d(jnp.sqrt(2 * t))
    lognoise_mean: float = -1.2
    lognoise_scale: float = 1.2
    min_noise: float = 0.0002
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
        beta_max: float = 20.0,
        std0: ArrayLike = 1.0,
        rngs: nnx.RngStream = None,
    ) -> None:
        """VP variant with beta range for SDE."""
        super().__init__(net, std0=std0, rngs=rngs)
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

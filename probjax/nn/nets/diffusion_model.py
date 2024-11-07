from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from flax import nnx


class DiffusionDenoiser(nnx.Module, experimental_pytree=True):
    # Can be overwritten
    scale_fn: Callable
    std_fn: Callable
    drift: Callable
    diffusion: Callable

    def __init__(
        self,
        net: nnx.Module,
        std0: ArrayLike = 1.0,
        scale_fn: Optional[Callable] = None,
        std_fn: Optional[Callable] = None,
        rngs=None,
    ):
        self.net = net
        if scale_fn is not None:
            self.scale_fn = scale_fn
        if std_fn is not None:
            self.std_fn = std_fn
        self.std0 = nnx.Variable(std0)

    def c_mu(self, t):
        return 1.0

    def c_in(self, t):
        return 1.0

    def c_out(self, t):
        return 1.0

    def c_t(self, t):
        return self.std_fn(t)

    def c_skip(self, t):
        return None

    def weight_fn(self, t):
        return 1.0

    def __call__(self, t, x, *args, **kwargs):
        # With preconditioning
        noise_embed = self.c_t(t)
        x_normed = jax.tree_util.tree_map(lambda x: x * self.c_in(t), x)

        x_pred = self.net(noise_embed, x_normed, *args, **kwargs)

        scale_out = self.c_out(t)
        scale_skip = self.c_skip(t)

        out = jax.tree_util.tree_map(lambda x: x * scale_out, x_pred)
        if scale_skip is not None:
            out = jax.tree_util.tree_map(lambda x, o: x * scale_skip + o, x, out)
        return out

    def score(self, t, x, *args, **kwargs):
        x = x / self.scale_fn(t)
        mean = self.__call__(t, x, *args, **kwargs)

        # Score by Tweedie's formula
        # mean = x + std**2 * score
        # score = (mean - x) / std**2
        std = self.std_fn(t)
        score = (mean - x) / (std**2 * self.scale_fn(t))

        return score

    def marginal_std(self, t):
        return jnp.sqrt(
            self.scale_fn(t) ** 2 * (self.std_fn(t) ** 2 + self.std0.value**2)
        )

    def drift(self, t, x, *args, **kwargs):
        scale = self.scale_fn(t)
        scale_dt = jax.grad(lambda t: jnp.sum(self.scale_fn(t)))(t)
        return (scale_dt / scale) * x

    def diffusion(self, t, x, *args, **kwargs):
        scale = self.scale_fn(t)
        std = self.std_fn(t)
        std_dt = jax.grad(lambda t: jnp.sum(self.std_fn(t)))(t)
        return scale * jnp.sqrt(2 * std_dt * std)


class EDM(DiffusionDenoiser):
    scale_fn = lambda _, t: jnp.array([1.0])
    std_fn = lambda _, t: jnp.atleast_1d(t)
    drift: lambda _, t, x: jnp.array([0.0])
    diffusion: lambda _, t, x: jnp.atleast_1d(jnp.sqrt(2 * t))
    lognoise_mean: float = 1.2
    lognoise_scale: float = 1.2
    min_noise: float = 0.0002

    def __init__(
        self,
        net: nnx.Module,
        std0: ArrayLike = 1.0,
        rngs=None,
    ):
        super().__init__(net, std0=std0, rngs=rngs)

    def c_in(self, t):
        total_std = jnp.sqrt(self.std0.value**2 + self.std_fn(t) ** 2)
        return 1.0 / total_std

    def c_out(self, t):
        std = self.std_fn(t)
        return std * self.std0.value / jnp.sqrt(self.std0.value**2 + std**2)

    def c_t(self, t):
        return 0.25 * jnp.log(self.std_fn(t))

    def c_skip(self, t):
        return self.std0.value / (self.std0.value**2 + self.std_fn(t) ** 2)

    def weight_fn(self, t):
        out_weight = self.c_out(t)
        return jnp.sum(1.0 / out_weight**2, axis=-1)

    def noise_schedule(self, rng, shape):
        logt = (
            jax.random.normal(rng, shape=shape) * self.lognoise_scale
            + self.lognoise_mean
        )
        return jnp.exp(logt) + self.min_noise


class VE(EDM):
    scale_fn = lambda _, t: jnp.array([1.0])
    std_fn = lambda _, t: jnp.atleast_1d(jnp.sqrt(t))
    drift: lambda _, t, x: jnp.array([0.0])
    diffusion: lambda _, t, x: jnp.atleast_1d((t**-0.5 * 2 * t**0.5) ** 0.5)


class VP(EDM):
    def __init__(
        self,
        net: nnx.Module,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        std0: ArrayLike = 1.0,
        rngs=None,
    ):
        super().__init__(net, std0=std0, rngs=rngs)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def std_fn(self, t):
        beta_min = self.beta_min
        beta_max = self.beta_max
        dbeta = beta_max - beta_min
        integral = 0.5 * dbeta * t**2 + beta_min * t
        term = jnp.exp(integral) - 1
        return jnp.atleast_1d(jnp.sqrt(term))

    def scale_fn(self, t):
        beta_min = self.beta_min
        beta_max = self.beta_max
        dbeta = beta_max - beta_min
        integral = 0.5 * dbeta * t**2 + beta_min * t
        term = jnp.exp(integral)
        return 1 / jnp.atleast_1d(jnp.sqrt(term))

    def noise_schedule(self, rng, shape):
        logt = (
            jax.random.normal(rng, shape=shape) * self.lognoise_scale
            + self.lognoise_mean
        )
        return jax.nn.sigmoid(logt) + self.min_noise

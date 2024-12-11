from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from probjax.nn.loss_fn.flow_matching import build_flow_matching_loss


class FlowMatcher(nnx.Module, experimental_pytree=True):
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
        net: nnx.Module,
        mu0: ArrayLike = 0.0,
        std0: ArrayLike = 1.0,
        mu1: ArrayLike = 0.0,
        std1: ArrayLike = 1.0,
        interpolation_fn: Optional[Callable] = None,
        interpolation_std_fn: Optional[Callable] = None,
        rngs=None,
    ):
        """Base class for flow matching models.


        Args:
            net (nnx.Module): Neural network to be trained.
            mu0 (Array): Initial mean.
            std0 (Array): Initial standard deviation.
            mu1 (Array): Final mean guess (e.g. data mean).
            std1 (Array): Final standard deviation guess (e.g. data std).
            mean_fn (Optional[Callable], optional): Interpolation function
                for the mean. Defaults to None.
            std_fn (Optional[Callable], optional): Interpolation noise
                which can be none (i.e. no noise). Defaults to None.
            rngs (_type_, optional): _description_. Defaults to None.
        """
        self.rngs = rngs
        self.net = net
        if interpolation_fn is not None:
            self.interpolation_fn = interpolation_fn
        if interpolation_std_fn is not None:
            self.interpolation_std_fn = interpolation_std_fn

        self.mu0 = nnx.Variable(mu0)
        self.std0 = nnx.Variable(std0)
        self.mu1 = nnx.Variable(mu1)
        self.std1 = nnx.Variable(std1)

    def __call__(self, t, x, *args, **kwargs):
        """Forward pass of the model - denosing x at time t."""
        # With preconditioning
        mu0 = self.mu0.value
        std0 = self.std0.value
        mu1 = self.mu1.value
        std1 = self.std1.value

        mut = self.interpolation_fn(mu0, mu1, t)
        stdt = jnp.sqrt(t**2 * std1**2 + (1 - t) ** 2 * std0**2)

        x_normed = (x - mut) / stdt

        pred_mu1 = self.net(t, x_normed, *args, **kwargs)
        pred_mu_t = self.interpolation_fn(mu0, pred_mu1, t)

        scale = ((1 - t) * std0**2 - t * std1**2) / (
            (1 - t) ** 2 * std0**2 + t**2 * std1**2
        )
        return (mu0 - mu1) + scale * (x - pred_mu_t)

    def score(self, t, x, *args, **kwargs):
        """Score function for the model."""
        raise NotImplementedError(
            "Implemented only for specific implementation of this base class"
        )

    def denoise(self, t, x):
        """Denoise the input x at time t."""
        raise NotImplementedError(
            "Implemented only for specific implementation of this base class"
        )


class RectifiedFlow(FlowMatcher):
    std_fn = None

    def __init__(self, net, mu0=0, std0=1, mu1=0.0, std1=1.0, rngs=None):
        mean_fn = lambda x0, x1, t: (1 - t) * x0 + t * x1

        super().__init__(
            net,
            mu0=mu0,
            std0=std0,
            mu1=mu1,
            std1=std1,
            interpolation_fn=mean_fn,
            rngs=rngs,
        )

        self._loss = build_flow_matching_loss(
            self,
            interpolation_fn=mean_fn,
            interpolation_noise_fn=None,
            weight_fn=None,
        )

    def denoise(self, t, x):
        # x0 is noise
        # x1 is data
        # xt = (1 - t) * x0 + t * x1
        # x0 = (xt - t * x1) / (1 - t)
        # E[x1-x0|xt] = E[x1 - (xt - t * x1) / (1 - t)|xt]
        # = (xt - E[x1|xt])/(1 - t)
        # So we can recover the denoiser by
        # E[x1|xt] = xt - (1-t)*E[x1-x0|xt]
        v = self.__call__(t, x)
        return x - (1 - t) * v

    def score(self, t, x, max_t=1 - 1e-3):
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
        return (-t * v + mu0 - x) / ((1 - t) * std0**2)

    def noise_schedule(self, rng, shape):
        return jax.nn.sigmoid(jax.random.normal(rng, shape=shape + (1,)) + 0.5)

    def solve_schedule(self, num_steps=50):
        return jnp.linspace(0, 1, num_steps)

    def loss(self, params, rng, data, *args, **kwargs):
        rng_source, rng_times = jax.random.split(rng, 2)
        x0 = jax.random.normal(rng_source, shape=data.shape)
        times = self.noise_schedule(rng_times, (data.shape[0],))
        loss = self._loss(params, times, x0, data, *args, **kwargs)
        return loss

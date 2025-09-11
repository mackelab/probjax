from datetime import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.tree_util
from flax import nnx
from jax.typing import ArrayLike
from jaxtyping import PyTree

from probjax.nn.loss_fn.flow_matching import (
    build_flow_matching_loss,
    build_mean_flow_matching_loss,
)


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
        net: nnx.Module,
        mu0: ArrayLike = 0.0,
        std0: ArrayLike = 1.0,
        mu1: ArrayLike = 0.0,
        std1: ArrayLike = 1.0,
        interpolation_fn: Optional[Callable] = None,
        interpolation_std_fn: Optional[Callable] = None,
        loss_kwargs: Optional[dict] = None,
        rngs=None,
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
        self.net = net
        if interpolation_fn is not None:
            self.interpolation_fn = interpolation_fn

        if interpolation_std_fn is not None:
            self.interpolation_std_fn = interpolation_std_fn
        else:
            self.interpolation_std_fn = None

        self.mu0 = nnx.Variable(mu0)
        self.std0 = nnx.Variable(std0)
        self.mu1 = nnx.Variable(mu1)
        self.std1 = nnx.Variable(std1)

        if loss_kwargs is None:
            loss_kwargs = {}

        self._loss = build_flow_matching_loss(
            self,
            interpolation_fn=self.interpolation_fn,
            interpolation_noise_fn=self.interpolation_std_fn,
            weight_fn=None,
            **loss_kwargs,
        )

    def __call__(self, t, x: ArrayLike, *args, **kwargs) -> ArrayLike:
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

        # Gaussian closed-form preconditioning
        approx_mut = self.interpolation_fn(mu0, mu1, t)
        approx_stdt = jnp.sqrt(t**2 * std1**2 + (1 - t) ** 2 * std0**2)

        x_normed = jax.tree_util.tree_map(lambda x: (x - approx_mut) / approx_stdt, x)
        scale = ((t * std1**2) - (1 - t) * std0**2) / (
            (1 - t) ** 2 * std0**2 + t**2 * std1**2
        )
        residual_correction = approx_stdt * self.net(t, x_normed, *args, **kwargs)

        def process_leaf(leaf_x, residual_correction):
            term1 = residual_correction + (mu1 - mu0) + scale * (leaf_x - approx_mut)
            return term1

        return jax.tree_util.tree_map(process_leaf, x, residual_correction)

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

    def loss(self, rng, data: ArrayLike, *args, **kwargs):
        rng_source, rng_times = jax.random.split(rng, 2)

        # Generate noise for x0
        x0 = (
            jax.random.normal(rng_source, shape=data.shape) * self.std0.value
            + self.mu0.value
        )

        # Get shape from data for time scheduling
        data_shape = data.shape
        ndims = data.ndim - 2
        times = self.noise_schedule(rng_times, (data_shape[0],) + (1,) * ndims)

        loss = self._loss(times, x0, data, *args, **kwargs)
        return loss


class MeanFlowMatcher(nnx.Module):
    def __init__(
        self,
        net,
        interpolation_fn,
        interpolation_std_fn=None,
        interpolation_grad_fn=None,
        interpolation_noise_grad_fn=None,
        mu0=0,
        std0=1,
        mu1=0.0,
        std1=1.0,
        rngs=None,
    ):
        self.net = net
        self.mu0 = nnx.Variable(mu0)
        self.std0 = nnx.Variable(std0)
        self.mu1 = nnx.Variable(mu1)
        self.std1 = nnx.Variable(std1)
        self.interpolation_fn = interpolation_fn
        self.interpolation_std_fn = interpolation_std_fn
        self.interpolation_grad_fn = interpolation_grad_fn
        self.interpolation_noise_grad_fn = interpolation_noise_grad_fn
        self.rngs = rngs
        self._loss = build_mean_flow_matching_loss(
            self,
            interpolation_fn=self.interpolation_fn,
            interpolation_noise_fn=self.interpolation_std_fn,
            interpolation_grad_fn=self.interpolation_grad_fn,
            interpolation_noise_grad_fn=self.interpolation_noise_grad_fn,
            weight_fn=None,
        )

    def __call__(
        self, t: ArrayLike, x: ArrayLike, r: Optional[ArrayLike] = None, *args, **kwargs
    ) -> ArrayLike:
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


        approx_mu_t = self.interpolation_fn(mu0, mu1, t)
        approx_std_t = jnp.sqrt(t**2 * std1**2 + (1 - t) ** 2 * std0**2)

        x_normed = jax.tree_util.tree_map(lambda x: (x - approx_mu_t) / approx_std_t, x)
        std_t = jnp.sqrt(t**2 * std1**2 + (1 - t) ** 2 * std0**2)
        std_r = jnp.sqrt(r**2 * std1**2 + (1 - r) ** 2 * std0**2)
        scale = ((t * std1**2) - (1 - t) * std0**2) / (
            (1 - t) ** 2 * std0**2 + t**2 * std1**2
        )
        def g(h):
            return 1. + jnp.tanh(h/0.1)
        geo_std = jnp.sqrt(std_r * std_t)
        scale_residual = geo_std * g(r - t)

        pred_mu1 = self.net(t, x_normed, *args, r=r, **kwargs)

        return mu1 - mu0 + scale * (x - approx_mu_t) + scale_residual * pred_mu1

    def noise_schedule(
        self,
        rng,
        shape,
        percent_rt=0.25,
        mu_rt=-0.4,
        scale_rt=1.0,
        mu_t=0.4,
        scale_t=1.0,
    ):
        batch_size = shape[0]
        batch_size_different = int(batch_size * percent_rt)
        batch_size_same = batch_size - batch_size_different

        rng_t, rng_r, rng_tr = jax.random.split(rng, 3)

        t1 = jax.nn.sigmoid(
            jax.random.normal(rng_t, (batch_size_different,) + shape[1:] + (1,))
            * scale_rt
            - mu_rt
        )
        r1 = jax.nn.sigmoid(
            jax.random.normal(rng_r, (batch_size_different,) + shape[1:] + (1,))
            * scale_rt
            - mu_rt
        )
        r1 = jnp.clip(t1 + r1, a_min=0.0, a_max=1.0)

        t2 = r2 = jax.nn.sigmoid(
            jax.random.normal(rng_tr, (batch_size_same,) + shape[1:] + (1,)) * scale_t
            - mu_t
        )

        t = jnp.concatenate([t1, t2], axis=0)
        r = jnp.concatenate([r1, r2], axis=0)

        return t, r

    def loss(
        self,
        rng,
        data: ArrayLike,
        *args,
        adaptive_weight_p: float = 0.3,
        adaptive_weight_eps: float = 1e-3,
        **kwargs,
    ):
        rng_source, rng_times = jax.random.split(rng, 2)
        ndims = data.ndim - 2
        times_t, times_r = self.noise_schedule(
            rng_times, (data.shape[0],) + (1,) * ndims
        )

        x0 = (
            jax.random.normal(rng_source, shape=data.shape) * self.std0.value
            + self.mu0.value
        )
        loss = self._loss(
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


class LinearFlow(FlowMatcher):
    def __init__(
        self,
        net: nnx.Module,
        mu0: ArrayLike = 0.0,
        std0: ArrayLike = 1.0,
        mu1: ArrayLike = 0.0,
        std1: ArrayLike = 1.0,
        rngs=None,
        loss_kwargs=None,
    ):
        super().__init__(
            net,
            mu0=mu0,
            std0=std0,
            mu1=mu1,
            std1=std1,
            rngs=rngs,
            interpolation_fn=lambda x0, x1, t: (1 - t) * x0 + t * x1,
            loss_kwargs=loss_kwargs,
        )

    def denoise(self, t, x: PyTree[ArrayLike]) -> PyTree[ArrayLike]:
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

    def score(self, t, x: PyTree[ArrayLike], max_t=1 - 1e-3) -> PyTree[ArrayLike]:
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

    def noise_schedule(self, rng, shape, mu=0.0, scale=1.0):
        return jax.nn.sigmoid(jax.random.normal(rng, shape=shape + (1,)) * scale + mu)

    def solve_schedule(self, num_steps=50):
        ts = jnp.linspace(0, 1, num_steps)
        return ts


class LinearMeanFlow(MeanFlowMatcher):
    def __init__(
        self,
        net,
        mu0=0,
        std0=1,
        mu1=0.0,
        std1=1.0,
        rngs=None,
    ):
        interpolation_fn = lambda x0, x1, t: (1 - t) * x0 + t * x1
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
        )

    def solve_schedule(self, num_steps=50):
        ts = jnp.linspace(0, 1, num_steps)
        return ts

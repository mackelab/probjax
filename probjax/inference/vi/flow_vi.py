"""Variational inference with a learned density as the variational family.

The objective is the reparameterised reverse KL, ``mean(log q(x) - log p(x))``
over draws ``x`` from ``q``. Each kernel step is one optimizer step, so the
fit runs through :func:`~probjax.inference.run` like any sampler:

>>> family, params = density(maf(dim, 4, rngs=nnx.Rngs(0)))
>>> kernel = flow_vi(logdensity_fn, family, optax.adam(1e-3))
>>> result = run(key, kernel, kernel.init(params), None, 2000, collect=("loss",))
>>> fitted = result.state.params

Reverse KL is **mode-seeking**. A flow fitted this way tends to under-cover: on
Neal's funnel it concentrates in the neck and reports a smaller variance than the
truth. That is a property of the objective, not of this implementation, and it is
the reason :func:`probjax.inference.neutra` exists -- running MCMC in the flow's
latent space stays asymptotically exact however imperfect the flow is, so the
flow only has to be *helpful*, not correct.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from probjax.inference.base import Density, Kernel
from probjax.utils.typing import Array

__all__ = ["FlowVIInfo", "FlowVIState", "flow_vi"]


class FlowVIState(NamedTuple):
    """Variational parameters and the optimizer state that drives them."""

    params: Any
    opt_state: Any


class FlowVIInfo(NamedTuple):
    """Per-step diagnostics.

    Attributes:
        loss: ``mean(log q - log p)``, the negative ELBO shifted by the
            target's unknown log-normaliser. It should go down; its absolute
            value is not interpretable.
    """

    loss: Array


def flow_vi(
    logdensity_fn: Callable,
    family: Density,
    optimizer: optax.GradientTransformation,
    *,
    num_samples: int = 100,
    stl_estimator: bool = True,
) -> Kernel:
    """Fit ``family`` to ``logdensity_fn`` by reverse KL.

    Args:
        logdensity_fn: the unnormalized target log-density, taking one position.
        family: the variational family; its ``sample`` must be reparameterised
            (flows from :func:`probjax.inference.neural.density` are).
        optimizer: an optax ``GradientTransformation``.
        num_samples: draws used to estimate the objective at each step.
        stl_estimator: use the sticking-the-landing gradient estimator, which
            lowers gradient variance at no cost in bias.

    Returns:
        A :class:`~probjax.inference.Kernel`. ``kernel.init(params)`` starts
        from the family's params; the step takes no kernel params, so pass
        ``None`` to the runner.
    """

    def init(params: Any) -> FlowVIState:
        return FlowVIState(params, optimizer.init(params))

    def step(rng, state: FlowVIState, params: Any = None, *args: Any):
        del params, args
        keys = jax.random.split(rng, num_samples)

        def objective(variational_params: Any) -> Array:
            draws = jax.vmap(family.sample, (0, None))(keys, variational_params)
            # Sticking the landing: with the entropy term's parameters
            # detached the score-function part of the gradient drops out, and
            # it is exactly zero in expectation -- so this removes variance
            # without adding bias.
            entropy_params = (
                jax.lax.stop_gradient(variational_params)
                if stl_estimator
                else variational_params
            )
            log_q = jax.vmap(family.logpdf, (0, None))(draws, entropy_params)
            log_p = jax.vmap(logdensity_fn)(draws)
            return jnp.mean(log_q - log_p)

        loss, gradient = jax.value_and_grad(objective)(state.params)
        updates, opt_state = optimizer.update(gradient, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return FlowVIState(new_params, opt_state), FlowVIInfo(loss)

    def init_params(state: FlowVIState) -> None:
        return None

    return Kernel(init, step, init_params)

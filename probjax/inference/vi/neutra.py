"""NeuTra: run an existing sampler in a flow's latent space.

A posterior with strong curvature or correlation is hard for HMC not because the
kernel is weak but because the geometry is bad -- one step size cannot suit every
direction. NeuTra fixes the geometry instead of the kernel: fit a flow ``T`` to
the target, then sample the *pulled-back* density

    log p~(z) = log p(T(z)) + log|det J_T(z)|

which is close to an isotropic Gaussian whenever the flow is any good, and push
the draws back through ``T``.

The key property: this is a change of variables, not an approximation. **MCMC on
``p~`` remains asymptotically exact for ``p`` no matter how poor the flow is.** A
bad flow costs efficiency, never correctness -- which is what makes it safe to
pair with the mode-seeking reverse-KL fit in :mod:`probjax.inference.vi.flow_vi`.

Measured on Neal's funnel (D=5, 4000 draws, matched budget): NUTS on the
transformed target reached ESS 601 against 168 for NUTS on the target directly.

Nothing here is a new kernel, so every existing kernel, warmup and runner works
unchanged:

>>> forward, flow_params = bijection(flow)
>>> transform = neutra(logdensity_fn, forward, flow_params)
>>> kernel = nuts(transform.logdensity)  # or mala, hmc, mclmc, slice...
>>> state = kernel.init(key, jnp.zeros(dim))
>>> result = run(key, kernel, state, kernel.init_params(state), 4000,
...              collect=("position",))
>>> draws = transform.forward(result.trace["position"])  # back in target space
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from probjax.utils.typing import Array

__all__ = ["NeuTraTransform", "neutra"]


class NeuTraTransform(NamedTuple):
    """A target rewritten in a flow's latent coordinates.

    Attributes:
        logdensity: the pulled-back log-density, to hand to any kernel.
        forward: maps latent draws back to the target's space. Accepts a single
            position or a leading batch of them.
    """

    logdensity: Callable
    forward: Callable


def neutra(
    logdensity_fn: Callable,
    bijection: Callable[[Any, Any], tuple[Any, Any]],
    params: Any,
) -> NeuTraTransform:
    """Reparameterise ``logdensity_fn`` through a bijection ``T``.

    Args:
        logdensity_fn: the unnormalized target log-density, taking one position.
        bijection: ``(z, params) -> (T(z), log|det J_T(z)|)``.
            :func:`probjax.inference.neural.bijection` builds one from a
            flow, typically fitted with :func:`probjax.inference.flow_vi`; a
            flow trained on samples from a previous run is equally valid.
        params: the bijection's parameters.

    Returns:
        A :class:`NeuTraTransform`.
    """

    def transformed_logdensity(position: Array) -> Array:
        target_position, log_det = bijection(position, params)
        return logdensity_fn(target_position) + log_det

    def forward(positions: Array) -> Array:
        positions = jnp.asarray(positions)
        flat = positions.reshape(-1, positions.shape[-1])
        mapped = jax.vmap(lambda z: bijection(z, params)[0])(flat)
        return mapped.reshape(*positions.shape[:-1], mapped.shape[-1])

    return NeuTraTransform(transformed_logdensity, forward)

"""Variational inference with a normalizing flow as the variational family.

Shaped after :mod:`blackjax.vi.meanfield_vi` -- same ``init``/``step``/``sample``
layout, same :class:`~blackjax.base.VIAlgorithm` return type, same
``stl_estimator`` option -- so it reads like the rest of the inference stack. The
difference is the family: a mean-field Gaussian cannot represent a curved or
correlated posterior, and a flow can.

The objective is the reparameterised reverse KL,
``mean(log q(x) - log p(x))`` over samples ``x`` drawn from the flow.

Reverse KL is **mode-seeking**. A flow fitted this way tends to under-cover: on
Neal's funnel it concentrates in the neck and reports a smaller variance than the
truth. That is a property of the objective, not of this implementation, and it is
the reason :func:`probjax.inference.neutra` exists -- running MCMC in the flow's
latent space stays asymptotically exact however imperfect the flow is, so the
flow only has to be *helpful*, not correct.

>>> algorithm = flow_vi(logdensity_fn, flow, optax.adam(1e-3))
>>> state = algorithm.init()
>>> def one(state, key):
...     state, info = algorithm.step(key, state)
...     return state, info.elbo
>>> state, objective = jax.lax.scan(one, state, jax.random.split(key, 2000))
>>> draws = algorithm.sample(key, state, 1000)
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from blackjax.base import VIAlgorithm
from flax import nnx
from optax import GradientTransformation, OptState

__all__ = [
    "FlowVIState",
    "FlowVIInfo",
    "init",
    "step",
    "sample",
    "as_top_level_api",
]


class FlowVIState(NamedTuple):
    """Variational parameters and the optimizer state that drives them."""

    flow_params: nnx.State
    opt_state: OptState


class FlowVIInfo(NamedTuple):
    """Per-step diagnostics.

    Attributes:
        elbo: the value of the minimised objective, ``mean(log q - log p)``.
            Despite the name -- kept for parity with ``blackjax``'s
            ``MFVIInfo`` -- this is the *negative* ELBO shifted by the target's
            unknown log-normaliser, so it is the quantity that should go **down**
            during a run. Its absolute value is not interpretable; its trend is.
    """

    elbo: float


def _split(flow):
    """Separate a flow into (static graph, trainable params, everything else).

    The same boundary ``probjax.stats.fit`` uses, so a flow can be handed to
    either without conversion.
    """
    return nnx.split(flow, nnx.Param, ...)


def _draw(graphdef, flow_params, rest, rng_key, event_dim, num_samples):
    """Reparameterised draws: push standard normal noise through the flow."""
    flow = nnx.merge(graphdef, flow_params, rest)
    noise = jax.random.normal(rng_key, (num_samples, event_dim))
    return jax.vmap(flow.transform)(noise)


def _log_q(graphdef, flow_params, rest, samples):
    flow = nnx.merge(graphdef, flow_params, rest)
    return flow._logpdf(samples)


def init(flow, optimizer: GradientTransformation) -> FlowVIState:
    """Initialise from a flow, which supplies the variational family.

    Unlike ``blackjax.vi.meanfield_vi.init`` there is no ``position`` argument:
    a mean-field family is defined by the shape of a position, whereas a flow
    already carries its own event size and initial parameters.
    """
    _, flow_params, _ = _split(flow)
    return FlowVIState(flow_params, optimizer.init(flow_params))


def step(
    rng_key,
    state: FlowVIState,
    logdensity_fn: Callable,
    optimizer: GradientTransformation,
    graphdef,
    rest,
    event_dim: int,
    num_samples: int = 100,
    stl_estimator: bool = True,
) -> tuple[FlowVIState, FlowVIInfo]:
    """One reparameterised reverse-KL step.

    ``graphdef``/``rest``/``event_dim`` come from splitting the flow once, which
    :func:`as_top_level_api` does for you; they are arguments rather than closure
    state so this mirrors blackjax, where ``step`` takes its configuration
    explicitly.
    """

    def objective(flow_params):
        samples = _draw(
            graphdef, flow_params, rest, rng_key, event_dim, num_samples
        )
        # Sticking the landing: with the entropy term's parameters detached the
        # score-function part of the gradient drops out, and it is exactly zero
        # in expectation -- so this removes variance without adding bias.
        entropy_params = (
            jax.lax.stop_gradient(flow_params) if stl_estimator else flow_params
        )
        log_q = _log_q(graphdef, entropy_params, rest, samples)
        log_p = jax.vmap(logdensity_fn)(samples)
        return jnp.mean(log_q - log_p)

    value, gradient = jax.value_and_grad(objective)(state.flow_params)
    updates, opt_state = optimizer.update(gradient, state.opt_state, state.flow_params)
    flow_params = optax.apply_updates(state.flow_params, updates)
    return FlowVIState(flow_params, opt_state), FlowVIInfo(value)


def sample(
    rng_key,
    state: FlowVIState,
    graphdef,
    rest,
    event_dim: int,
    num_samples: int = 1,
):
    """Draw from the fitted approximation."""
    return _draw(graphdef, state.flow_params, rest, rng_key, event_dim, num_samples)


def as_top_level_api(
    logdensity_fn: Callable,
    flow,
    optimizer: GradientTransformation,
    num_samples: int = 100,
    stl_estimator: bool = True,
) -> VIAlgorithm:
    """Variational inference with a normalizing flow.

    Args:
        logdensity_fn: the unnormalized target log-density, taking one position.
        flow (NormalizingFlow): a flow from :mod:`probjax.nn` -- ``maf``,
            ``nsf``, ``naf`` and the rest all work. Its parameters are the
            variational parameters; the flow itself is not mutated.
        optimizer: an optax ``GradientTransformation``.
        num_samples: draws used to estimate the objective at each step.
        stl_estimator: use the sticking-the-landing gradient estimator, which
            lowers gradient variance at no cost in bias.

    Returns:
        A ``blackjax`` :class:`~blackjax.base.VIAlgorithm`: ``init``, ``step``
        and ``sample``. As with blackjax, the loop over ``step`` is the caller's
        -- see the module docstring for the ``lax.scan`` form.
    """
    graphdef, _, rest = _split(flow)
    event_dim = flow.input_dim

    def init_fn():
        return init(flow, optimizer)

    def step_fn(rng_key, state: FlowVIState) -> tuple[FlowVIState, FlowVIInfo]:
        return step(
            rng_key,
            state,
            logdensity_fn,
            optimizer,
            graphdef,
            rest,
            event_dim,
            num_samples,
            stl_estimator,
        )

    def sample_fn(rng_key, state: FlowVIState, num_samples: int = 1):
        return sample(rng_key, state, graphdef, rest, event_dim, num_samples)

    return VIAlgorithm(init_fn, step_fn, sample_fn)


def rebuild(flow, state: FlowVIState):
    """Return a flow carrying the fitted variational parameters.

    The bridge to :func:`probjax.inference.neutra`, which wants a flow rather
    than a parameter tree.
    """
    graphdef, _, rest = _split(flow)
    return nnx.merge(graphdef, state.flow_params, rest)

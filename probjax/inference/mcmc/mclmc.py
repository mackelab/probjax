from __future__ import annotations

from typing import Callable, NamedTuple

import blackjax
import jax.numpy as jnp
from blackjax.mcmc import integrators
from blackjax.mcmc.adjusted_mclmc_dynamic import trajectory_length
from blackjax.mcmc.dynamic_hmc import DynamicHMCState
from blackjax.mcmc.hmc import HMCInfo, HMCState
from blackjax.mcmc.integrators import IntegratorState
from blackjax.mcmc.mclmc import MCLMCInfo

from probjax.inference.mcmc._dynamic_stepping import (
    get_dynamic_stepping,
    init_dynamic_arg,
)
from probjax.inference.mcmc.adaptation import step_size_adaptor
from probjax.inference.mcmc.base import (
    MarkovKernelAPI,
    init_hmc_like_params,
    scale_step_size_by_grad,
)
from probjax.inference.mcmc.warmup import mclmc_warmup
from probjax.utils.typing import Array, ArrayLike, PyTree, Scalar


class MCLMCParams(NamedTuple):
    """Static parameters of the unadjusted MCLMC sampler.

    Attributes
    ----------
    step_size : float
        Integrator step size.
    L : float
        Trajectory length.
    inverse_mass_matrix : ArrayLike
        Inverse mass matrix used to precondition the dynamics.
    """

    step_size: Scalar
    L: Scalar
    inverse_mass_matrix: ArrayLike


def init_mclmc_params(
    state: PyTree,
    step_size: float = 0.1,
    L: float = 1.0,
    inverse_mass_matrix: Array | None = None,
) -> MCLMCParams:
    """Initialize the parameters of the unadjusted MCLMC sampler.

    Parameters
    ----------
    state : PyTree
        Initial state, used to pick a default inverse mass matrix and to scale
        the step size by the gradient magnitude.
    step_size : float, optional
        Initial integrator step size. Default is 0.1.
    L : float, optional
        Trajectory length. Default is 1.0.
    inverse_mass_matrix : Array, optional
        Inverse mass matrix. Inferred from ``state`` when ``None``.

    Returns
    -------
    MCLMCParams
        Resolved sampler parameters.
    """
    inverse_mass_matrix = init_hmc_like_params(state, inverse_mass_matrix)
    scaled_step_size = scale_step_size_by_grad(state, step_size)
    return MCLMCParams(
        step_size=scaled_step_size, L=L, inverse_mass_matrix=inverse_mass_matrix
    )


def build_mclmc_step(
    logdensity_fn: Callable,
    integrator: Callable = integrators.isokinetic_mclachlan,
) -> Callable:
    """Build the step; see :class:`mclmc` for the arguments."""
    # blackjax >= 1.6 takes logdensity_fn and inverse_mass_matrix on the kernel
    # rather than the builder, so the kernel no longer closes over per-step
    # parameters and can be built once here.
    kernel = blackjax.mclmc.build_kernel(integrator=integrator)

    def step(
        rng,
        state: IntegratorState,
        params: MCLMCParams,
    ) -> tuple[IntegratorState, MCLMCInfo]:
        return kernel(
            rng,
            state,
            logdensity_fn,
            inverse_mass_matrix=params.inverse_mass_matrix,
            L=params.L,
            step_size=params.step_size,
        )

    return step


def _adjusted_mclmc_adaptation(target: float = 0.9):
    return step_size_adaptor(target=target)


class mclmc(MarkovKernelAPI):
    """Microcanonical Langevin Monte Carlo without Metropolis adjustment.

    Isokinetic dynamics with partial momentum refreshment; asymptotically
    biased, so tune ``step_size`` and ``L`` with :func:`mclmc_warmup` (the
    default adaptation). Tunable parameters come from
    ``kernel.init_params(state, step_size=0.1, L=1.0,
    inverse_mass_matrix=None)`` (see :class:`MCLMCParams`).

    Args:
        logdensity_fn: Log density of the target.
        integrator: Isokinetic integrator from ``blackjax.mcmc.integrators``.
    """

    init = staticmethod(blackjax.mclmc.init)
    init_params = staticmethod(init_mclmc_params)
    build_step = staticmethod(build_mclmc_step)
    default_adaptation_fn = staticmethod(mclmc_warmup)


class AdjustedMCLMCParams(NamedTuple):
    """Static parameters of the adjusted MCLMC sampler.

    Attributes
    ----------
    step_size : float
        Integrator step size.
    num_integration_steps : int
        Number of integration steps per proposal.
    inverse_mass_matrix : ArrayLike
        Inverse mass matrix used to precondition the dynamics.
    L_proposal_factor : float
        Factor controlling the proposal trajectory length.
    """

    step_size: Scalar
    num_integration_steps: int
    inverse_mass_matrix: ArrayLike
    L_proposal_factor: Scalar


def init_params(
    state: PyTree,
    step_size: float = 0.5,
    num_integration_steps: int = 10,
    inverse_mass_matrix: Array | None = None,
    L_proposal_factor: float = jnp.inf,
) -> AdjustedMCLMCParams:
    """Initialize the parameters of the adjusted MCLMC sampler.

    Parameters
    ----------
    state : PyTree
        Initial state, used to pick a default inverse mass matrix and to scale
        the step size by the gradient magnitude.
    step_size : float, optional
        Initial integrator step size. Default is 0.5.
    num_integration_steps : int, optional
        Number of integration steps per proposal. Default is 10.
    inverse_mass_matrix : Array, optional
        Inverse mass matrix. Inferred from ``state`` when ``None``.
    L_proposal_factor : float, optional
        Factor controlling the proposal trajectory length. Default is ``inf``.

    Returns
    -------
    AdjustedMCLMCParams
        Resolved sampler parameters.
    """
    inverse_mass_matrix = init_hmc_like_params(state, inverse_mass_matrix)
    scaled_step_size = scale_step_size_by_grad(state, step_size)
    return AdjustedMCLMCParams(
        step_size=scaled_step_size,
        num_integration_steps=num_integration_steps,
        inverse_mass_matrix=inverse_mass_matrix,
        L_proposal_factor=L_proposal_factor,
    )


def build_step(
    logdensity_fn: Callable,
    integrator: Callable = integrators.isokinetic_mclachlan,
    divergence_threshold: float = 1000.0,
) -> Callable:
    """Build the step; see :class:`adjusted_mclmc` for the arguments."""
    kernel = blackjax.adjusted_mclmc.build_kernel(
        integrator=integrator,
        divergence_threshold=divergence_threshold,
    )

    def step(
        rng,
        state: HMCState,
        params: AdjustedMCLMCParams,
    ) -> tuple[HMCState, HMCInfo]:
        return kernel(
            rng,
            state,
            logdensity_fn,
            step_size=params.step_size,
            # blackjax >= 1.6 passes the step count through a tuple that the
            # kernel unpacks, so that it can be adapted without a rebuild.
            integration_steps_params=(params.num_integration_steps,),
            inverse_mass_matrix=params.inverse_mass_matrix,
            L_proposal_factor=params.L_proposal_factor,
        )

    return step


class adjusted_mclmc(MarkovKernelAPI):
    """Metropolis-adjusted MCLMC with a fixed number of integration steps.

    Tunable parameters come from ``kernel.init_params(state, step_size=0.5,
    num_integration_steps=10, inverse_mass_matrix=None,
    L_proposal_factor=inf)`` (see :class:`AdjustedMCLMCParams`). Default
    adaptation: step-size dual averaging towards a 0.9 acceptance rate.

    Args:
        logdensity_fn: Log density of the target.
        integrator: Isokinetic integrator from ``blackjax.mcmc.integrators``.
        divergence_threshold: Energy error above which a transition is
            flagged as divergent.
    """

    init = staticmethod(blackjax.adjusted_mclmc.init)
    init_params = staticmethod(init_params)
    build_step = staticmethod(build_step)
    default_adaptation_fn = staticmethod(_adjusted_mclmc_adaptation)


class AdjustedMCLMCDynamicParams(NamedTuple):
    """Static parameters of the adjusted MCLMC dynamic sampler.

    Attributes
    ----------
    step_size : float
        Integrator step size.
    inverse_mass_matrix : ArrayLike
        Inverse mass matrix used to precondition the dynamics.
    L_proposal_factor : float
        Factor controlling the proposal trajectory length.
    """

    step_size: Scalar
    inverse_mass_matrix: ArrayLike
    L_proposal_factor: Scalar


def init_dynamic_params(
    state: PyTree,
    step_size: float = 0.5,
    inverse_mass_matrix: Array | None = None,
    L_proposal_factor: float = jnp.inf,
) -> AdjustedMCLMCDynamicParams:
    """Initialize the parameters of the adjusted MCLMC dynamic sampler.

    Parameters
    ----------
    state : PyTree
        Initial state, used to pick a default inverse mass matrix and to scale
        the step size by the gradient magnitude.
    step_size : float, optional
        Initial integrator step size. Default is 0.5.
    inverse_mass_matrix : Array, optional
        Inverse mass matrix. Inferred from ``state`` when ``None``.
    L_proposal_factor : float, optional
        Factor controlling the proposal trajectory length. Default is ``inf``.

    Returns
    -------
    AdjustedMCLMCDynamicParams
        Resolved sampler parameters.
    """
    inverse_mass_matrix = init_hmc_like_params(state, inverse_mass_matrix)
    scaled_step_size = scale_step_size_by_grad(state, step_size)
    return AdjustedMCLMCDynamicParams(
        step_size=scaled_step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        L_proposal_factor=L_proposal_factor,
    )


def build_dynamic_step(
    logdensity_fn: Callable,
    average_integration_steps: int = 10,
    integration_steps_sequence: str = "halton",
    divergence_threshold: float = 1000.0,
    integrator: Callable = integrators.isokinetic_mclachlan,
    next_random_arg_fn: Callable | None = None,
    integration_steps_fn: Callable | None = None,
) -> Callable:
    """Build the step; see :class:`adjusted_mclmc_dynamic` for the arguments."""
    if integration_steps_fn is None:
        random_arg_next_fn, integration_steps_fn = get_dynamic_stepping(
            integration_steps_sequence,
            average_integration_steps,
            length_fn=trajectory_length,
        )
    else:
        random_arg_next_fn = next_random_arg_fn

    if random_arg_next_fn is None:
        raise ValueError(
            "next_random_arg_fn must be provided when integration_steps_fn is set"
        )

    kernel = blackjax.adjusted_mclmc_dynamic.build_kernel(
        integration_steps_fn=integration_steps_fn,
        integrator=integrator,
        divergence_threshold=divergence_threshold,
        next_random_arg_fn=random_arg_next_fn,
    )

    def step(
        rng,
        state: DynamicHMCState,
        params: AdjustedMCLMCDynamicParams,
    ) -> tuple[DynamicHMCState, HMCInfo]:
        return kernel(
            rng,
            state,
            logdensity_fn,
            step_size=params.step_size,
            L_proposal_factor=params.L_proposal_factor,
            inverse_mass_matrix=params.inverse_mass_matrix,
        )

    return step


def init_dynamic(
    position,
    logdensity_fn,
    rng,
    integration_steps_sequence: str = "halton",
):
    """Initialize the state of the adjusted MCLMC dynamic sampler.

    Parameters
    ----------
    position : array_like
        Initial position of the chain.
    logdensity_fn : callable
        Function returning the (unnormalized) log density at a position.
    rng : jax.random.PRNGKey
        Random key used to initialize the random generator argument.
    integration_steps_sequence : str, optional
        Name of the integration-step sequence (e.g. ``"halton"``). Default is
        ``"halton"``.

    Returns
    -------
    DynamicHMCState
        Initial sampler state.
    """
    random_generator_arg = init_dynamic_arg(rng, integration_steps_sequence)
    return blackjax.adjusted_mclmc_dynamic.init(
        position, logdensity_fn, random_generator_arg=random_generator_arg
    )


class adjusted_mclmc_dynamic(MarkovKernelAPI):
    """Metropolis-adjusted MCLMC with a varying number of integration steps.

    Tunable parameters come from ``kernel.init_params(state, step_size=0.5,
    inverse_mass_matrix=None, L_proposal_factor=inf)`` (see
    :class:`AdjustedMCLMCDynamicParams`). Default adaptation:
    step-size dual averaging towards a 0.9 acceptance rate.

    Args:
        logdensity_fn: Log density of the target.
        average_integration_steps: Mean number of integrator steps.
        integration_steps_sequence: How the step counts are drawn:
            ``"halton"``, ``"random"`` or a ``(next_random_arg_fn,
            integration_steps_fn)`` pair.
        divergence_threshold: Energy error above which a transition is
            flagged as divergent.
        integrator: Isokinetic integrator from ``blackjax.mcmc.integrators``.
        next_random_arg_fn: Advances the random argument; required with
            ``integration_steps_fn``.
        integration_steps_fn: Custom step-count function, overriding
            ``integration_steps_sequence``.
    """

    init = staticmethod(init_dynamic)
    init_params = staticmethod(init_dynamic_params)
    build_step = staticmethod(build_dynamic_step)
    default_adaptation_fn = staticmethod(_adjusted_mclmc_adaptation)

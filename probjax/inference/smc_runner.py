from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from probjax.utils.typing import RngKey

from probjax.inference.smc.base import SMCKernel
from probjax.utils.jaxutils import WithProgressBarAPI, print_scan


def _ess_from_weights(state, _info):
    """Effective sample size from normalized SMC weights: 1 / sum(w^2)."""
    return 1.0 / jnp.sum(state.weights**2)


class SMC(WithProgressBarAPI):
    """Run an SMC kernel, optionally displaying a progress bar with diagnostics.

    Defaults track ESS (the canonical particle-collapse indicator),
    log-likelihood increment, and an acceptance rate sourced either from the
    SMC info or from the inner MCMC kernel info (``info.update_info``).

    Args:
        kernel: An :class:`SMCKernel`.
        verbose: If ``True``, display a progress bar with running stats.
        tracked_stats: Names of scalars to display. Each name is looked up on
            **state**, then **info**, then ``info.update_info`` (the inner
            MCMC kernel info), then in the computed registry (``"ess"``).
            Names may be dotted paths. Defaults to
            ``("ess", "log_likelihood_increment", "acceptance_rate")``.
    """

    _default_tracked_stats = ("ess", "log_likelihood_increment", "acceptance_rate")
    _computed_stats = {"ess": _ess_from_weights}

    def __init__(
        self,
        kernel: SMCKernel,
        verbose: bool = False,
        tracked_stats: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.kernel = kernel
        self.verbose = verbose
        self.tracked_stats = tracked_stats or self._default_tracked_stats

    def _stat_objects(self, state, info):
        # Inner MCMC kernel info commonly carries acceptance_rate / logdensity.
        return (state, info, getattr(info, "update_info", None))

    @partial(jax.jit, static_argnums=(0, 5))
    def run(
        self,
        key: RngKey,
        state,
        tempering_params: jnp.ndarray,
        mcmc_parameters: dict,
        tune_params: bool = False,
        tune_kwargs: Optional[dict] = None,
    ):
        if tune_kwargs is None:
            tune_kwargs = {}

        def scan_fn(carry, t):
            key, state, params = carry
            key, new_key = jax.random.split(key)
            new_state, info = self.kernel.step(
                new_key, state, tempering_param=t, mcmc_parameters=params
            )
            if tune_params:
                params = self.kernel.tune_params(new_state, info, params, **tune_kwargs)
            stats = self._extract_stats(new_state, info) if self.verbose else None
            return (key, new_state, params), stats

        if not self.verbose:
            (key, out_state, out_params), _ = jax.lax.scan(
                scan_fn, (key, state, mcmc_parameters), tempering_params
            )
            return out_state, out_params

        update_stats, print_fn, init_stats, print_rate = self._make_verbose_fns(
            tempering_params.shape[0]
        )
        (key, out_state, out_params), _ = print_scan(
            scan_fn,
            (key, state, mcmc_parameters),
            init_stats,
            xs=tempering_params,
            length=tempering_params.shape[0],
            update_stats=update_stats,
            print_fn=print_fn,
            print_rate=print_rate,
        )
        return out_state, out_params

    def sample(
        self,
        key: RngKey,
        state,
        tempering_params: jnp.ndarray,
        mcmc_parameters: dict,
        tune_params: bool = False,
        tune_kwargs: Optional[dict] = None,
    ):
        if tune_kwargs is None:
            tune_kwargs = {}

        num_steps = tempering_params.shape[0]
        particles = jax.tree_util.tree_map(
            lambda x: jnp.empty((num_steps,) + x.shape), state.particles
        )
        weights = jnp.empty((num_steps,) + state.weights.shape)

        def scan_fn(carry, i):
            particles, weights, key, state, params = carry
            key, new_key = jax.random.split(key)
            t = tempering_params[i]
            new_state, info = self.kernel.step(
                new_key, state, tempering_param=t, mcmc_parameters=params
            )
            if tune_params:
                params = self.kernel.tune_params(new_state, info, params, **tune_kwargs)

            particles = jax.tree_util.tree_map(
                lambda s, s_new: s.at[i].set(s_new), particles, new_state.particles
            )
            weights = weights.at[i].set(new_state.weights)
            return (particles, weights, new_key, new_state, params), None

        (particles, weights, _, final_state, final_params), _ = jax.lax.scan(
            scan_fn,
            (particles, weights, key, state, mcmc_parameters),
            jnp.arange(num_steps),
            length=num_steps,
        )

        return particles, weights, final_state, final_params

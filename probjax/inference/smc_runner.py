from functools import partial
from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
from probjax.utils.typing import RngKey

from probjax.inference.smc.base import SMCKernel
from probjax.utils.jaxutils import WithProgressBarAPI, print_scan


class SMC(WithProgressBarAPI):
    _running_stats = ("log_likelihood_increment",)
    _state_gamma = 0.9

    def __init__(self, kernel: SMCKernel, verbose: bool = False) -> None:
        self.kernel = kernel
        self.verbose = verbose

    @partial(jax.jit, static_argnums=(0,))
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
            return (key, new_state, params), info_filter(info)

        if not self.verbose:
            info_filter = lambda x: None
            (key, out_state, out_params), _ = jax.lax.scan(
                scan_fn, (key, state, mcmc_parameters), tempering_params
            )
            return out_state, out_params
        else:
            info_filter = lambda x: tuple(
                getattr(x, stat) for stat in self._running_stats
            )
            update_stats = lambda stats, _, y: tuple(
                self._state_gamma * stats[i] + (1 - self._state_gamma) * y[i]
                for i in range(len(stats))
            )
            print_fn = lambda i, total, state: self._print_progress(
                type(self), i, total, state
            )
            init_stats = tuple([0.0 for _ in self._running_stats])
            (key, out_state, out_params), _ = print_scan(
                scan_fn,
                (key, state, mcmc_parameters),
                init_stats,
                xs=tempering_params,
                length=tempering_params.shape[0],
                update_stats=update_stats,
                print_fn=print_fn,
                print_rate=tempering_params.shape[0] // self._print_rate + 1,
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

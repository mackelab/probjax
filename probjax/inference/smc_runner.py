from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from probjax.inference.base import Adaptor, RunnerMixin, SMCResult
from probjax.inference.smc.base import SMCKernel
from probjax.utils.jaxutils import WithProgressBarAPI
from probjax.utils.typing import RngKey


def _ess_from_weights(state, _info):
    return 1.0 / jnp.sum(state.weights**2)


def _smc_run_body(kernel, params, collect, stats_fn=None):
    """Single-step scan body shared by silent and verbose SMC ``run``."""

    def one_step(state, xs):
        tempering_param, step_key = xs
        state, info = kernel.step(
            step_key,
            state,
            tempering_param=tempering_param,
            mcmc_parameters=params,
        )
        output = (state, info) if collect else None
        if stats_fn is None:
            return state, output
        return state, (stats_fn(state, info), output)

    return one_step


def _smc_adapt_body(kernel, adaptor, collect, stats_fn=None):
    """Single-step scan body shared by silent and verbose SMC ``adapt``."""

    def one_step(carry, xs):
        state, params, adapt_state = carry
        tempering_param, step_key = xs
        state, info = kernel.step(
            step_key,
            state,
            tempering_param=tempering_param,
            mcmc_parameters=params,
        )
        adapt_state, params, adapt_info = adaptor.update(
            state, info, adapt_state, params
        )
        output = (state, info, adapt_info) if collect else None
        if stats_fn is None:
            return (state, params, adapt_state), output
        return (state, params, adapt_state), (stats_fn(state, info), output)

    return one_step


class SMC(WithProgressBarAPI, RunnerMixin):
    """Compiled standard execution for an SMC kernel."""

    _default_tracked_stats = ("ess", "log_likelihood_increment", "acceptance_rate")
    _computed_stats = {"ess": _ess_from_weights}
    _print_rate = 50

    def __init__(
        self,
        kernel: SMCKernel,
        verbose: bool = False,
        tracked_stats: Optional[Tuple[str, ...]] = None,
        collect: bool = False,
    ) -> None:
        self.kernel = kernel
        self.verbose = verbose
        self.tracked_stats = (
            self._default_tracked_stats if tracked_stats is None else tracked_stats
        )
        self.collect = collect

    def _stat_objects(self, state, info):
        return (state, info, getattr(info, "update_info", None))

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=("kernel", "collect"),
    )
    def run_kernel(
        key: RngKey,
        kernel: SMCKernel,
        state,
        tempering_params: jnp.ndarray,
        params,
        *,
        collect: bool = False,
    ) -> SMCResult:
        """Run a compiled SMC schedule without constructing a runner."""
        keys = jax.random.split(key, tempering_params.shape[0])

        one_step = _smc_run_body(kernel, params, collect)

        state, trace = jax.lax.scan(one_step, state, (tempering_params, keys))
        return SMCResult(state, params, trace if collect else None)

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=("kernel", "adaptor", "collect"),
    )
    def adapt_kernel(
        key: RngKey,
        kernel: SMCKernel,
        adaptor: Adaptor,
        state,
        tempering_params: jnp.ndarray,
        params,
        *,
        collect: bool = False,
    ) -> SMCResult:
        """Run a compiled SMC schedule with parameter adaptation."""
        adapt_state = adaptor.init(state, params)
        keys = jax.random.split(key, tempering_params.shape[0])

        one_step = _smc_adapt_body(kernel, adaptor, collect)

        (state, params, adapt_state), trace = jax.lax.scan(
            one_step,
            (state, params, adapt_state),
            (tempering_params, keys),
        )
        params, final_info = adaptor.finalize(adapt_state, params)
        info = (trace, final_info) if collect else None
        return SMCResult(state, params, info)

    def _run_verbose(self, key, state, tempering_params, params, collect):
        keys = jax.random.split(key, tempering_params.shape[0])
        xs = (tempering_params, keys)

        one_step = _smc_run_body(
            self.kernel, params, collect, stats_fn=self._extract_stats
        )

        state, (_, trace) = self._verbose_scan(
            one_step,
            state,
            xs,
            tempering_params.shape[0],
            stats_fn=lambda _carry, y: y[0],
        )
        return SMCResult(state, params, trace if collect else None)

    def run(self, key, state, tempering_params, params) -> SMCResult:
        """Run the schedule with this runner's configuration."""
        if self.verbose:
            return self._run_verbose(key, state, tempering_params, params, self.collect)
        return self.run_kernel(
            key,
            self.kernel,
            state,
            tempering_params,
            params,
            collect=self.collect,
        )

    def sample(self, key, state, tempering_params, params) -> SMCResult:
        """Run and retain every population and transition diagnostic."""
        if not self.verbose:
            return self.run_kernel(
                key,
                self.kernel,
                state,
                tempering_params,
                params,
                collect=True,
            )
        return self._run_verbose(key, state, tempering_params, params, True)

    def adapt(
        self,
        key: RngKey,
        adaptor: Adaptor,
        state,
        tempering_params: jnp.ndarray,
        params,
    ) -> SMCResult:
        """Run the schedule with a composable parameter adaptor."""
        if not self.verbose:
            return self.adapt_kernel(
                key,
                self.kernel,
                adaptor,
                state,
                tempering_params,
                params,
                collect=self.collect,
            )

        keys = jax.random.split(key, tempering_params.shape[0])
        xs = (tempering_params, keys)
        adapt_state = adaptor.init(state, params)

        one_step = _smc_adapt_body(
            self.kernel, adaptor, self.collect, stats_fn=self._extract_stats
        )

        (state, params, adapt_state), (_, trace) = self._verbose_scan(
            one_step,
            (state, params, adapt_state),
            xs,
            tempering_params.shape[0],
            stats_fn=lambda _carry, y: y[0],
        )
        params, final_info = adaptor.finalize(adapt_state, params)
        info = (trace, final_info) if self.collect else None
        return SMCResult(state, params, info)

from functools import partial
from typing import Optional, Tuple

import jax

from probjax.inference.adaptation import adapt_step, adaptor_warmup
from probjax.inference.base import (
    AdaptationResult,
    Adaptor,
    MCMCResult,
    Warmup,
    WarmupResult,
)
from probjax.inference.mcmc.base import MarkovKernel, Params, State
from probjax.utils.jaxutils import WithProgressBarAPI, print_scan
from probjax.utils.typing import RngKey


def _select(value, fields):
    return {field: getattr(value, field) for field in fields}


class MCMC(WithProgressBarAPI):
    """Compiled standard execution for an MCMC kernel.

    The static methods are standalone compiled primitives. Instance methods
    apply the runner's kernel, collection, and progress configuration.
    """

    _default_tracked_stats = ("logdensity", "acceptance_rate")
    _print_rate = 50

    def __init__(
        self,
        kernel: MarkovKernel,
        verbose: bool = False,
        tracked_stats: Optional[Tuple[str, ...]] = None,
        collect_state: Tuple[str, ...] = (),
        collect_info: Tuple[str, ...] = (),
    ) -> None:
        self.kernel = kernel
        self.verbose = verbose
        self.tracked_stats = (
            self._default_tracked_stats if tracked_stats is None else tracked_stats
        )
        self.collect_state = collect_state
        self.collect_info = collect_info

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "kernel",
            "num_steps",
            "collect_state",
            "collect_info",
        ),
    )
    def run_kernel(
        key: RngKey,
        kernel: MarkovKernel,
        state: State,
        num_steps: int,
        params: Params,
        args: Optional[Tuple] = None,
        *,
        collect_state: Tuple[str, ...] = (),
        collect_info: Tuple[str, ...] = (),
    ) -> MCMCResult:
        """Run a compiled kernel scan without constructing a runner."""
        keys = jax.random.split(key, num_steps)
        xs = keys if args is None else (keys, args)

        def one_step(state, xs):
            if args is None:
                step_key, step_args = xs, ()
            else:
                step_key, step_args = xs
            state, info = kernel(step_key, state, params, *step_args)
            return state, (
                _select(state, collect_state),
                _select(info, collect_info),
            )

        state, (states, info) = jax.lax.scan(one_step, state, xs)
        trace = states if collect_state else None
        diagnostics = info if collect_info else None
        return MCMCResult(state, trace, diagnostics)

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=(
            "kernel",
            "num_samples",
            "thin",
            "collect_info",
        ),
    )
    def sample_kernel(
        key: RngKey,
        kernel: MarkovKernel,
        state: State,
        num_samples: int,
        params: Params,
        args: Optional[Tuple] = None,
        *,
        thin: int = 1,
        collect_info: Tuple[str, ...] = (),
    ) -> MCMCResult:
        """Collect positions from a compiled kernel scan."""
        keys = jax.random.split(key, (num_samples, thin))
        step_args = None
        if args is not None:
            step_args = jax.tree.map(
                lambda value: value.reshape((num_samples, thin) + value.shape[1:]),
                args,
            )
        xs = keys if step_args is None else (keys, step_args)

        def collect_one(state, xs):
            if step_args is None:
                sample_keys, sample_args = xs, None
            else:
                sample_keys, sample_args = xs
            inner_xs = (
                sample_keys if sample_args is None else (sample_keys, sample_args)
            )

            def transition(state, inner_xs):
                if sample_args is None:
                    step_key, current_args = inner_xs, ()
                else:
                    step_key, current_args = inner_xs
                state, info = kernel(step_key, state, params, *current_args)
                return state, (info, _select(info, collect_info))

            state, (step_info, selected_info) = jax.lax.scan(
                transition, state, inner_xs
            )
            diagnostics = jax.tree.map(lambda value: value[-1], selected_info)
            return state, (state.position, diagnostics)

        state, (samples, info) = jax.lax.scan(collect_one, state, xs)
        return MCMCResult(state, samples, info if collect_info else None)

    adaptor_warmup = staticmethod(adaptor_warmup)
    adapt_step = staticmethod(adapt_step)

    def _prepare_params(self, state: State, params: Optional[Params]) -> Params:
        return self.kernel.init_params(state) if params is None else params

    def _verbose_scan(self, f, init, xs, length, stats_fn):
        """Run a scan with a rate-limited progress bar.

        ``print_scan`` requires a tuple carry, so we wrap the kernel state.
        """

        def wrapped(carry, x):
            state, y = f(carry[0], x)
            return (state,), y

        update_stats, print_fn, init_stats, print_rate = self._make_verbose_fns(
            length, stats_fn=lambda carry, y: stats_fn(carry[0], y)
        )
        (state,), y = print_scan(
            wrapped,
            (init,),
            init_stats,
            xs=xs,
            length=length,
            update_stats=update_stats,
            print_rate=print_rate,
            print_fn=print_fn,
        )
        return state, y

    def run(
        self,
        key: RngKey,
        state: State,
        num_steps: int,
        params: Optional[Params] = None,
        args: Optional[Tuple] = None,
    ) -> MCMCResult:
        """Run ``num_steps`` transitions with this runner's configuration."""
        params = self._prepare_params(state, params)
        if not self.verbose:
            return self.run_kernel(
                key,
                self.kernel,
                state,
                num_steps,
                params,
                args,
                collect_state=self.collect_state,
                collect_info=self.collect_info,
            )

        keys = jax.random.split(key, num_steps)
        xs = keys if args is None else (keys, args)

        def one_step(state, xs):
            if args is None:
                step_key, step_args = xs, ()
            else:
                step_key, step_args = xs
            state, info = self.kernel(step_key, state, params, *step_args)
            return state, (
                self._extract_stats(state, info),
                _select(state, self.collect_state),
                _select(info, self.collect_info),
            )

        state, (_, states, info) = self._verbose_scan(
            one_step,
            state,
            xs,
            num_steps,
            stats_fn=lambda _carry, y: y[0],
        )
        trace = states if self.collect_state else None
        diagnostics = info if self.collect_info else None
        return MCMCResult(state, trace, diagnostics)

    def sample(
        self,
        key: RngKey,
        state: State,
        num_samples: int,
        params: Optional[Params] = None,
        thin: int = 1,
        args: Optional[Tuple] = None,
    ) -> MCMCResult:
        """Collect positions after every ``thin`` transitions."""
        params = self._prepare_params(state, params)
        if not self.verbose:
            return self.sample_kernel(
                key,
                self.kernel,
                state,
                num_samples,
                params,
                args,
                thin=thin,
                collect_info=self.collect_info,
            )

        keys = jax.random.split(key, (num_samples, thin))
        step_args = None
        if args is not None:
            step_args = jax.tree.map(
                lambda value: value.reshape((num_samples, thin) + value.shape[1:]),
                args,
            )
        xs = keys if step_args is None else (keys, step_args)

        def collect_one(state, xs):
            if step_args is None:
                sample_keys, sample_args = xs, None
            else:
                sample_keys, sample_args = xs
            inner_xs = (
                sample_keys if sample_args is None else (sample_keys, sample_args)
            )

            def transition(state, inner_xs):
                if sample_args is None:
                    step_key, current_args = inner_xs, ()
                else:
                    step_key, current_args = inner_xs
                state, info = self.kernel(step_key, state, params, *current_args)
                return state, (info, _select(info, self.collect_info))

            state, (step_info, selected_info) = jax.lax.scan(
                transition, state, inner_xs
            )
            last_info = jax.tree.map(lambda value: value[-1], step_info)
            diagnostics = jax.tree.map(lambda value: value[-1], selected_info)
            return state, (
                self._extract_stats(state, last_info),
                state.position,
                diagnostics,
            )

        state, (_, samples, info) = self._verbose_scan(
            collect_one,
            state,
            xs,
            num_samples,
            stats_fn=lambda _carry, y: y[0],
        )
        return MCMCResult(state, samples, info if self.collect_info else None)

    def adapt(
        self,
        key: RngKey,
        adaptor: Adaptor,
        state: State,
        params: Params,
        num_steps: int,
        args: Optional[Tuple] = None,
        *,
        collect: bool = False,
    ) -> AdaptationResult:
        """Fit kernel parameters with a composable adaptor."""
        return self.adaptor_warmup(
            key,
            self.kernel,
            adaptor,
            state,
            params,
            num_steps,
            args,
            collect=collect,
        )

    @staticmethod
    def warmup_kernel(
        key: RngKey,
        kernel: MarkovKernel,
        warmup: Warmup,
        state: State,
        params: Params,
        num_steps: int,
    ) -> WarmupResult:
        return warmup.run(key, kernel, state, params, num_steps)

    def warmup(
        self,
        key: RngKey,
        warmup: Warmup,
        state: State,
        params: Params,
        num_steps: int,
    ) -> WarmupResult:
        """Run a specialized state-producing warmup procedure."""
        return self.warmup_kernel(key, self.kernel, warmup, state, params, num_steps)

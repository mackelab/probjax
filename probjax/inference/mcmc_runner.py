from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax

from probjax.inference.adaptation import adapt_step, adaptor_warmup, as_warmup
from probjax.inference.base import (
    AdaptationPolicy,
    AdaptationResult,
    Adaptor,
    MCMCResult,
    RunnerMixin,
    Warmup,
    WarmupOutcome,
)
from probjax.inference.mcmc.base import (
    MarkovKernel,
    Params,
    State,
    _mcmc_collect_body,
    _reshape_thin_args,
    _select,
    sample_kernel,
)


def _mcmc_run_body(
    kernel: MarkovKernel,
    params: Params,
    args: Any,
    collect_state: tuple[str, ...],
    collect_info: tuple[str, ...],
    stats_fn: Callable[[Any, Any], Any] | None = None,
) -> Callable[[Any, Any], tuple[Any, tuple[Any, ...]]]:
    """Single-transition scan body shared by silent and verbose ``run``.

    Yields ``(stats, state_select, info_select)`` per transition, where
    ``stats`` is ``None`` when no ``stats_fn`` is supplied.
    """

    def one_step(state, xs):
        if args is None:
            step_key, step_args = xs, ()
        else:
            step_key, step_args = xs
        state, info = kernel(step_key, state, params, *step_args)
        selects = (
            _select(state, collect_state),
            _select(info, collect_info),
        )
        stats = None if stats_fn is None else stats_fn(state, info)
        return state, (stats, *selects)

    return one_step


class MCMC(RunnerMixin):
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
        tracked_stats: tuple[str, ...] | None = None,
        collect_state: tuple[str, ...] = (),
        collect_info: tuple[str, ...] = (),
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
        rng,
        kernel: MarkovKernel,
        state: State,
        num_steps: int,
        params: Params,
        args: Any = None,
        *,
        collect_state: tuple[str, ...] = (),
        collect_info: tuple[str, ...] = (),
    ) -> MCMCResult:
        """Run a compiled kernel scan without constructing a runner."""
        keys = jax.random.split(rng, num_steps)
        xs = keys if args is None else (keys, args)

        one_step = _mcmc_run_body(kernel, params, args, collect_state, collect_info)

        state, (_, states, info) = jax.lax.scan(one_step, state, xs)
        trace = states if collect_state else None
        diagnostics = info if collect_info else None
        return MCMCResult(state, trace, diagnostics)

    # Re-exposed from mcmc.base so kernels can sample without importing the runner.
    sample_kernel = staticmethod(sample_kernel)

    adaptor_warmup = staticmethod(adaptor_warmup)
    adapt_step = staticmethod(adapt_step)

    def _prepare_params(self, state: State, params: Params | None) -> Params:
        return self.kernel.init_params(state) if params is None else params

    def run(
        self,
        rng,
        state: State,
        num_steps: int,
        params: Params | None = None,
        args: Any = None,
    ) -> MCMCResult:
        """Run ``num_steps`` transitions with this runner's configuration."""
        params = self._prepare_params(state, params)
        if not self.verbose:
            return self.run_kernel(
                rng,
                self.kernel,
                state,
                num_steps,
                params,
                args,
                collect_state=self.collect_state,
                collect_info=self.collect_info,
            )

        keys = jax.random.split(rng, num_steps)
        xs = keys if args is None else (keys, args)

        one_step = _mcmc_run_body(
            self.kernel,
            params,
            args,
            self.collect_state,
            self.collect_info,
            stats_fn=self._extract_stats,
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
        rng,
        state: State,
        num_samples: int,
        params: Params | None = None,
        thin: int = 1,
        args: Any = None,
    ) -> MCMCResult:
        """Collect positions after every ``thin`` transitions."""
        params = self._prepare_params(state, params)
        if not self.verbose:
            return self.sample_kernel(
                rng,
                self.kernel,
                state,
                num_samples,
                params,
                args,
                thin=thin,
                collect_info=self.collect_info,
            )

        keys = jax.random.split(rng, (num_samples, thin))
        step_args = _reshape_thin_args(args, num_samples, thin)
        xs = keys if step_args is None else (keys, step_args)

        collect_one = _mcmc_collect_body(
            self.kernel,
            params,
            step_args,
            self.collect_info,
            stats_fn=self._extract_stats,
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
        rng,
        adaptor: Adaptor,
        state: State,
        params: Params,
        num_steps: int,
        args: Any = None,
        *,
        collect: bool = False,
    ) -> AdaptationResult:
        """Fit kernel parameters with a composable adaptor."""
        return self.adaptor_warmup(
            rng,
            self.kernel,
            adaptor,
            state,
            params,
            num_steps,
            args,
            collect=collect,
        )

    @staticmethod
    def _resolve_policy(policy: AdaptationPolicy) -> Warmup:
        """Accept a bare adaptor (auto-wrapped) or a warmup policy."""
        if isinstance(policy, Adaptor):
            return as_warmup(policy)
        if isinstance(policy, Warmup):
            return policy
        raise TypeError(
            f"adaptation must be an Adaptor or a Warmup, got {type(policy).__name__}."
        )

    @staticmethod
    def warmup_kernel(
        rng,
        kernel: MarkovKernel,
        adaptation: AdaptationPolicy,
        state: State,
        params: Params,
        num_steps: int,
    ) -> WarmupOutcome:
        return MCMC._resolve_policy(adaptation).run(
            rng, kernel, state, params, num_steps
        )

    def warmup(
        self,
        rng,
        adaptation: AdaptationPolicy | None = None,
        state: State | None = None,
        params: Params | None = None,
        num_steps: int | None = None,
        **adaptation_kwargs: Any,
    ) -> WarmupOutcome:
        """Run a specialized state-producing warmup procedure.

        An explicitly passed ``adaptation`` (an ``Adaptor`` or a ``Warmup``)
        always wins; ``adaptation=None`` uses the kernel's attached default
        (``kernel.default_adaptation``). ``adaptation_kwargs`` are forwarded
        to the default factory.
        """
        if state is None or params is None or num_steps is None:
            raise TypeError(
                "warmup() missing required arguments: state, params, num_steps"
            )
        policy: AdaptationPolicy
        if adaptation is None:
            default = self.kernel.default_adaptation
            if default is None:
                raise ValueError(
                    "This kernel has no default adaptation; "
                    "pass adaptation= explicitly."
                )
            policy = default(**adaptation_kwargs)
        else:
            if adaptation_kwargs:
                raise ValueError(
                    "adaptation_kwargs need adaptation=None (they configure the default)."
                )
            policy = adaptation
        return self.warmup_kernel(rng, self.kernel, policy, state, params, num_steps)

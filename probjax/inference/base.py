"""Shared types for the inference stack.

Kernels are pure transitions, adaptors fit their parameters and runners provide
the compiled execution paths. The aliases below describe those three roles so
that kernels, runners and user code agree on what a "state", an "info" or an
"adaptation policy" is.

Every algorithm API follows the same three steps:

1. an **API class** (:class:`~probjax.utils.jaxutils.KernelAPI`) holds
   ``init``, ``init_params`` and ``build_step``;
2. calling it with the model's functions builds a **kernel** -- a pure
   ``(init, step, ...)`` container that is callable and hashable;
3. a **runner** (:class:`MCMC`, :class:`SMC`) drives the kernel in a compiled
   scan.

Options that are known before the model is can be bound with
``api.configure(**options)``, which returns a
:class:`~probjax.utils.jaxutils.Configured` value; calling that with the
model's functions builds the kernel.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, NamedTuple, TypeAlias, TypeVar

from probjax.utils.jaxutils import WithProgressBarAPI, print_scan
from probjax.utils.typing import Array, ArrayTree, PyTree, RngKey

StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")
InfoT = TypeVar("InfoT")

# ---------------- Domain aliases ----------------
#: The chain position: any pytree of arrays.
Position: TypeAlias = ArrayTree
#: An unnormalized log density over a position. Extra arguments are allowed
#: (e.g. the pseudo-marginal estimator takes a random key).
LogDensityFn: TypeAlias = Callable[..., Array]
#: A single transition: ``(key, state, params, *args) -> (state, info)``.
StepFn: TypeAlias = Callable[..., tuple[Any, Any]]


class Kernel(NamedTuple):
    """A pure transition and the functions needed to initialize it.

    ``default_adaptation`` is an optional factory returning the kernel's
    recommended adaptation policy (an ``Adaptor`` or a ``Warmup``).
    Runners use it when no explicit policy is passed; an explicitly
    passed policy always wins.
    """

    init: Callable[..., Any]
    step: StepFn
    init_params: Callable[..., Any]
    default_adaptation: Callable[..., Any] | None = None

    def __call__(
        self,
        key: RngKey,
        state: Any,
        params: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        return self.step(key, state, params, *args, **kwargs)


class AdaptationTrace(NamedTuple):
    kernel: Any = None
    adaptor: Any = None


class AdaptationResult(NamedTuple, Generic[StateT, ParamsT, InfoT]):
    state: StateT
    params: ParamsT
    trace: PyTree | None = None
    final_info: InfoT | None = None


class WarmupResult(NamedTuple, Generic[StateT, ParamsT, InfoT]):
    state: StateT
    params: ParamsT
    info: InfoT | None = None


class Adaptor(NamedTuple):
    """A local parameter-adaptation state machine.

    ``init`` receives ``(state, params)``. ``update`` consumes one completed
    transition and can therefore be used during warmup or regular sampling.
    """

    init: Callable[[Any, Any], Any]
    update: Callable[[Any, Any, Any, Any], tuple[Any, Any, Any]]
    finalize: Callable[[Any, Any], tuple[Any, Any]]


class Warmup(NamedTuple):
    """A finite initialization policy for a kernel and its parameters."""

    run: Callable[..., "WarmupOutcome"]


#: Either kind of adaptation policy: a local ``Adaptor`` or a ``Warmup``.
AdaptationPolicy: TypeAlias = Adaptor | Warmup
#: What a warmup policy returns: a local adaptor yields an
#: ``AdaptationResult``, a specialized procedure a ``WarmupResult``.
WarmupOutcome: TypeAlias = AdaptationResult | WarmupResult


class MCMCResult(NamedTuple, Generic[StateT, InfoT]):
    state: StateT
    samples: PyTree | None = None
    info: InfoT | None = None


class SMCResult(NamedTuple, Generic[StateT, ParamsT, InfoT]):
    state: StateT
    params: ParamsT
    info: InfoT | None = None
    log_evidence: Array | None = None
    final_info: InfoT | None = None
    num_steps: int | Array | None = None
    completed: bool | Array | None = None
    key: RngKey | None = None


class FilteringResult(NamedTuple, Generic[StateT, InfoT]):
    initial_state: StateT
    states: PyTree
    info: InfoT | None = None


class RunnerMixin(WithProgressBarAPI):
    """Shared progress reporting for compiled inference scans."""

    def _verbose_scan(
        self,
        f: Callable[[Any, Any], tuple[Any, Any]],
        init: Any,
        xs: Any,
        length: int,
        stats_fn: Callable[[Any, Any], Any],
        unroll: int = 1,
    ) -> tuple[Any, Any]:
        """Run a scan with a rate-limited progress bar.

        ``print_scan`` requires a tuple carry, so the kernel state is wrapped
        in a one-element tuple. ``f`` receives the state itself and returns
        the new state together with the scan output.
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
            unroll=unroll,
            update_stats=update_stats,
            print_fn=print_fn,
            print_rate=print_rate,
        )
        return state, y

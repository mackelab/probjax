from typing import Any, Callable, NamedTuple, Optional


class Kernel(NamedTuple):
    """A pure transition and the functions needed to initialize it."""

    init: Callable
    step: Callable
    init_params: Callable

    def __call__(self, key, state, params=None, *args, **kwargs):
        return self.step(key, state, params, *args, **kwargs)


class Adaptor(NamedTuple):
    """A local parameter-adaptation state machine.

    ``init`` receives ``(state, params)``. ``update`` consumes one completed
    transition and can therefore be used during warmup or regular sampling.
    """

    init: Callable
    update: Callable
    finalize: Callable


class AdaptationResult(NamedTuple):
    state: Any
    params: Any
    trace: Optional[Any] = None
    final_info: Optional[Any] = None


class AdaptationTrace(NamedTuple):
    kernel: Optional[Any] = None
    adaptor: Optional[Any] = None


class Warmup(NamedTuple):
    """A finite initialization policy for a kernel and its parameters."""

    run: Callable


class WarmupResult(NamedTuple):
    state: Any
    params: Any
    info: Optional[Any] = None


class MCMCResult(NamedTuple):
    state: Any
    samples: Optional[Any] = None
    info: Optional[Any] = None


class SMCResult(NamedTuple):
    state: Any
    params: Any
    info: Optional[Any] = None
    log_evidence: Optional[Any] = None
    final_info: Optional[Any] = None
    num_steps: Optional[Any] = None
    completed: Optional[Any] = None
    key: Optional[Any] = None


class FilteringResult(NamedTuple):
    initial_state: Any
    states: Any
    info: Optional[Any] = None

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
3. a **runner** (:func:`~probjax.inference.run`) drives the
   kernel in a compiled scan.

Options that are known before the model is can be bound with
``api.configure(**options)``, which returns a
:class:`~probjax.utils.jaxutils.Configured` value; calling that with the
model's functions builds the kernel.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, TypeAlias

# ---------------- Domain aliases ----------------
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
        rng,
        state: Any,
        params: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        return self.step(rng, state, params, *args, **kwargs)


class Density(NamedTuple):
    """A parameterised distribution, as inference needs it.

    The parameters are an explicit argument, never closure state, so a kernel
    can carry them in its ``params`` and an adaptor can train them inside a
    compiled scan. :mod:`probjax.inference.neural` builds one from a
    :mod:`probjax.nn` model.

    Attributes:
        sample: ``(key, params) -> x``, one draw. Differentiable in ``params``
            when the model is reparameterisable (flows are).
        logpdf: ``(x, params) -> log q(x)`` for one point.
    """

    sample: Callable[[Any, Any], Any]
    logpdf: Callable[[Any, Any], Any]


class Adaptor(NamedTuple):
    """A local parameter-adaptation state machine.

    ``init`` receives ``(state, params)``. ``update`` consumes one completed
    transition and can therefore be used during warmup or regular sampling.
    """

    init: Callable[[Any, Any], Any]
    update: Callable[[Any, Any, Any, Any], tuple[Any, Any, Any]]
    finalize: Callable[[Any, Any], tuple[Any, Any]]


class Warmup(NamedTuple):
    """A finite initialization policy for a kernel and its parameters.

    ``run(key, kernel, state, params, num_steps)`` returns a
    :class:`RunResult` whose ``trace`` holds the policy's diagnostics.
    """

    run: Callable[..., "RunResult"]


class RunResult(NamedTuple):
    """The outcome of :func:`~probjax.inference.run` or a warmup.

    ``trace`` maps each collected name to its values stacked over the kept
    steps. A warmup policy stores its own diagnostics there instead.
    """

    state: Any
    params: Any
    trace: Any = None

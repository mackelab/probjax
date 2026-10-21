from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax

from probjax.inference.base import (
    AdaptationResult,
    AdaptationTrace,
    Adaptor,
    Kernel,
    Warmup,
)
from probjax.inference.smc.base import _ensure_param_batch, _params_to_dict
from probjax.utils import guards as _guards

__all__ = ["get_param", "replace_params"]


def get_param(params: Any, name: str) -> Any:
    """Read ``params[name]`` for mappings, else ``params.<name>``."""
    return _guards.get_field(params, name)


def replace_params(params: Any, **updates: Any) -> Any:
    """Replace fields in a NamedTuple, dataclass, or mapping parameter PyTree."""
    return _guards.replace_field(params, **updates)


def compose_adaptors(**adaptors: Adaptor) -> Adaptor:
    """Compose adaptors, threading parameter updates in declaration order."""
    if not adaptors:
        raise ValueError("At least one adaptor is required")
    names = tuple(adaptors)
    children = tuple(adaptors.values())

    def init(state: Any, params: Any) -> tuple[Any, ...]:
        return tuple(child.init(state, params) for child in children)

    def update(
        state: Any, info: Any, adaptor_states: tuple[Any, ...], params: Any
    ) -> tuple[tuple[Any, ...], Any, dict[str, Any]]:
        new_states = []
        child_info = {}
        for name, child, child_state in zip(
            names, children, adaptor_states, strict=True
        ):
            child_state, params, info_out = child.update(
                state, info, child_state, params
            )
            new_states.append(child_state)
            child_info[name] = info_out
        return tuple(new_states), params, child_info

    def finalize(
        adaptor_states: tuple[Any, ...], params: Any
    ) -> tuple[Any, dict[str, Any]]:
        child_info = {}
        for name, child, child_state in zip(
            names, children, adaptor_states, strict=True
        ):
            params, info_out = child.finalize(child_state, params)
            child_info[name] = info_out
        return params, child_info

    return Adaptor(init, update, finalize)


def _stateless_adaptor(tune_fn: Callable[..., dict[str, Any]]) -> Adaptor:
    """Adaptor that retunes shared-batched params from each transition.

    ``tune_fn(params_dict, state, info)`` computes the tuned parameter dict;
    init/finalize are no-ops. Used by the SMC tuning adaptors.
    """

    def init(_state: Any, _params: Any) -> tuple[Any, ...]:
        return ()

    def update(
        state: Any, info: Any, adaptor_state: Any, params: Any
    ) -> tuple[Any, Any, None]:
        tuned = tune_fn(_params_to_dict(params), state, info)
        return adaptor_state, _ensure_param_batch(tuned, shared=True), None

    def finalize(_adaptor_state: Any, params: Any) -> tuple[Any, None]:
        return params, None

    return Adaptor(init, update, finalize)


@partial(jax.jit, static_argnames=("kernel", "adaptor"))
def adapt_step(
    rng,
    kernel: Kernel,
    adaptor: Adaptor,
    state: Any,
    params: Any,
    adaptor_state: Any,
    *args: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Advance a kernel once and apply one local parameter-adaptation update."""
    state, kernel_info = kernel.step(rng, state, params, *args)
    adaptor_state, params, adaptor_info = adaptor.update(
        state, kernel_info, adaptor_state, params
    )
    return state, params, adaptor_state, kernel_info, adaptor_info


@partial(
    jax.jit,
    static_argnames=("kernel", "adaptor", "num_steps", "collect"),
)
def adaptor_warmup(
    rng,
    kernel: Kernel,
    adaptor: Adaptor,
    state: Any,
    params: Any,
    num_steps: int,
    args: Any = None,
    *,
    collect: bool = False,
) -> AdaptationResult:
    """Run a kernel while updating its parameters with an adaptor."""
    adaptor_state = adaptor.init(state, params)
    keys = jax.random.split(rng, num_steps)
    xs = keys if args is None else (keys, args)

    def one_step(carry, xs):
        state, params, adaptor_state = carry
        if args is None:
            step_key, step_args = xs, ()
        else:
            step_key, step_args = xs
        state, params, adaptor_state, kernel_info, adaptor_info = adapt_step(
            step_key,
            kernel,
            adaptor,
            state,
            params,
            adaptor_state,
            *step_args,
        )
        output = (kernel_info, adaptor_info) if collect else None
        return (state, params, adaptor_state), output

    (state, params, adaptor_state), info = jax.lax.scan(
        one_step, (state, params, adaptor_state), xs
    )
    params, final_info = adaptor.finalize(adaptor_state, params)
    trace = AdaptationTrace(*info) if info is not None else None
    return AdaptationResult(state, params, trace, final_info if collect else None)


def as_warmup(adaptor: Adaptor, *, collect: bool = False) -> Warmup:
    """Create the standard fixed-length warmup policy from a local adaptor."""

    def run(
        rng,
        kernel: Kernel,
        state: Any,
        params: Any,
        num_steps: int,
        args: Any = None,
    ) -> AdaptationResult:
        return adaptor_warmup(
            rng,
            kernel,
            adaptor,
            state,
            params,
            num_steps,
            args,
            collect=collect,
        )

    return Warmup(run)


# The generic fixed-length warmup executor retains its concise historical name.
adapt = adaptor_warmup

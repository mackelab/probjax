from collections.abc import Mapping
from functools import partial

import jax

from probjax.inference.base import (
    AdaptationResult,
    AdaptationTrace,
    Adaptor,
    Kernel,
)


def get_param(params, name):
    if isinstance(params, Mapping):
        if name not in params:
            raise ValueError(f"params must have a {name} field")
        return params[name]
    if not hasattr(params, name):
        raise ValueError(f"params must have a {name} field")
    return getattr(params, name)


def replace_params(params, **updates):
    """Replace fields in a NamedTuple, dataclass, or mapping parameter PyTree."""
    if isinstance(params, Mapping):
        return type(params)(params, **updates)
    if hasattr(params, "_replace"):
        return params._replace(**updates)
    try:
        from dataclasses import replace

        return replace(params, **updates)
    except TypeError as error:
        raise TypeError("params must be a mapping, NamedTuple, or dataclass") from error


def compose_adaptors(**adaptors) -> Adaptor:
    """Compose adaptors, threading parameter updates in declaration order."""
    if not adaptors:
        raise ValueError("At least one adaptor is required")
    names = tuple(adaptors)
    children = tuple(adaptors.values())

    def init(state, params):
        return tuple(child.init(state, params) for child in children)

    def update(state, info, adaptor_states, params):
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

    def finalize(adaptor_states, params):
        child_info = {}
        for name, child, child_state in zip(
            names, children, adaptor_states, strict=True
        ):
            params, info_out = child.finalize(child_state, params)
            child_info[name] = info_out
        return params, child_info

    return Adaptor(init, update, finalize)


@partial(jax.jit, static_argnames=("kernel", "adaptor"))
def adapt_step(
    key, kernel: Kernel, adaptor: Adaptor, state, params, adaptor_state, *args
):
    """Advance a kernel once and apply one local parameter-adaptation update."""
    state, kernel_info = kernel.step(key, state, params, *args)
    adaptor_state, params, adaptor_info = adaptor.update(
        state, kernel_info, adaptor_state, params
    )
    return state, params, adaptor_state, kernel_info, adaptor_info


@partial(
    jax.jit,
    static_argnames=("kernel", "adaptor", "num_steps", "collect"),
)
def adaptor_warmup(
    key,
    kernel: Kernel,
    adaptor: Adaptor,
    state,
    params,
    num_steps: int,
    args=None,
    *,
    collect: bool = False,
) -> AdaptationResult:
    """Run a kernel while updating its parameters with an adaptor."""
    adaptor_state = adaptor.init(state, params)
    keys = jax.random.split(key, num_steps)
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
    trace = AdaptationTrace(*info) if collect else None
    return AdaptationResult(state, params, trace, final_info if collect else None)


def as_warmup(adaptor: Adaptor, *, collect: bool = False):
    """Create the standard fixed-length warmup policy from a local adaptor."""
    from probjax.inference.base import Warmup

    def run(key, kernel, state, params, num_steps, args=None):
        return adaptor_warmup(
            key,
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

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from probjax.inference.base import Adaptor
from probjax.inference.smc.base import _ensure_param_batch
from probjax.utils import guards as _guards
from probjax.utils.guards import params_to_dict

__all__ = ["compose_adaptors", "get_param", "pooled", "replace_params"]


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
        tuned = tune_fn(params_to_dict(params), state, info)
        return adaptor_state, _ensure_param_batch(tuned, shared=True), None

    def finalize(_adaptor_state: Any, params: Any) -> tuple[Any, None]:
        return params, None

    return Adaptor(init, update, finalize)


def pooled(adaptor: Adaptor, axis_name: str = "chain") -> Adaptor:
    """Share one adaptation across chains run under ``jax.vmap``.

    Run the chains with ``jax.vmap(..., axis_name=axis_name)``. After every
    update each chain's floating-point adaptor statistics and parameters are
    replaced by their mean over chains, so all chains adapt as one. For dual
    averaging this equals adapting on the mean acceptance rate. Welford
    estimates measure deviations from the pooled mean, so they include the
    spread between chains; they start slightly low (the bias shrinks like
    ``log(n) / n`` after ``n`` updates).
    """

    def mean_over_chains(tree: Any) -> Any:
        def mean(value):
            if jnp.issubdtype(jnp.result_type(value), jnp.inexact):
                return jax.lax.pmean(value, axis_name)
            return value

        return jax.tree.map(mean, tree)

    def update(
        state: Any, info: Any, adaptor_state: Any, params: Any
    ) -> tuple[Any, Any, Any]:
        adaptor_state, params, adaptor_info = adaptor.update(
            state, info, adaptor_state, params
        )
        return mean_over_chains(adaptor_state), mean_over_chains(params), adaptor_info

    return Adaptor(adaptor.init, update, adaptor.finalize)

from typing import Any, Callable, Sequence

from jax.extend.core import Jaxpr, Var
from jaxtyping import Array

from probjax.core.jaxpr_propagation.engine import (
    identity_reducer,
    naive_cost_fn,
    run_jaxpr,
)
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule, ProcessingRule


def propagate(
    jaxpr: Jaxpr,
    consts: Sequence[Array],
    invars: Sequence[Var],
    inputs: Sequence[Array],
    outvars: Sequence[Var],
    process_eqn: ProcessingRule | Callable[..., Any] = ForwardProcessingRule(),
    cost_fn: Callable = naive_cost_fn,
    process_all_eqns: bool = False,
    reducer: Callable = identity_reducer,
    initial_state: Any = None,
    return_state: bool = False,
    return_env: bool = False,
    state_namespace: str = "default",
):
    return run_jaxpr(
        jaxpr,
        consts,
        invars,
        inputs,
        outvars,
        process_eqn=process_eqn,
        scheduler="priority",
        cost_fn=cost_fn,
        process_all_eqns=process_all_eqns,
        recurse_policy="missing_inputs",
        run_post_nested_process=True,
        reducer=reducer,
        initial_state=initial_state,
        return_state=return_state,
        return_env=return_env,
        state_namespace=state_namespace,
    )

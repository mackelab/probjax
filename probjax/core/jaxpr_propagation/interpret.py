from typing import Any, Callable, Sequence

from jax.extend.core import Jaxpr, Var
from jaxtyping import Array

from probjax.core.jaxpr_propagation.engine import identity_reducer, run_jaxpr
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule, ProcessingRule


def interpret(
    jaxpr: Jaxpr,
    consts: Sequence[Array],
    invars: Sequence[Var],
    inputs: Sequence[Array],
    outvars: Sequence[Var],
    process_eqn: ProcessingRule | Callable[..., Any] = ForwardProcessingRule(),
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
        scheduler="topological",
        recurse_policy="always",
        reducer=reducer,
        initial_state=initial_state,
        return_state=return_state,
        return_env=return_env,
        state_namespace=state_namespace,
    )

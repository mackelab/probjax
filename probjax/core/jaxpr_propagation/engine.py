from __future__ import annotations

import math
import weakref
from typing import (
    Any,
    Callable,
    Literal as TypingLiteral,
    Mapping,
    Optional,
    Sequence,
    cast,
)

from jax._src import core as jax_core
from jax._src.util import safe_map as map
from jax.extend.core import Jaxpr, JaxprEqn, Literal as JaxLiteral, Var
from jaxtyping import Array

from probjax.core.jaxpr_propagation.context import ExecutionContext
from probjax.core.jaxpr_propagation.extended import (
    EqnId,
    ExtendedEquation,
    ExtendedJaxpr,
)
from probjax.core.jaxpr_propagation.utils import (
    CostFunction,
    Environment,
    ForwardProcessingRule,
    ProcessingRule,
    ReducerFunction,
    RuleOutput,
    as_sequence,
    supports_context_argument,
)
from probjax.utils.containers import PriorityQueue

Scheduler = TypingLiteral["topological", "priority"]
RecursePolicy = TypingLiteral["always", "missing_inputs", "never"]
State = Any
CostFn = CostFunction
Reducer = ReducerFunction
ProcessResult = RuleOutput | None
ProcessEqn = ProcessingRule | Callable[..., ProcessResult]


def naive_cost_fn(
    eqn: JaxprEqn,
    is_known_invars: Sequence[bool],
    is_known_outvars: Sequence[bool],
) -> float:
    del eqn, is_known_outvars
    if all(is_known_invars):
        return 0.0
    return math.inf


def identity_reducer(
    env: Environment,
    eqn: JaxprEqn,
    state: State,
    eqn_state: State,
) -> State:
    del env, eqn, eqn_state
    return state


ProcessAdapter = Callable[
    [
        ExtendedEquation,
        Sequence[Optional[Array]],
        Sequence[Optional[Array]],
        ExecutionContext,
    ],
    ProcessResult,
]
ReducerAdapter = Callable[
    [Environment, ExtendedEquation, State, State, ExecutionContext], State
]

_PROCESS_ADAPTER_CACHE: weakref.WeakKeyDictionary[Any, ProcessAdapter] = (
    weakref.WeakKeyDictionary()
)
_REDUCER_ADAPTER_CACHE: weakref.WeakKeyDictionary[Any, ReducerAdapter] = (
    weakref.WeakKeyDictionary()
)


def _build_process_adapter(process_eqn: ProcessEqn) -> ProcessAdapter:
    process_fn = cast(Callable[..., ProcessResult], process_eqn)
    use_context = supports_context_argument(cast(Callable, process_eqn), 4)

    if use_context:

        def invoke(
            extended_eqn: ExtendedEquation,
            known_inputs: Sequence[Optional[Array]],
            known_outputs: Sequence[Optional[Array]],
            context: ExecutionContext,
        ) -> ProcessResult:
            return process_fn(extended_eqn.eqn, known_inputs, known_outputs, context)

    else:

        def invoke(
            extended_eqn: ExtendedEquation,
            known_inputs: Sequence[Optional[Array]],
            known_outputs: Sequence[Optional[Array]],
            context: ExecutionContext,
        ) -> ProcessResult:
            del context
            return process_fn(extended_eqn.eqn, known_inputs, known_outputs)

    return invoke


def _compile_process_adapter(process_eqn: ProcessEqn) -> ProcessAdapter:
    try:
        cached = _PROCESS_ADAPTER_CACHE.get(process_eqn)
    except TypeError:
        return _build_process_adapter(process_eqn)

    if cached is not None:
        return cached

    compiled = _build_process_adapter(process_eqn)
    _PROCESS_ADAPTER_CACHE[process_eqn] = compiled
    return compiled


def _build_reducer_adapter(reducer: Reducer) -> ReducerAdapter:
    reducer_fn = cast(Callable[..., State], reducer)
    use_context = supports_context_argument(cast(Callable, reducer), 5)

    if use_context:

        def invoke(
            env: Environment,
            extended_eqn: ExtendedEquation,
            state: State,
            eqn_state: State,
            context: ExecutionContext,
        ) -> State:
            return reducer_fn(env, extended_eqn.eqn, state, eqn_state, context)

    else:

        def invoke(
            env: Environment,
            extended_eqn: ExtendedEquation,
            state: State,
            eqn_state: State,
            context: ExecutionContext,
        ) -> State:
            del context
            return reducer_fn(env, extended_eqn.eqn, state, eqn_state)

    return invoke


def _compile_reducer_adapter(reducer: Reducer) -> ReducerAdapter:
    try:
        cached = _REDUCER_ADAPTER_CACHE.get(reducer)
    except TypeError:
        return _build_reducer_adapter(reducer)

    if cached is not None:
        return cached

    compiled = _build_reducer_adapter(reducer)
    _REDUCER_ADAPTER_CACHE[reducer] = compiled
    return compiled


def _can_use_eval_jaxpr_fast_path(
    jaxpr: Jaxpr,
    invars: Sequence[Var],
    outvars: Sequence[Var],
    process_eqn: ProcessEqn,
    scheduler: Scheduler,
    process_all_eqns: bool,
    recurse_policy: RecursePolicy,
    run_post_nested_process: bool,
    reducer: Reducer,
    initial_state: State,
    return_state: bool,
    return_env: bool,
) -> bool:
    if scheduler != "topological":
        return False
    if process_all_eqns:
        return False
    if run_post_nested_process:
        return False
    if return_state or return_env:
        return False
    if initial_state is not None:
        return False
    if reducer is not identity_reducer:
        return False
    if tuple(invars) != tuple(jaxpr.invars):
        return False
    if tuple(outvars) != tuple(jaxpr.outvars):
        return False

    return (
        isinstance(process_eqn, ForwardProcessingRule)
        and process_eqn.__class__ is ForwardProcessingRule
    )


def _get_closed_sub_jaxpr(eqn: Any):
    primitive_name = getattr(getattr(eqn, "primitive", None), "name", None)
    if primitive_name == "scan":
        return None

    candidate = eqn.params.get("jaxpr")
    if (
        candidate is not None
        and hasattr(candidate, "jaxpr")
        and hasattr(candidate, "consts")
    ):
        return candidate

    candidate = eqn.params.get("call_jaxpr")
    if (
        candidate is not None
        and hasattr(candidate, "jaxpr")
        and hasattr(candidate, "consts")
    ):
        return candidate

    return None


def _should_recurse(
    eqn: Any,
    recurse_policy: RecursePolicy,
    known_inputs: Sequence[Optional[Array]],
) -> bool:
    if recurse_policy == "never":
        return False
    if _get_closed_sub_jaxpr(eqn) is None:
        return False
    if recurse_policy == "always":
        return True
    return not all(v is not None for v in known_inputs)


def _parse_process_result(
    result: ProcessResult,
) -> tuple[Sequence[Any], Sequence[Any], State]:
    if result is None:
        return (), (), None

    if not isinstance(result, tuple):
        raise TypeError("Processing rules must return None or a tuple.")

    if len(result) == 2:
        outvars, outvals = result
        return as_sequence(outvars), as_sequence(outvals), None
    if len(result) == 3:
        outvars, outvals, eqn_state = result
        return as_sequence(outvars), as_sequence(outvals), eqn_state

    raise ValueError(
        "Processing rules must return (outvars, outvals) or "
        "(outvars, outvals, eqn_state)."
    )


def _run_process_rule(
    process_adapter: ProcessAdapter,
    extended_eqn: ExtendedEquation,
    known_inputs: Sequence[Optional[Array]],
    known_outputs: Sequence[Optional[Array]],
    context: ExecutionContext,
) -> tuple[Sequence[Any], Sequence[Any], State]:
    result = process_adapter(extended_eqn, known_inputs, known_outputs, context)
    return _parse_process_result(cast(ProcessResult, result))


def _run_nested(
    extended_eqn: ExtendedEquation,
    known_inputs: Sequence[Optional[Array]],
    known_outputs: Sequence[Optional[Array]],
    process_eqn: ProcessEqn,
    process_adapter: ProcessAdapter,
    scheduler: Scheduler,
    cost_fn: CostFn,
    process_all_eqns: bool,
    recurse_policy: RecursePolicy,
    run_post_nested_process: bool,
    reducer: Reducer,
    initial_state: State,
    state_namespace: str,
    context: ExecutionContext,
) -> tuple[Sequence[Any], Sequence[Any], State]:
    closed_sub_jaxpr = _get_closed_sub_jaxpr(extended_eqn)
    if closed_sub_jaxpr is None:
        return (), (), None

    sub_jaxpr = closed_sub_jaxpr.jaxpr
    sub_consts = closed_sub_jaxpr.consts

    nested_invars = tuple(sub_jaxpr.invars)
    nested_outvars = tuple(sub_jaxpr.outvars)
    nested_vars = nested_invars + nested_outvars
    outer_vars = tuple(extended_eqn.invars) + tuple(extended_eqn.outvars)
    known_vals = tuple(known_inputs) + tuple(known_outputs)

    known_sub_vars: list[Any] = []
    known_sub_vals: list[Any] = []
    target_sub_vars: list[Any] = []
    target_outer_vars: list[Any] = []

    for sub_var, outer_var, val in zip(
        nested_vars,
        outer_vars,
        known_vals,
        strict=False,
    ):
        if val is None:
            target_sub_vars.append(sub_var)
            target_outer_vars.append(outer_var)
        else:
            known_sub_vars.append(sub_var)
            known_sub_vals.append(val)

    target_vals: Sequence[Any] = ()
    nested_state: State = None
    if target_sub_vars:
        nested_result = cast(
            tuple[list[Any], State],
            run_jaxpr(
                sub_jaxpr,
                sub_consts,
                known_sub_vars,
                known_sub_vals,
                target_sub_vars,
                process_eqn=process_eqn,
                scheduler=scheduler,
                cost_fn=cost_fn,
                process_all_eqns=process_all_eqns,
                recurse_policy=recurse_policy,
                run_post_nested_process=run_post_nested_process,
                reducer=reducer,
                initial_state=initial_state,
                return_state=True,
                path_prefix=extended_eqn.eqn_id,
                state_namespace=state_namespace,
            ),
        )
        target_vals, nested_state = nested_result

    # Remap nested state from inner variables to outer variables
    # This is needed because the nested jaxpr has different Var objects
    # than the outer jaxpr, even for corresponding positions
    eqn_state: State = None
    if nested_state is not None:
        inner_to_outer = dict(zip(nested_vars, outer_vars, strict=False))
        remapped_state = {}
        for inner_var, val in nested_state.items():
            outer_var = inner_to_outer.get(inner_var)
            if outer_var is not None:
                remapped_state[outer_var] = val
        eqn_state = remapped_state if remapped_state else None
        # Store remapped state for post-processing rules to access
        context.set_transient_state(state_namespace, eqn_state)

    if run_post_nested_process and any(v is not None for v in known_outputs):
        _, _, post_state = _run_process_rule(
            process_adapter,
            extended_eqn,
            known_inputs,
            known_outputs,
            context,
        )
        if post_state is not None:
            eqn_state = post_state

    return target_outer_vars, target_vals, eqn_state


def _write_outputs(
    env: Environment,
    outvars: Sequence[Any],
    outvals: Sequence[Any],
) -> list[Any]:
    written_vars: list[Any] = []
    for var, val in zip(outvars, outvals, strict=False):
        env.write(var, val)
        written_vars.append(var)
    return written_vars


class _EquationQueue:
    def __init__(
        self,
        neighbors: Mapping[Any, tuple[EqnId, ...]],
        env: Environment,
        equations: Sequence[ExtendedEquation],
        cost_fn: CostFn,
    ):
        self.neighbors = neighbors
        self.env = env
        self.cost_fn = cost_fn
        self.equation_by_id = {
            extended_eqn.eqn_id: extended_eqn for extended_eqn in equations
        }

        self.processed_eqns: set[EqnId] = set()
        self.queue = PriorityQueue()

        self._initialize()

    def _initialize(self):
        seed_eqn_ids: set[EqnId] = set()
        for var in self.env:
            seed_eqn_ids.update(self.neighbors.get(var, ()))

        for extended_eqn in self.equation_by_id.values():
            if all(map(self.env.known, extended_eqn.invars)):
                seed_eqn_ids.add(extended_eqn.eqn_id)

        for eqn_id in seed_eqn_ids:
            self.push(eqn_id)

    def _compute_cost(self, eqn_id: EqnId) -> float:
        extended_eqn = self.equation_by_id[eqn_id]
        known_invars = map(self.env.known, extended_eqn.invars)
        known_outvars = map(self.env.known, extended_eqn.outvars)
        return self.cost_fn(extended_eqn.eqn, known_invars, known_outvars)

    def push(self, eqn_id: EqnId):
        if eqn_id in self.processed_eqns:
            return
        cost = self._compute_cost(eqn_id)
        if eqn_id in self.queue:
            self.queue.update_cost(eqn_id, cost)
            return
        self.queue.insert(eqn_id, cost)

    def pop(self) -> ExtendedEquation:
        eqn_id = self.queue.pop()
        self.processed_eqns.add(eqn_id)
        return self.equation_by_id[eqn_id]

    def is_empty(self) -> bool:
        return self.queue.is_empty()


def _write_equation_state(
    env: Environment,
    extended_eqn: ExtendedEquation,
    eqn_state: State,
    state_namespace: str,
) -> None:
    if eqn_state is None:
        return
    env.write_state(extended_eqn.eqn_id, eqn_state, namespace=state_namespace)


def _clean_up_dead_vars(
    extended_eqn: ExtendedEquation,
    env: Environment,
    last_used: Mapping[Any, EqnId | None],
    protected_vars: set[Any],
) -> None:
    invars = {var for var in extended_eqn.invars if not isinstance(var, JaxLiteral)}
    for var in invars:
        if var in protected_vars:
            continue
        if last_used.get(var) == extended_eqn.eqn_id:
            env.pop(var, None)


def _format_return(
    outputs: list[Optional[Array]],
    state: State,
    env: Environment,
    *,
    return_state: bool,
    return_env: bool,
):
    if return_state and return_env:
        return outputs, state, env
    if return_state:
        return outputs, state
    if return_env:
        return outputs, env
    return outputs


def run_jaxpr(
    jaxpr: Jaxpr,
    consts: Sequence[Array],
    invars: Sequence[Var],
    inputs: Sequence[Array],
    outvars: Sequence[Var],
    process_eqn: ProcessEqn = ForwardProcessingRule(),
    *,
    scheduler: Scheduler = "topological",
    cost_fn: CostFn = naive_cost_fn,
    process_all_eqns: bool = False,
    recurse_policy: RecursePolicy = "missing_inputs",
    run_post_nested_process: bool = False,
    reducer: Reducer = identity_reducer,
    initial_state: State = None,
    return_state: bool = False,
    return_env: bool = False,
    path_prefix: EqnId = (),
    state_namespace: str = "default",
    extended_jaxpr: ExtendedJaxpr | None = None,
):
    if _can_use_eval_jaxpr_fast_path(
        jaxpr,
        invars,
        outvars,
        process_eqn,
        scheduler,
        process_all_eqns,
        recurse_policy,
        run_post_nested_process,
        reducer,
        initial_state,
        return_state,
        return_env,
    ):
        return jax_core.eval_jaxpr(jaxpr, consts, *inputs)

    env = Environment()
    map(env.write, jaxpr.constvars, consts)
    map(env.write, invars, inputs)

    extended = extended_jaxpr or ExtendedJaxpr.from_jaxpr(
        jaxpr,
        path_prefix=path_prefix,
    )
    context = ExecutionContext(
        env=env,
        extended_jaxpr=extended,
        scheduler=scheduler,
        state_namespace=state_namespace,
    )

    process_adapter = _compile_process_adapter(process_eqn)
    reducer_adapter = _compile_reducer_adapter(reducer)
    state = initial_state
    if state is not None:
        context.write_run_state(state, namespace=state_namespace)
    protected_outvars = {var for var in outvars if not isinstance(var, JaxLiteral)}

    if scheduler == "topological":
        for extended_eqn in extended.equations:
            context.set_current_equation(extended_eqn.eqn_id)
            known_inputs = map(env.read, extended_eqn.invars)
            known_outputs = map(env.read, extended_eqn.outvars)

            if _should_recurse(extended_eqn, recurse_policy, known_inputs):
                output_vars, output_vals, eqn_state = _run_nested(
                    extended_eqn,
                    known_inputs,
                    known_outputs,
                    process_eqn,
                    process_adapter,
                    scheduler,
                    cost_fn,
                    process_all_eqns,
                    recurse_policy,
                    run_post_nested_process,
                    reducer,
                    initial_state,
                    state_namespace,
                    context,
                )
            else:
                output_vars, output_vals, eqn_state = _run_process_rule(
                    process_adapter,
                    extended_eqn,
                    known_inputs,
                    known_outputs,
                    context,
                )

            _write_outputs(env, output_vars, output_vals)
            _write_equation_state(env, extended_eqn, eqn_state, state_namespace)
            state = reducer_adapter(env, extended_eqn, state, eqn_state, context)
            context.write_run_state(state, namespace=state_namespace)
            _clean_up_dead_vars(
                extended_eqn,
                env,
                extended.last_used,
                protected_outvars,
            )

            if not process_all_eqns and all(map(env.known, outvars)):
                break

        outputs = map(env.read, outvars)
        return _format_return(
            outputs,
            state,
            env,
            return_state=return_state,
            return_env=return_env,
        )

    equation_queue = _EquationQueue(
        extended.neighbors, env, extended.equations, cost_fn
    )

    while not equation_queue.is_empty():
        extended_eqn = equation_queue.pop()
        context.set_current_equation(extended_eqn.eqn_id)

        known_inputs = map(env.read, extended_eqn.invars)
        known_outputs = map(env.read, extended_eqn.outvars)

        if _should_recurse(extended_eqn, recurse_policy, known_inputs):
            output_vars, output_vals, eqn_state = _run_nested(
                extended_eqn,
                known_inputs,
                known_outputs,
                process_eqn,
                process_adapter,
                scheduler,
                cost_fn,
                process_all_eqns,
                recurse_policy,
                run_post_nested_process,
                reducer,
                initial_state,
                state_namespace,
                context,
            )
        else:
            output_vars, output_vals, eqn_state = _run_process_rule(
                process_adapter,
                extended_eqn,
                known_inputs,
                known_outputs,
                context,
            )

        written_vars = _write_outputs(env, output_vars, output_vals)
        _write_equation_state(env, extended_eqn, eqn_state, state_namespace)
        state = reducer_adapter(env, extended_eqn, state, eqn_state, context)
        context.write_run_state(state, namespace=state_namespace)

        for var in written_vars:
            eqn_ids = extended.neighbors.get(var, ())
            for eqn_id in eqn_ids:
                equation_queue.push(eqn_id)

        if not process_all_eqns and all(map(env.known, outvars)):
            break

    outputs = map(env.read, outvars)
    return _format_return(
        outputs,
        state,
        env,
        return_state=return_state,
        return_env=return_env,
    )

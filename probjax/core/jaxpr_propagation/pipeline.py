from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from probjax.core.jaxpr_propagation.context import ExecutionContext

ProcessResult = (
    tuple[Sequence[Any | None], Sequence[Any | None]]
    | tuple[Sequence[Any | None], Sequence[Any | None], Any]
    | None
)


@dataclass(frozen=True, slots=True)
class NamespacedStatePayload(Mapping[str, Any]):
    """Compact, allocation-light namespaced equation state payload."""

    names: tuple[str, ...]
    state_values: tuple[Any, ...]
    name_to_index: dict[str, int]

    def __getitem__(self, key: str) -> Any:
        return self.state_values[self.name_to_index[key]]

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def get(self, key: str, default: Any = None) -> Any:
        index = self.name_to_index.get(key)
        if index is None:
            return default
        return self.state_values[index]


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _supports_context_argument(func: Callable) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    return has_varargs or len(positional_params) >= 4


def _compile_rule(
    rule: Callable,
) -> Callable[[Any, Any, Any, ExecutionContext | None], ProcessResult]:
    use_context = _supports_context_argument(rule)
    if use_context:

        def call(equation, known_inputs, known_outputs, context):
            return rule(equation, known_inputs, known_outputs, context)

    else:

        def call(equation, known_inputs, known_outputs, context):
            del context
            return rule(equation, known_inputs, known_outputs)

    return call


def _parse_result(result: ProcessResult):
    if result is None:
        return (), (), None
    if len(result) == 2:
        outvars, outvals = result
        return _as_sequence(outvars), _as_sequence(outvals), None
    outvars, outvals, eqn_state = result
    return _as_sequence(outvars), _as_sequence(outvals), eqn_state


@dataclass(frozen=True)
class InterpreterSpec:
    name: str
    rule: Callable
    observe_only: bool = False
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CompiledInterpreter:
    name: str
    call: Callable[[Any, Any, Any, ExecutionContext | None], ProcessResult]
    observe_only: bool
    required_indices: tuple[int, ...]


class InterpreterPipeline:
    """Composable interpreter wrapper.

    It executes interpreter rules in order and returns equation state namespaced
    by interpreter name.
    """

    def __init__(self, interpreters: Sequence[InterpreterSpec]) -> None:
        if not interpreters:
            raise ValueError("InterpreterPipeline requires at least one interpreter.")

        names = [interpreter.name for interpreter in interpreters]
        if len(set(names)) != len(names):
            raise ValueError("Interpreter names in a pipeline must be unique.")

        name_to_position = {name: idx for idx, name in enumerate(names)}
        compiled: list[_CompiledInterpreter] = []
        for interpreter in interpreters:
            required_indices: list[int] = []
            for dependency in interpreter.requires:
                if dependency not in name_to_position:
                    raise ValueError(
                        f"Interpreter '{interpreter.name}' depends on unknown '{dependency}'."
                    )
                dep_index = name_to_position[dependency]
                cur_index = name_to_position[interpreter.name]
                if dep_index >= cur_index:
                    raise ValueError(
                        f"Interpreter '{interpreter.name}' requires '{dependency}' to run first."
                    )
                required_indices.append(dep_index)

            compiled.append(
                _CompiledInterpreter(
                    name=interpreter.name,
                    call=_compile_rule(interpreter.rule),
                    observe_only=interpreter.observe_only,
                    required_indices=tuple(required_indices),
                )
            )

        self._compiled = tuple(compiled)
        self._names = tuple(names)
        self._name_to_index = {name: idx for idx, name in enumerate(self._names)}

    def __call__(
        self,
        equation: Any,
        known_inputs,
        known_outputs,
        context: ExecutionContext | None = None,
    ) -> ProcessResult:
        selected_outputs = None
        states: list[Any] = [None] * len(self._compiled)

        for index, interpreter in enumerate(self._compiled):
            for dep_index in interpreter.required_indices:
                if states[dep_index] is None:
                    dep_name = self._names[dep_index]
                    raise ValueError(
                        f"Interpreter '{interpreter.name}' missing dependency state '{dep_name}'."
                    )

            result = interpreter.call(equation, known_inputs, known_outputs, context)
            outvars, outvals, eqn_state = _parse_result(result)

            if selected_outputs is None and outvars and not interpreter.observe_only:
                selected_outputs = (outvars, outvals)

            states[index] = eqn_state
            if context is not None:
                context.set_transient_state(interpreter.name, eqn_state)

        if selected_outputs is None:
            return None

        namespaced_state = NamespacedStatePayload(
            names=self._names,
            state_values=tuple(states),
            name_to_index=self._name_to_index,
        )
        return selected_outputs[0], selected_outputs[1], namespaced_state

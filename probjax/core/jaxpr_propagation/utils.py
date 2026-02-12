from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple

from jax.extend.core import JaxprEqn, Literal
from jaxtyping import Array

# High level API


class Environment(dict):
    """A compute environment that stores intermediate computations."""

    def __init__(self):
        super().__init__()
        self.eqn_states: dict[Any, Any] = {}
        self.eqn_states_by_namespace: dict[Any, dict[str, Any]] = {}
        self.run_states: dict[str, Any] = {}

    def __getitem__(self, var: Any) -> Optional[Array]:
        if isinstance(var, Literal):
            return var.val
        elif var in self:
            return super().__getitem__(var)
        else:
            return None

    def __setitem__(self, var: Any, val: Array | None) -> None:
        if not isinstance(var, Literal):
            super().__setitem__(var, val)

    def read(self, var: Any) -> Array | None:
        return self[var]

    def write(self, var: Any, val: Array | None) -> None:
        self[var] = val

    def known(self, var: Any) -> bool:
        return isinstance(var, Literal) or var in self

    def read_state(self, eqn: Any, namespace: str | None = None) -> Any:
        if namespace is None:
            value = self.eqn_states.get(eqn)
            if value is not None:
                return value
            if isinstance(eqn, int):
                return self.eqn_states.get((eqn,))
            return None

        value = self.eqn_states_by_namespace.get(eqn, {}).get(namespace)
        if value is not None:
            return value
        if isinstance(eqn, int):
            return self.eqn_states_by_namespace.get((eqn,), {}).get(namespace)
        return None

    def write_state(
        self,
        eqn: Any,
        state: Any,
        namespace: str | None = None,
    ) -> None:
        if namespace is None:
            namespace = "default"
        self.eqn_states[eqn] = state
        states = self.eqn_states_by_namespace.get(eqn)
        if states is None:
            self.eqn_states_by_namespace[eqn] = {namespace: state}
        else:
            states[namespace] = state

    def read_run_state(self, namespace: str | None = None) -> Any:
        if namespace is None:
            namespace = "default"
        return self.run_states.get(namespace)

    def write_run_state(self, state: Any, namespace: str | None = None) -> None:
        if namespace is None:
            namespace = "default"
        self.run_states[namespace] = state


RuleOutput = (
    Tuple[Sequence[Any | None], Sequence[Any | None]]
    | Tuple[Sequence[Any | None], Sequence[Any | None], Any]
)


class ProcessingRule(ABC):
    """A processing rule for equations."""

    def __init__(self, propagator: Callable | None = None):
        self.propagator = propagator

    @abstractmethod
    def __call__(
        self,
        eqn: JaxprEqn,
        known_inputs: Sequence[Any | None],
        known_outputs: Sequence[Any | None],
    ) -> RuleOutput | None:
        pass


class ForwardProcessingRule(ProcessingRule):
    def __call__(
        self,
        eqn: JaxprEqn,
        known_inputs: Sequence[Array | None],
        _: Sequence[Array | None],
    ) -> RuleOutput:
        # assert (
        #     (known_inputs != None) and (None not in known_inputs)
        # ), "All inputs must be known for the forward pass."
        primitive = eqn.primitive
        subfuns, bind_params = primitive.get_bind_params(eqn.params)
        # `bind` is how a primitive is called
        outvals = primitive.bind(*subfuns, *known_inputs, **bind_params)
        # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:
            outvals = [outvals]

        return eqn.outvars, outvals  # type: ignore


# Helper utilities intentionally kept minimal; runtime graph construction lives
# in `extended.py` and execution logic lives in `engine.py`.

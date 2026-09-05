from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from probjax.core.jaxpr_propagation.extended import EqnId, ExtendedJaxpr
from probjax.core.jaxpr_propagation.utils import Environment


@dataclass
class ExecutionContext:
    """Per-run execution context used by interpreters and reducers."""

    env: Environment
    extended_jaxpr: ExtendedJaxpr
    scheduler: str
    state_namespace: str
    metadata: dict[str, Any] = field(default_factory=dict)
    current_eqn_id: EqnId | None = None
    _transient_eqn_states: dict[str, Any] = field(default_factory=dict)

    def set_current_equation(self, eqn_id: EqnId) -> None:
        self.current_eqn_id = eqn_id
        self._transient_eqn_states = {}

    def set_transient_state(self, namespace: str, state: Any) -> None:
        self._transient_eqn_states[namespace] = state

    def read_transient_state(self, namespace: str, default: Any = None) -> Any:
        return self._transient_eqn_states.get(namespace, default)

    def read_eqn_state(
        self,
        eqn: Any,
        namespace: str | None = None,
    ) -> Any:
        return self.env.read_state(eqn, namespace=namespace)

    def write_run_state(self, state: Any, namespace: str | None = None) -> None:
        self.env.write_run_state(state, namespace=namespace)

    def read_run_state(self, namespace: str | None = None) -> Any:
        return self.env.read_run_state(namespace=namespace)

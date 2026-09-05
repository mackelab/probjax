from typing import Any, Sequence

from jax.extend.core import JaxprEqn
from jaxtyping import Array

from probjax.core.custom_primitives.contracts import parse_random_variable_call_params
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule
from probjax.core.registry import ProcessedResult


class IntervenedProcessingRule(ForwardProcessingRule):
    def __init__(self, interventions: dict[str, Array]) -> None:
        """Fix random-variable outputs to provided intervention values.

        Args:
            interventions (dict[str, Array]): Name-to-value intervention map.
        """
        self.interventions = dict(interventions)

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> ProcessedResult:
        if eqn.primitive is rv_p:
            name = parse_random_variable_call_params(eqn.params).name
            if name in self.interventions:
                return ProcessedResult(eqn.outvars, [self.interventions[name]])

        return super().__call__(eqn, known_inputs, _)

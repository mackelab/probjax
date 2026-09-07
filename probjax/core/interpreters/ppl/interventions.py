from typing import Any, Sequence

from jax.extend.core import JaxprEqn
from jaxtyping import Array

from probjax.core.interpreters.ppl._common import rv_site_name
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
        name = rv_site_name(eqn)
        if name is not None and name in self.interventions:
            return ProcessedResult(eqn.outvars, [self.interventions[name]])

        return super().__call__(eqn, known_inputs, _)

from typing import Any, Sequence, Tuple

from jax.extend.core import JaxprEqn
from jaxtyping import Array

from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule


class IntervenedProcessingRule(ForwardProcessingRule):
    def __init__(self, interventions: dict[str, Array]) -> None:
        """Fix random-variable outputs to provided intervention values.

        Args:
            interventions (dict[str, Array]): Name-to-value intervention map.
        """
        self.interventions = dict(interventions)

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None]]:
        if eqn.primitive is rv_p:
            name = eqn.params["name"]
            if name in self.interventions:
                return eqn.outvars, [self.interventions[name]]

        return super().__call__(eqn, known_inputs, _)

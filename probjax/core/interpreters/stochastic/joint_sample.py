from typing import Any, Iterable, Optional, Sequence, Tuple

from jaxtyping import Array

from jax.extend.core import JaxprEqn

from probjax.core.custom_primitives.contracts import parse_random_variable_call_params
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule


class JointSampleProcessingRule(ForwardProcessingRule):
    def __init__(
        self,
        rvs: Optional[Iterable] = None,
        interventions: Optional[dict[str, Array] | Sequence[str]] = None,
    ) -> None:
        """Subset of random variables to be sampled jointly. By default all are sampled!

        Args:
            rvs (Optional[Iterable], optional): Subset of random variable names.
                Defaults to None.
        """
        self.rvs = rvs

        if interventions is None:
            self.interventions = set()
        elif isinstance(interventions, dict):
            self.interventions = set(interventions.keys())
        else:
            self.interventions = set(interventions)

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None], dict[str, Any]]:
        result = super().__call__(eqn, known_inputs, _)
        outvars, outvals = result[0], result[1]
        eqn_state = {}
        if eqn.primitive is rv_p:
            rv_params = parse_random_variable_call_params(eqn.params)
            name = rv_params.name
            intervened = eqn.params.get("intervened", False) or name in self.interventions
            if not intervened and (self.rvs is None or name in self.rvs):
                eqn_state[name] = outvals[0]
        return outvars, outvals, eqn_state


def joint_sample_state_reducer(env, eqn, state, eqn_state, context=None):
    del env, eqn, context
    if state is None:
        state = {}
    if not eqn_state:
        return state
    return state | eqn_state

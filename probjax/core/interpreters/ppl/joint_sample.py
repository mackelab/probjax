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
        fixed_values: Optional[dict[str, Array]] = None,
        fixed_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Subset of random variables to be sampled jointly. By default all are sampled!

        Args:
            rvs (Optional[Iterable], optional): Subset of random variable names.
                Defaults to None.
        """
        self.rvs = rvs
        self.fixed_values = {} if fixed_values is None else dict(fixed_values)
        fixed_set = set(self.fixed_values.keys())
        if fixed_names is not None:
            fixed_set.update(fixed_names)
        self.fixed_names = fixed_set

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None], dict[str, Any]]:
        result = super().__call__(eqn, known_inputs, _)
        outvars, outvals = result[0], result[1]
        eqn_state = {}
        if eqn.primitive is rv_p:
            rv_params = parse_random_variable_call_params(eqn.params)
            name = rv_params.name
            if name in self.fixed_values:
                return outvars, [self.fixed_values[name]], eqn_state

            fixed = eqn.params.get("intervened", False) or name in self.fixed_names
            if not fixed and (self.rvs is None or name in self.rvs):
                eqn_state[name] = outvals[0]
        return outvars, outvals, eqn_state


def joint_sample_state_reducer(env, eqn, state, eqn_state, context=None):
    del env, eqn, context
    if state is None:
        state = {}
    if not eqn_state:
        return state
    return state | eqn_state

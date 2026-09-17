from typing import Any, Iterable, Optional, Sequence

from jax.extend.core import JaxprEqn
from jaxtyping import Array

from probjax.core.interpreters.ppl._common import rv_site_name
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule, merge_dict_state
from probjax.core.registry import ProcessedResult


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
    ) -> ProcessedResult:
        result = super().__call__(eqn, known_inputs, _)
        outvars, outvals = result.resolved_vars, list(result.resolved_vals)
        eqn_state: dict[str, Any] = {}
        name = rv_site_name(eqn)
        if name is not None:
            if name in self.fixed_values:
                return ProcessedResult(outvars, [self.fixed_values[name]], eqn_state)

            fixed = eqn.params.get("intervened", False) or name in self.fixed_names
            if not fixed and (self.rvs is None or name in self.rvs):
                eqn_state[name] = outvals[0]
        return ProcessedResult(outvars, outvals, eqn_state)


def joint_sample_state_reducer(env, eqn, state, eqn_state, context=None):
    return merge_dict_state(env, eqn, state, eqn_state, context)

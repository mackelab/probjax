from typing import Any, Iterable, Optional, Sequence, Tuple

from jax.extend.core import JaxprEqn

from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule


class TraceProcessingRule(ForwardProcessingRule):
    def __init__(self, traced_vars: Optional[Iterable] = None) -> None:
        """Subset of random variables to be sampled jointly. By default all are sampled!

        Args:
            rvs (Optional[Iterable], optional): Subset of random variable names.
                Defaults to None.
        """
        self.traced_vars = traced_vars

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None], dict[str, Any]]:
        result = super().__call__(eqn, known_inputs, _)
        outvars, outvals = result[0], result[1]
        eqn_state = {}
        for o, v in zip(outvars, outvals, strict=False):
            if self.traced_vars is None or str(o) in self.traced_vars:
                eqn_state[str(o)] = v
        return outvars, outvals, eqn_state


def trace_state_reducer(env, eqn, state, eqn_state, context=None):
    del env, eqn, context
    if state is None:
        state = {}
    if not eqn_state:
        return state
    return state | eqn_state

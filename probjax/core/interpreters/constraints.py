# TODO propagate constraints
from typing import Any, Iterable, Sequence, Tuple

from jax import lax
from jax.extend.core import JaxprEqn

from probjax.core.interpreters.trace import TraceProcessingRule
from probjax.stats.constraints import (
    Negative,
    Positive,
    Real,
    Square,
    UnitInterval,
)

_UNIVARIATE_CONSTRAINTS = {
    lax.tanh_p: (Real, UnitInterval),
    lax.erf_p: (Real, UnitInterval),
    lax.exp_p: (Real, Positive),
    lax.log_p: (Positive, Real),
    lax.sin_p: (Real, UnitInterval),
    lax.cos_p: (Real, UnitInterval),
    lax.tan_p: (Real, UnitInterval),
}


class ConstraintTraceProcessingRule(TraceProcessingRule):
    def __init__(self, init_constraints, traced_vars: Iterable | None = None) -> None:
        super().__init__(traced_vars)
        self.invar_constraints = init_constraints

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> Tuple[Sequence[Any | None], Sequence[Any | None]]:
        outvars, outvals = super().__call__(eqn, known_inputs, _)
        # primitive = eqn.primitive
        # in_constraints = [self.traced_samples[str(i)] for i in eqn.invars]
        for o, v in zip(outvars, outvals):
            if self.traced_vars is None or str(o) in self.traced_vars:
                self.traced_samples[str(o)] = v
        return outvars, outvals

    def _default_processing_rule(primitive, in_constraint, outvars):
        pass


def _check_constraint(value, constraint):
    if constraint == Real:
        return True
    elif constraint == UnitInterval:
        return 0 <= value <= 1
    elif constraint == Positive:
        return value > 0
    elif constraint == Negative:
        return value < 0
    elif constraint == Square:
        return value.shape[-1] == value.shape[-2]
    else:
        return True

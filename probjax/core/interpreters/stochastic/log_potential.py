from typing import Dict, Sequence

from jax.extend.core import JaxprEqn
from jaxtyping import Array

from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule


class LogPotentialProcessingRule(ForwardProcessingRule):
    # Here we compute per-equation log-probability contributions.
    joint_samples: Dict[str, Array]

    def __init__(
        self,
        joint_samples: Dict[str, Array],
        interventions: Dict[str, Array] | Sequence[str] | None = None,
    ):
        self.joint_samples = joint_samples
        if interventions is None:
            self.intervened_names = set()
        elif isinstance(interventions, dict):
            self.intervened_names = set(interventions.keys())
        else:
            self.intervened_names = set(interventions)

    def __call__(
        self,
        eqn: JaxprEqn,
        in_known: Sequence[Array | None],
        out_known: Sequence[Array | None],
    ):
        if eqn.primitive is rv_p:
            # We do not have to sample -> Already given
            name = eqn.params["name"]
            intervened = (
                eqn.params.get("intervened", False) or name in self.intervened_names
            )
            if not intervened:
                if name not in self.joint_samples:
                    raise KeyError(
                        f"Missing joint sample for random variable '{name}'."
                    )
                outvars = eqn.outvars
                outvals = [self.joint_samples[name]]

                in_known_values = list(in_known)
                in_known_values[0] = outvals[0]
                log_pdf_fn = eqn.params["logpdf_fn"]
                eqn_state = log_pdf_fn(*in_known_values)
            else:
                result = super().__call__(eqn, in_known, out_known)
                outvars, outvals = result[0], result[1]
                eqn_state = None
        else:
            result = super().__call__(eqn, in_known, out_known)
            outvars, outvals = result[0], result[1]
            eqn_state = None

        return outvars, outvals, eqn_state


def log_potential_state_reducer(env, eqn, state, eqn_state, context=None):
    del env, eqn, context
    if state is None:
        state = 0.0
    if eqn_state is None:
        return state
    return state + eqn_state

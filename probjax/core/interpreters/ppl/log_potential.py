from typing import Dict, Mapping, Sequence

from jax.extend.core import JaxprEqn
from jaxtyping import Array

from probjax.core.custom_primitives.contracts import parse_random_variable_call_params
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule
from probjax.core.registry import ProcessedResult


class LogPotentialProcessingRule(ForwardProcessingRule):
    # Here we compute per-equation log-probability contributions.
    joint_samples: Dict[str, Array]

    def __init__(
        self,
        joint_samples: Dict[str, Array],
        interventions: Mapping[str, Array] | Sequence[str] | None = None,
        observations: Mapping[str, Array] | None = None,
        replay: Mapping[str, Array] | None = None,
        strict: bool = True,
        allow_partial: bool = False,
    ):
        self.joint_samples = dict(joint_samples)
        self.strict = bool(strict)
        self.allow_partial = bool(allow_partial)

        if interventions is None:
            self.intervention_values: Dict[str, Array] = {}
            self.intervened_names = set()
        elif isinstance(interventions, Mapping):
            self.intervention_values = dict(interventions)
            self.intervened_names = set(self.intervention_values.keys())
        else:
            self.intervention_values = {}
            self.intervened_names = set(interventions)

        self.observation_values = {} if observations is None else dict(observations)
        self.replay_values = {} if replay is None else dict(replay)

    @staticmethod
    def _logpdf_from_inputs(logpdf_fn, in_known, value):
        in_known_values = list(in_known)
        in_known_values[0] = value
        return logpdf_fn(*in_known_values)

    def __call__(
        self,
        eqn: JaxprEqn,
        in_known: Sequence[Array | None],
        out_known: Sequence[Array | None],
    ) -> ProcessedResult:
        if eqn.primitive is rv_p:
            rv_params = parse_random_variable_call_params(eqn.params)
            name = rv_params.name

            if name in self.intervention_values:
                return ProcessedResult(
                    eqn.outvars, [self.intervention_values[name]], None
                )

            if eqn.params.get("intervened", False) or name in self.intervened_names:
                result = super().__call__(eqn, in_known, out_known)
                return ProcessedResult(
                    result.resolved_vars, list(result.resolved_vals), None
                )

            if name in self.observation_values:
                value = self.observation_values[name]
                eqn_state = self._logpdf_from_inputs(
                    rv_params.logpdf_fn, in_known, value
                )
                return ProcessedResult(eqn.outvars, [value], eqn_state)

            if name in self.replay_values:
                value = self.replay_values[name]
                eqn_state = self._logpdf_from_inputs(
                    rv_params.logpdf_fn, in_known, value
                )
                return ProcessedResult(eqn.outvars, [value], eqn_state)

            if name in self.joint_samples:
                value = self.joint_samples[name]
                eqn_state = self._logpdf_from_inputs(
                    rv_params.logpdf_fn, in_known, value
                )
                return ProcessedResult(eqn.outvars, [value], eqn_state)

            if self.strict and not self.allow_partial:
                raise KeyError(f"Missing joint sample for random variable '{name}'.")

            result = super().__call__(eqn, in_known, out_known)
            return ProcessedResult(
                result.resolved_vars, list(result.resolved_vals), None
            )

        result = super().__call__(eqn, in_known, out_known)
        return ProcessedResult(result.resolved_vars, list(result.resolved_vals), None)


def log_potential_state_reducer(env, eqn, state, eqn_state, context=None):
    del env, eqn, context
    if state is None:
        state = 0.0
    if eqn_state is None:
        return state
    return state + eqn_state

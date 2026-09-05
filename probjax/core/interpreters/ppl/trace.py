from typing import Any, Iterable, Mapping, Optional, Sequence

from jax.extend.core import JaxprEqn

from probjax.core.custom_primitives.contracts import parse_random_variable_call_params
from probjax.core.custom_primitives.random_variable import rv_p
from probjax.core.jaxpr_propagation.utils import ForwardProcessingRule
from probjax.core.registry import ProcessedResult


class TraceProcessingRule(ForwardProcessingRule):
    def __init__(
        self,
        traced_vars: Optional[Iterable] = None,
        *,
        sites: bool = False,
        interventions: Mapping[str, Any] | None = None,
        observations: Mapping[str, Any] | None = None,
        replay: Mapping[str, Any] | None = None,
        kind_labels: str = "probabilistic",
    ) -> None:
        """Trace interpreter rule.

        Args:
            traced_vars: Optional subset of variables/sites to trace.
            sites: If True, trace only stochastic sites with metadata.
            interventions: Site values fixed by do-intervention.
            observations: Site values fixed by conditioning.
            replay: Site values fixed by replay/substitution.
            kind_labels: Label convention for ``kind``. ``"probabilistic"``
                yields ``sample``/``observe``/``condition``/``do`` while
                ``"legacy"`` yields ``sample``/``observe``/``replay``/``intervene``.
        """
        if kind_labels not in {"probabilistic", "legacy"}:
            raise ValueError("kind_labels must be either 'probabilistic' or 'legacy'.")

        self.traced_vars = traced_vars
        self.sites = bool(sites)
        self.interventions = {} if interventions is None else dict(interventions)
        self.observations = {} if observations is None else dict(observations)
        self.replay = {} if replay is None else dict(replay)
        self.kind_labels = kind_labels

    @staticmethod
    def _legacy_site_kind(name: str, interventions, observations, replay) -> str:
        if name in interventions:
            return "intervene"
        if name in observations:
            return "observe"
        if name in replay:
            return "replay"
        return "sample"

    @staticmethod
    def _probabilistic_site_kind(legacy_kind: str) -> str:
        if legacy_kind == "intervene":
            return "do"
        if legacy_kind == "replay":
            return "condition"
        return legacy_kind

    def __call__(
        self, eqn: JaxprEqn, known_inputs: Sequence[Any | None], _: Sequence[Any | None]
    ) -> ProcessedResult:
        result = super().__call__(eqn, known_inputs, _)
        outvars, outvals = result.resolved_vars, list(result.resolved_vals)
        eqn_state: dict[str, Any] = {}

        if self.sites:
            if eqn.primitive is not rv_p:
                return ProcessedResult(outvars, outvals, eqn_state)

            rv_params = parse_random_variable_call_params(eqn.params)
            name = rv_params.name

            if name in self.interventions:
                value = self.interventions[name]
                outvals = [value]
            elif name in self.observations:
                value = self.observations[name]
                outvals = [value]
            elif name in self.replay:
                value = self.replay[name]
                outvals = [value]
            else:
                value = outvals[0]

            if self.traced_vars is None or name in self.traced_vars:
                legacy_kind = self._legacy_site_kind(
                    name,
                    self.interventions,
                    self.observations,
                    self.replay,
                )
                probabilistic_kind = self._probabilistic_site_kind(legacy_kind)
                if probabilistic_kind == "do":
                    log_prob = None
                else:
                    in_known_values = list(known_inputs)
                    in_known_values[0] = value
                    log_prob = rv_params.logpdf_fn(*in_known_values)

                if self.kind_labels == "probabilistic":
                    kind = probabilistic_kind
                else:
                    kind = legacy_kind

                eqn_state[name] = {
                    "name": name,
                    "kind": kind,
                    "kind_probabilistic": probabilistic_kind,
                    "kind_legacy": legacy_kind,
                    "value": value,
                    "log_prob": log_prob,
                    "dist": rv_params.dist,
                    "shape": rv_params.shape,
                }
            return ProcessedResult(outvars, outvals, eqn_state)

        for outvar, outval in zip(outvars, outvals, strict=False):
            var_name = str(outvar)
            if self.traced_vars is None or var_name in self.traced_vars:
                eqn_state[var_name] = outval
        return ProcessedResult(outvars, outvals, eqn_state)


def trace_state_reducer(env, eqn, state, eqn_state, context=None):
    del env, eqn, context
    if state is None:
        state = {}
    if not eqn_state:
        return state
    return state | eqn_state

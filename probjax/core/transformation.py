from functools import wraps
from typing import Any, Callable, Iterable, Mapping, Optional, cast

import jax
from jax import numpy as jnp
from jaxtyping import Array

from probjax.core.custom_primitives.random_variable import name_stack
from probjax.core.interpreters import (
    INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
    IntervenedProcessingRule,
    InverseAndLogAbsDetProcessingRule,
    InverseProcessingRule,
    JointSampleProcessingRule,
    LogPotentialProcessingRule,
    TraceProcessingRule,
    inverse_and_logabsdet_state_reducer,
    inverse_cost_fn,
    joint_sample_state_reducer,
    log_potential_state_reducer,
    maybe_inverse_custom_inverse,
    trace_state_reducer,
)
from probjax.core.jaxpr_propagation.interpret import interpret
from probjax.core.jaxpr_propagation.propagate import propagate


def _resolve_invertible_index(args, invertible_arg: int) -> int:
    flat_args, _ = jax.tree_util.tree_flatten(args)
    n_args = len(flat_args)
    index = n_args + invertible_arg if invertible_arg < 0 else invertible_arg
    if index < 0 or index >= n_args:
        raise IndexError(
            f"invertible_arg={invertible_arg} is out of range for {n_args} flattened args."
        )
    return index


def _prepare_inverse_problem(
    jaxpr_invars,
    args,
    invertible_arg,
):
    if invertible_arg is None:
        return [], list(jaxpr_invars), list(args)

    adjusted_index = _resolve_invertible_index(args, invertible_arg)
    flatten_args, _ = jax.tree_util.tree_flatten(args)
    out_arg = [flatten_args[adjusted_index]]
    args_for_propagate = list(
        flatten_args[:adjusted_index] + flatten_args[adjusted_index + 1 :] + out_arg
    )
    known_invars = list(
        jaxpr_invars[:adjusted_index] + jaxpr_invars[adjusted_index + 1 :]
    )
    target_invars = [jaxpr_invars[adjusted_index]]
    return known_invars, target_invars, args_for_propagate


def _sum_log_dets_for_vars(log_dets: dict, vars_) -> jax.Array:
    total = jnp.asarray(0.0)
    for var in vars_:
        total = total + jnp.asarray(log_dets.get(var, 0.0))
    return total


def _leaf_signature(leaf):
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return ("array", tuple(leaf.shape), str(leaf.dtype))

    try:
        hash(leaf)
        return ("py", leaf)
    except TypeError:
        return ("obj", type(leaf).__name__, repr(leaf))


def _trace_signature(args, kwargs):
    flat, tree = jax.tree_util.tree_flatten((args, kwargs))
    return tree, tuple(_leaf_signature(leaf) for leaf in flat)


def _cached_jaxpr_getter(fun: Callable, static_argnums=()):
    jaxpr_maker = jax.make_jaxpr(fun, static_argnums=static_argnums)
    cache = {}

    def get_jaxpr(*args, **kwargs):
        key = _trace_signature(args, kwargs)
        if key not in cache:
            cache[key] = jaxpr_maker(*args, **kwargs)
        return cache[key]

    return get_jaxpr


_INTERVENTIONS_ATTR = "_probjax_interventions_map"
_OBSERVATIONS_ATTR = "_probjax_observations_map"
_REPLAY_ATTR = "_probjax_replay_map"


def _flatten_call_inputs(args, kwargs):
    flat_inputs, _ = jax.tree_util.tree_flatten((args, kwargs))
    return tuple(flat_inputs)


def _collect_stochastic_maps(fun: Callable):
    interventions = dict(getattr(fun, _INTERVENTIONS_ATTR, {}))
    observations = dict(getattr(fun, _OBSERVATIONS_ATTR, {}))
    replay = dict(getattr(fun, _REPLAY_ATTR, {}))
    return interventions, observations, replay


def _get_base_fun(fun: Callable) -> Callable:
    return cast(Callable, getattr(fun, "_probjax_base_fun", fun))


def _set_stochastic_maps(
    wrapped: Callable,
    *,
    interventions: Mapping[str, Any],
    observations: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> Callable:
    setattr(wrapped, _INTERVENTIONS_ATTR, dict(interventions))
    setattr(wrapped, _OBSERVATIONS_ATTR, dict(observations))
    setattr(wrapped, _REPLAY_ATTR, dict(replay))
    setattr(wrapped, "_probjax_interventions", frozenset(interventions.keys()))
    return wrapped


def _merge_maps(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for name, value in dict(update).items():
        merged[name] = value
    return merged


def _make_substitution_wrapper(
    fun: Callable, substitutions: Mapping[str, Any]
) -> Callable:
    """Create a callable wrapper that replaces stochastic sites with fixed values."""
    base_fun = _get_base_fun(fun)
    get_jaxpr = _cached_jaxpr_getter(base_fun)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        jaxpr = get_jaxpr(*args, **kwargs)
        out_flat = interpret(
            jaxpr.jaxpr,
            jaxpr.consts,
            jaxpr.jaxpr.invars,
            _flatten_call_inputs(args, kwargs),
            jaxpr.jaxpr.outvars,
            process_eqn=IntervenedProcessingRule(interventions=dict(substitutions)),
        )
        out_template = jax.eval_shape(fun, *args, **kwargs)
        out_tree = jax.tree_util.tree_structure(out_template)
        return jax.tree_util.tree_unflatten(out_tree, out_flat)

    return wrapped


def joint_sample(fun: Callable, rvs: Optional[Iterable] = None) -> Callable:
    """Samples all random variables called in the probabilistic function. If rvs is
    given, it only samples the random variables in rvs.

    Args:
        fun (Callable): Probabilistic function
        rvs (Optional[Iterable], optional): Subset of random variables in the
            probabilistic program. Defaults to None.

    Returns:
        Callable: Sampling function
    """
    base_fun = _get_base_fun(fun)
    get_jaxpr = _cached_jaxpr_getter(base_fun)
    interventions, observations, replay = _collect_stochastic_maps(fun)
    fixed_values = {}
    fixed_values.update(replay)
    fixed_values.update(observations)
    fixed_values.update(interventions)
    legacy_fixed = getattr(fun, "_probjax_interventions", None)

    def wrapped(*args, **kwargs):
        processing_rule = JointSampleProcessingRule(
            rvs=rvs,
            fixed_values=fixed_values,
            fixed_names=legacy_fixed,
        )
        jaxpr = get_jaxpr(*args, **kwargs)
        joint_result = cast(
            tuple[list, dict],
            interpret(
                jaxpr.jaxpr,
                jaxpr.consts,
                jaxpr.jaxpr.invars,
                _flatten_call_inputs(args, kwargs),
                jaxpr.jaxpr.outvars,
                process_eqn=processing_rule,
                reducer=joint_sample_state_reducer,
                initial_state={},
                return_state=True,
            ),
        )
        joint_samples = joint_result[1]

        return joint_samples

    return wrapped


def intervene(fun: Callable, rvs: Mapping[str, Array], *args, **kwargs):
    """Fix stochastic sites via intervention values.

    This is equivalent to a causal ``do`` operation. The name ``do`` is the
    probabilistic alias for this function.

    This does not sample the random variables, but fixes them to the given values.

    The wrapped function uses interpreter-level overrides for the selected
    random variables while leaving all other equations unchanged.

    Args:
        fun (Callable): A function to transform.
        rvs (dict[str, Array]): A dictionary of random variable names and values to
            intervene.

    Returns:
        Callable: Wrapped probabilistic function with interventions.
    """

    del args, kwargs
    base_fun = _get_base_fun(fun)
    current_interventions, current_observations, current_replay = (
        _collect_stochastic_maps(fun)
    )

    merged_interventions = _merge_maps(current_interventions, rvs)
    merged_observations = {
        name: value
        for name, value in current_observations.items()
        if name not in merged_interventions
    }
    merged_replay = {
        name: value
        for name, value in current_replay.items()
        if name not in merged_interventions
    }

    substitutions = {}
    substitutions.update(merged_replay)
    substitutions.update(merged_observations)
    substitutions.update(merged_interventions)

    wrapped = _make_substitution_wrapper(base_fun, substitutions)
    setattr(wrapped, "_probjax_base_fun", base_fun)
    return _set_stochastic_maps(
        wrapped,
        interventions=merged_interventions,
        observations=merged_observations,
        replay=merged_replay,
    )


def do(fun: Callable, interventions: Mapping[str, Array], *args, **kwargs) -> Callable:
    """Apply a causal ``do`` intervention to stochastic sites.

    This is a probabilistic alias for :func:`intervene`.
    """
    return intervene(fun, interventions, *args, **kwargs)


def condition(fun: Callable, observations: Mapping[str, Array]) -> Callable:
    """Condition a probabilistic program on observed site values.

    The preferred probabilistic name is :func:`observe`.
    """
    base_fun = _get_base_fun(fun)
    current_interventions, current_observations, current_replay = (
        _collect_stochastic_maps(fun)
    )

    merged_observations = _merge_maps(current_observations, observations)
    merged_observations = {
        name: value
        for name, value in merged_observations.items()
        if name not in current_interventions
    }

    merged_replay = {
        name: value
        for name, value in current_replay.items()
        if name not in current_interventions and name not in merged_observations
    }

    substitutions = {}
    substitutions.update(merged_replay)
    substitutions.update(merged_observations)
    substitutions.update(current_interventions)

    wrapped = _make_substitution_wrapper(base_fun, substitutions)
    setattr(wrapped, "_probjax_base_fun", base_fun)
    return _set_stochastic_maps(
        wrapped,
        interventions=current_interventions,
        observations=merged_observations,
        replay=merged_replay,
    )


def observe(fun: Callable, observations: Mapping[str, Array]) -> Callable:
    """Observe stochastic sites in a probabilistic program.

    This is a probabilistic alias for :func:`condition`.
    """
    return condition(fun, observations)


def _normalize_substitute_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")

    normalized_mode = mode.strip().lower()
    if normalized_mode in {"condition", "replay"}:
        return "condition"
    if normalized_mode in {"do", "intervene"}:
        return "do"

    raise ValueError("mode must be one of {'condition', 'replay', 'do', 'intervene'}.")


def substitute(
    fun: Callable, values: Mapping[str, Array], mode: str = "condition"
) -> Callable:
    """Substitute stochastic sites with fixed values.

    Args:
        fun: Probabilistic function.
        values: Site-value mapping.
        mode: `"condition"` (or legacy `"replay"`) includes substituted sites
            in log-probability; `"do"` (or legacy `"intervene"`) applies
            intervention semantics and drops substituted-site log terms.
    """
    normalized_mode = _normalize_substitute_mode(mode)

    if normalized_mode == "condition":
        base_fun = _get_base_fun(fun)
        current_interventions, current_observations, current_replay = (
            _collect_stochastic_maps(fun)
        )
        merged_replay = _merge_maps(current_replay, values)
        merged_replay = {
            name: value
            for name, value in merged_replay.items()
            if name not in current_interventions and name not in current_observations
        }

        substitutions = {}
        substitutions.update(merged_replay)
        substitutions.update(current_observations)
        substitutions.update(current_interventions)

        wrapped = _make_substitution_wrapper(base_fun, substitutions)
        setattr(wrapped, "_probjax_base_fun", base_fun)
        return _set_stochastic_maps(
            wrapped,
            interventions=current_interventions,
            observations=current_observations,
            replay=merged_replay,
        )

    if normalized_mode == "do":
        return do(fun, values)

    raise AssertionError("unreachable")


def scope(name: str):
    """Create a naming scope for stochastic sites."""
    return name_stack.scope(name)


def log_potential_fn(
    fun: Callable,
    *args,
    strict: bool = True,
    allow_partial: bool = False,
    **kwargs,
):
    """Compute the unnormalized log density of a probabilistic function.

    This is the legacy name for :func:`log_joint_fn`.

    This does not include the normalizing constant.

    Args:
        fun (Callable): Probabilistic function

    Returns:
        Callable: Log potential function
    """
    base_fun = _get_base_fun(fun)
    interventions, observations, replay = _collect_stochastic_maps(fun)
    legacy_interventions = getattr(fun, "_probjax_interventions", None)

    jaxpr = jax.make_jaxpr(base_fun)(jax.random.PRNGKey(0), *args, **kwargs)
    model_inputs = _flatten_call_inputs((jax.random.PRNGKey(0),) + args, kwargs)

    def log_potential(**joint_samples):
        intervention_config = interventions if interventions else legacy_interventions
        processing_rule = LogPotentialProcessingRule(
            joint_samples=joint_samples,
            interventions=intervention_config,
            observations=observations,
            replay=replay,
            strict=strict,
            allow_partial=allow_partial,
        )

        log_potential_result = cast(
            tuple[list, jax.Array],
            interpret(
                jaxpr.jaxpr,
                jaxpr.consts,
                jaxpr.jaxpr.invars,
                model_inputs,
                jaxpr.jaxpr.outvars,
                process_eqn=processing_rule,
                reducer=log_potential_state_reducer,
                initial_state=jnp.asarray(0.0),
                return_state=True,
            ),
        )
        log_prob = log_potential_result[1]

        return jnp.nan_to_num(log_prob, nan=-jnp.inf, posinf=jnp.inf, neginf=-jnp.inf)

    return log_potential


def log_joint_fn(
    fun: Callable,
    *args,
    strict: bool = True,
    allow_partial: bool = False,
    **kwargs,
):
    """Compute the model log-joint (up to a constant)."""
    return log_potential_fn(
        fun,
        *args,
        strict=strict,
        allow_partial=allow_partial,
        **kwargs,
    )


def log_prob_fn(
    fun: Callable,
    *args,
    strict: bool = True,
    allow_partial: bool = False,
    **kwargs,
):
    """Backward-compatible alias for :func:`log_joint_fn`."""
    return log_joint_fn(
        fun,
        *args,
        strict=strict,
        allow_partial=allow_partial,
        **kwargs,
    )


def trace(
    fun: Callable,
    traced_vars=None,
    *,
    sites: bool = False,
    kind_labels: str = "probabilistic",
):
    base_fun = _get_base_fun(fun)
    get_jaxpr = _cached_jaxpr_getter(base_fun)
    interventions, observations, replay = _collect_stochastic_maps(fun)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        processing_rule = TraceProcessingRule(
            traced_vars=traced_vars,
            sites=sites,
            interventions=interventions,
            observations=observations,
            replay=replay,
            kind_labels=kind_labels,
        )
        jaxpr = get_jaxpr(*args, **kwargs)
        trace_result = cast(
            tuple[list, dict],
            interpret(
                jaxpr.jaxpr,
                jaxpr.consts,
                jaxpr.jaxpr.invars,
                _flatten_call_inputs(args, kwargs),
                jaxpr.jaxpr.outvars,
                process_eqn=processing_rule,
                reducer=trace_state_reducer,
                initial_state={},
                return_state=True,
            ),
        )
        traced_samples = trace_result[1]

        return traced_samples

    return wrapped


def inverse(fun: Callable, static_argnums=(), invertible_arg=None):
    maybe_custom = maybe_inverse_custom_inverse(
        fun,
        static_argnums=static_argnums,
        invertible_arg=invertible_arg,
    )
    if maybe_custom is not None:
        return maybe_custom

    get_jaxpr = _cached_jaxpr_getter(fun, static_argnums=static_argnums)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        processing_rule = InverseProcessingRule()
        jaxpr = get_jaxpr(*args, **kwargs)
        known_invars, target_invars, args_for_propagate = _prepare_inverse_problem(
            jaxpr.jaxpr.invars,
            args,
            invertible_arg,
        )
        out = propagate(
            jaxpr.jaxpr,
            jaxpr.consts,
            known_invars + jaxpr.jaxpr.outvars,
            args_for_propagate,
            target_invars,
            process_eqn=processing_rule,
            cost_fn=inverse_cost_fn,
            process_all_eqns=True,
        )

        return out[0]

    return wrapped


def inverse_and_logabsdet(fun: Callable, static_argnums=(), invertible_arg=None):
    get_jaxpr = _cached_jaxpr_getter(fun, static_argnums=static_argnums)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        processing_rule = InverseAndLogAbsDetProcessingRule(
            state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE
        )
        jaxpr = get_jaxpr(*args, **kwargs)
        known_invars, target_invars, args_for_propagate = _prepare_inverse_problem(
            jaxpr.jaxpr.invars,
            args,
            invertible_arg,
        )
        invars = known_invars + jaxpr.jaxpr.outvars
        outvars = target_invars

        inverse_result = cast(
            tuple[list, dict],
            propagate(
                jaxpr.jaxpr,
                jaxpr.consts,
                invars,
                args_for_propagate,
                outvars,
                process_eqn=processing_rule,
                cost_fn=inverse_cost_fn,
                process_all_eqns=True,
                reducer=inverse_and_logabsdet_state_reducer,
                initial_state={},
                return_state=True,
                state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
            ),
        )
        out, log_dets = inverse_result

        log_det = _sum_log_dets_for_vars(log_dets, outvars)
        return out[0], log_det

    return wrapped

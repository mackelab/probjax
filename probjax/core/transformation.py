from functools import wraps
from typing import Any, Callable, Iterable, Mapping, Optional, cast

import jax
from jax import numpy as jnp
from jaxtyping import Array

from probjax.core.custom_primitives.random_variable import (
    enable_rv_tracing,
    name_stack,
)
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
from probjax.core.interpreters.inverse.affine import solve_affine_inverse
from probjax.core.interpreters.inverse.recovery import make_affine_recovery
from probjax.core.jaxpr_propagation import interpret, propagate
from probjax.core.jaxpr_propagation.utils import KnownessLevel
from probjax.core.registry import invalid_inverse_value


def _normalize_argnums(argnums, n_args: int, *, name: str) -> tuple[int, ...]:
    if isinstance(argnums, int):
        argnums = (argnums,)
    normalized = []
    for argnum in argnums:
        index = n_args + argnum if argnum < 0 else argnum
        if index < 0 or index >= n_args:
            raise IndexError(f"{name}={argnum} is out of range for {n_args} args.")
        if index not in normalized:
            normalized.append(index)
    return tuple(normalized)


def _resolve_invertible_index(args, invertible_arg: int | None) -> int:
    n_args = len(args)
    if n_args == 0:
        raise ValueError("inverse requires at least one positional argument")
    if invertible_arg is None:
        invertible_arg = 0
    index = n_args + invertible_arg if invertible_arg < 0 else invertible_arg
    if index < 0 or index >= n_args:
        raise IndexError(
            f"invertible_arg={invertible_arg} is out of range for {n_args} args."
        )
    return index


def _prepare_inverse_problem(
    jaxpr_invars,
    args,
    kwargs,
    invertible_arg,
    static_argnums,
):
    target_index = _resolve_invertible_index(args, invertible_arg)
    static_indices = set(
        _normalize_argnums(static_argnums, len(args), name="static_argnums")
    )
    if target_index in static_indices:
        raise ValueError("invertible_arg cannot also be listed in static_argnums")

    dynamic_values = []
    target_leaf_indices = []
    target_tree = None
    for index, arg in enumerate(args):
        if index in static_indices:
            continue
        leaves, tree = jax.tree_util.tree_flatten(arg)
        start = len(dynamic_values)
        dynamic_values.extend(leaves)
        if index == target_index:
            target_leaf_indices.extend(range(start, len(dynamic_values)))
            target_tree = tree

    kwarg_values, _ = jax.tree_util.tree_flatten(kwargs)
    dynamic_values.extend(kwarg_values)

    if len(dynamic_values) != len(jaxpr_invars):
        raise ValueError(
            "Dynamic argument leaves do not match the traced JAXPR inputs: "
            f"got {len(dynamic_values)} values for {len(jaxpr_invars)} variables."
        )
    if target_tree is None:
        raise ValueError(
            "invertible_arg did not identify a dynamic positional argument"
        )

    target_indices = set(target_leaf_indices)
    known_invars = [
        var for index, var in enumerate(jaxpr_invars) if index not in target_indices
    ]
    known_values = [
        value
        for index, value in enumerate(dynamic_values)
        if index not in target_indices
    ]
    target_invars = [jaxpr_invars[index] for index in target_leaf_indices]
    output_values, output_tree = jax.tree_util.tree_flatten(args[target_index])
    return (
        known_invars,
        target_invars,
        known_values + output_values,
        target_tree,
        output_tree,
    )


def _sum_log_dets_for_vars(log_dets: dict, vars_) -> jax.Array:
    total = jnp.asarray(0.0)
    for var in vars_:
        total = total + jnp.asarray(log_dets.get(var, 0.0))
    return total


def _materialize_inverse_targets(values, target_vars, env):
    materialized = []
    complete = True
    for value, var in zip(values, target_vars, strict=False):
        if env.get_knowness_level(var) == KnownessLevel.COMPLETE and value is not None:
            materialized.append(value)
            continue
        complete = False
        materialized.append(
            invalid_inverse_value(
                var.aval,
                message="inverse target could not be reconstructed completely",
            )
        )
    return materialized, complete


def _affine_fallback(
    jaxpr, known_invars, args_for_propagate, target_invars, *, compute_logdet=True
):
    """Try a linear solve where equation-by-equation propagation gave up.

    Only reached when propagation could not reconstruct the target, so this can
    never change an answer the interpreter already produced -- it replaces NaN
    with a value, or returns None and leaves the NaN in place.
    """
    num_outputs = len(jaxpr.jaxpr.outvars)
    known_values = args_for_propagate[: len(known_invars)]
    output_values = args_for_propagate[len(args_for_propagate) - num_outputs :]
    return solve_affine_inverse(
        jaxpr.jaxpr,
        jaxpr.consts,
        target_invars,
        dict(zip(known_invars, known_values, strict=False)),
        output_values,
        compute_logdet=compute_logdet,
    )


def _validate_inverse_shapes(value, forward_output):
    """The output is also the presumed input signature for automatic inversion."""
    leaves, tree = jax.tree_util.tree_flatten_with_path(value)
    expected, expected_tree = jax.tree_util.tree_flatten(forward_output)
    if tree != expected_tree:
        raise ValueError(
            "Inverse output structure does not match function outputs; automatic "
            "inversion requires matching input/output pytree structures."
        )
    for (path, leaf), output in zip(leaves, expected, strict=True):
        shape = jnp.shape(leaf)
        if shape != output.shape:
            raise ValueError(
                "Automatic inversion requires matching input/output shapes: "
                f"leaf {jax.tree_util.keystr(path) or '<root>'} has presumed input "
                f"shape {shape}, but the traced output has shape {output.shape}."
            )


def _leaf_signature(leaf):
    """Create a hashable signature for a pytree leaf for caching."""
    # Fast path for JAX arrays (most common case)
    if isinstance(leaf, jax.Array):
        return ("array", leaf.shape, leaf.dtype)
    # NumPy arrays or similar
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return ("array", tuple(leaf.shape), str(leaf.dtype))
    # Hashable Python objects
    try:
        hash(leaf)
        return ("py", leaf)
    except TypeError:
        return ("obj", type(leaf).__name__, repr(leaf))


def _flatten_and_signature(args, kwargs):
    """Flatten args/kwargs and compute cache signature in one pass.

    Returns:
        (flat_inputs, cache_key) where flat_inputs is a tuple and cache_key
        is a hashable tuple suitable for dict keys.
    """
    flat, tree = jax.tree_util.tree_flatten((args, kwargs))
    sig = (tree, tuple(_leaf_signature(leaf) for leaf in flat))
    return tuple(flat), sig


def _cached_jaxpr_and_tree_getter(fun: Callable, static_argnums=()):
    """Create a cached getter for both JAXPR and output tree structure.

    This avoids re-tracing and re-computing eval_shape on every call.

    Tracing goes through a wrapper whose identity is unique to this getter:
    JAX's global trace cache is keyed on callable identity, so tracing ``fun``
    directly can silently return a jaxpr whose consts capture stale closure
    state — e.g. an nnx module's weights from before in-place training. The
    per-getter wrapper guarantees this getter's first trace sees the current
    state; repeat calls on the same getter reuse its local cache (snapshot
    semantics).
    """

    def fun_snapshot(*args, **kwargs):
        return fun(*args, **kwargs)

    jaxpr_maker = jax.make_jaxpr(fun_snapshot, static_argnums=static_argnums)
    cache: dict = {}

    def get_jaxpr_and_tree(flat_inputs, cache_key, args, kwargs):
        """Get cached JAXPR and output tree, tracing if needed."""
        if cache_key not in cache:
            jaxpr = jaxpr_maker(*args, **kwargs)
            out_template = jax.eval_shape(fun_snapshot, *args, **kwargs)
            out_tree = jax.tree_util.tree_structure(out_template)
            cache[cache_key] = (jaxpr, out_tree)
        return cache[cache_key]

    return get_jaxpr_and_tree


def _cached_jaxpr_getter(fun: Callable, static_argnums=(), *, return_shape=False):
    """Create a cached getter for JAXPR only (backward compatible)."""
    def fun_snapshot(*args, **kwargs):
        return fun(*args, **kwargs)

    jaxpr_maker = jax.make_jaxpr(
        fun_snapshot, static_argnums=static_argnums, return_shape=return_shape
    )
    cache: dict = {}

    def get_jaxpr(*args, **kwargs):
        _, cache_key = _flatten_and_signature(args, kwargs)
        if cache_key not in cache:
            cache[cache_key] = jaxpr_maker(*args, **kwargs)
        return cache[cache_key]

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
    """Merge two mappings, with update taking precedence."""
    return {**base, **update}


def _apply_stochastic_substitution(
    fun: Callable,
    *,
    update_interventions: Optional[Mapping[str, Any]] = None,
    update_observations: Optional[Mapping[str, Any]] = None,
    update_replay: Optional[Mapping[str, Any]] = None,
) -> Callable:
    """Unified substitution logic for intervene/condition/substitute.

    Priority order: interventions > observations > replay.
    Sites in higher priority categories are excluded from lower priority categories.
    """
    base_fun = _get_base_fun(fun)
    current_int, current_obs, current_replay = _collect_stochastic_maps(fun)

    # Merge with priority (interventions override observations override replay)
    merged_int = _merge_maps(current_int, update_interventions or {})
    merged_obs = {
        k: v
        for k, v in _merge_maps(current_obs, update_observations or {}).items()
        if k not in merged_int
    }
    merged_replay = {
        k: v
        for k, v in _merge_maps(current_replay, update_replay or {}).items()
        if k not in merged_int and k not in merged_obs
    }

    # Build substitutions dict with priority order
    substitutions = {**merged_replay, **merged_obs, **merged_int}

    wrapped = _make_substitution_wrapper(base_fun, substitutions)
    setattr(wrapped, "_probjax_base_fun", base_fun)
    return _set_stochastic_maps(
        wrapped,
        interventions=merged_int,
        observations=merged_obs,
        replay=merged_replay,
    )


def _make_substitution_wrapper(
    fun: Callable, substitutions: Mapping[str, Any]
) -> Callable:
    """Create a callable wrapper that replaces stochastic sites with fixed values."""
    base_fun = _get_base_fun(fun)
    get_jaxpr_and_tree = _cached_jaxpr_and_tree_getter(base_fun)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        # Single flatten pass for both cache key and inputs
        flat_inputs, cache_key = _flatten_and_signature(args, kwargs)
        with enable_rv_tracing():
            jaxpr, out_tree = get_jaxpr_and_tree(flat_inputs, cache_key, args, kwargs)
        out_flat = interpret(
            jaxpr.jaxpr,
            jaxpr.consts,
            jaxpr.jaxpr.invars,
            flat_inputs,
            jaxpr.jaxpr.outvars,
            process_eqn=IntervenedProcessingRule(interventions=dict(substitutions)),
        )
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
        with enable_rv_tracing():
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


def intervene(fun: Callable, rvs: Mapping[str, Array]) -> Callable:
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
    return _apply_stochastic_substitution(fun, update_interventions=rvs)


def do(fun: Callable, interventions: Mapping[str, Array]) -> Callable:
    """Apply a causal ``do`` intervention to stochastic sites.

    This is a probabilistic alias for :func:`intervene`.
    """
    return intervene(fun, interventions)


def condition(fun: Callable, observations: Mapping[str, Array]) -> Callable:
    """Condition a probabilistic program on observed site values.

    The preferred probabilistic name is :func:`observe`.
    """
    return _apply_stochastic_substitution(fun, update_observations=observations)


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
        return _apply_stochastic_substitution(fun, update_replay=values)

    if normalized_mode == "do":
        return intervene(fun, values)

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

    with enable_rv_tracing():
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
        with enable_rv_tracing():
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
    """Return a function computing the inverse of ``fun``.

    Traces ``fun`` to a jaxpr and walks it backwards, replacing each primitive
    with its registered inverse rule, so the result is ordinary JAX code with no
    interpreter left at runtime:

    >>> inverse(lambda x: 2 * jnp.exp(x))(jnp.asarray(4.0))
    Array(0.6931472, dtype=float32)

    Args:
        fun: The function to invert. May itself be a ``custom_inverse``, in
            which case its registered inverse is used directly.
        static_argnums: Positional arguments held fixed rather than inverted.
        invertible_arg: Which positional argument to solve for (default 0).

    Returns:
        A callable mapping outputs back to the invertible argument.

    Raises:
        ValueError: if traced input/output shapes or pytree structures differ.
            Automatic inversion uses the supplied output as the presumed input
            signature; it requires a shape-preserving function. Some violations
            (such as a broadcast that becomes an identity at that signature)
            cannot be detected without the original input specification.

    Note:
        **An unresolved inversion returns NaN.** The interpreter works
        one equation at a time, so it inverts a *tree* of operations; a value
        used twice stalls it, because the bivariate rules need exactly one
        unknown operand.

        On a stall, structurally proven affine sections are recovered by linear
        solves and propagation resumes. This supports ``exp(3*x-x)`` and
        ``3*exp(x)-exp(x)``, including sequential compositions and nested jit.
        Sections have one array-valued input and output with equal element
        counts; internal shape changes are allowed. Whole-program affine solves
        also support multiple input leaves. Analysis is lazy and cached; each
        dense section Jacobian costs O(n^2) storage. Generated inverses are
        ordinary JAX computations and can be jitted, vmapped, and differentiated.
        Repeated eager calls still run the Python propagation interpreter.

        These remain silently unsupported and produce NaN:

        * a value used more than once **nonlinearly** -- ``x * x``, or a
          residual ``x + f(x)``. A residual is invertible by fixed-point
          iteration when its branch is a contraction, but a jaxpr carries no
          Lipschitz bound, so register it with :class:`custom_inverse` instead.
        * ``lax.fori_loop``, and any ``lax.scan`` carrying something that is not
          itself invertible (a counter, a running sum). Plain ``scan`` and
          ``lax.cond`` do work.
        * ``inverse(inverse(f))``.

        ``lax.while_loop`` raises rather than returning NaN. Checking
        ``jnp.isfinite`` on the result is the reliable way to detect the rest.
    """
    maybe_custom = maybe_inverse_custom_inverse(
        fun,
        static_argnums=static_argnums,
        invertible_arg=invertible_arg,
    )
    if maybe_custom is not None:
        return maybe_custom

    get_jaxpr = _cached_jaxpr_getter(
        fun, static_argnums=static_argnums, return_shape=True
    )
    recovery_cache = {}

    @wraps(fun)
    def wrapped(*args, **kwargs):
        processing_rule = InverseProcessingRule()
        jaxpr, forward_output = get_jaxpr(*args, **kwargs)
        _validate_inverse_shapes(
            args[_resolve_invertible_index(args, invertible_arg)], forward_output
        )
        (
            known_invars,
            target_invars,
            args_for_propagate,
            target_tree,
            _,
        ) = _prepare_inverse_problem(
            jaxpr.jaxpr.invars,
            args,
            kwargs,
            invertible_arg,
            static_argnums,
        )
        if len(args_for_propagate) - len(known_invars) != len(jaxpr.jaxpr.outvars):
            raise ValueError("Inverse output structure does not match function outputs")
        out, env = cast(
            tuple[list, Any],
            propagate(
                jaxpr.jaxpr,
                jaxpr.consts,
                known_invars + jaxpr.jaxpr.outvars,
                args_for_propagate,
                target_invars,
                process_eqn=processing_rule,
                stall_recovery=make_affine_recovery(
                    jaxpr,
                    known_invars,
                    args_for_propagate,
                    target_invars,
                    processing_rule,
                    recovery_cache,
                    with_logdet=False,
                ),
                cost_fn=inverse_cost_fn,
                process_all_eqns=True,
                return_env=True,
            ),
        )
        out, complete = _materialize_inverse_targets(out, target_invars, env)
        if not complete:
            solved = _affine_fallback(
                jaxpr,
                known_invars,
                args_for_propagate,
                target_invars,
                compute_logdet=False,
            )
            if solved is not None:
                out = solved[0]

        return jax.tree_util.tree_unflatten(target_tree, out)

    return wrapped


def inverse_and_logabsdet(fun: Callable, static_argnums=(), invertible_arg=None):
    """Return a function computing the inverse of ``fun`` and its log-det.

    The log-determinant is that of the **inverse** map -- ``log|d(inv)/dy|``,
    summed over the event -- which is the term a change of variables needs:

    >>> inverse_and_logabsdet(lambda x: 2 * jnp.exp(x))(jnp.asarray(4.0))
    (Array(0.6931472, dtype=float32), Array(-1.3862944, dtype=float32))

    Args:
        fun: The function to invert, possibly a ``custom_inverse``.
        static_argnums: Positional arguments held fixed rather than inverted.
        invertible_arg: Which positional argument to solve for (default 0).

    Returns:
        ``(inverse_value, log_abs_det)``. Both are NaN if the inversion could
        not be completed.

    Raises:
        ValueError: if traced boundary shapes or pytree structures differ,
            with the same input-signature limitation as :func:`inverse`.
        NotImplementedError: if a primitive on the inverse path has no
            log-determinant rule and is not elementwise. Guessing one by
            differentiating the inverse elementwise -- the old behaviour --
            silently returned a number that was not a log-determinant.

    Note:
        Everything :func:`inverse` cannot do applies here too, and the log-det
        additionally requires a rule for every primitive involved. A primitive
        that inverts fine may still have no log-det: ``dynamic_slice`` recovers
        only its window, leaving the input partially known and no square
        Jacobian to take a determinant of.
    """
    maybe_custom = maybe_inverse_custom_inverse(
        fun,
        static_argnums=static_argnums,
        invertible_arg=invertible_arg,
    )
    if maybe_custom is not None:
        @wraps(fun)
        def custom_wrapped(*args, **kwargs):
            value, logdet = fun.inv_and_logdet(*args, **kwargs)
            total_logdet = sum(
                (jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(logdet)),
                jnp.asarray(0.0),
            )
            return value, total_logdet

        return custom_wrapped

    get_jaxpr = _cached_jaxpr_getter(
        fun, static_argnums=static_argnums, return_shape=True
    )
    recovery_cache = {}

    @wraps(fun)
    def wrapped(*args, **kwargs):
        processing_rule = InverseAndLogAbsDetProcessingRule(
            state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE
        )
        jaxpr, forward_output = get_jaxpr(*args, **kwargs)
        _validate_inverse_shapes(
            args[_resolve_invertible_index(args, invertible_arg)], forward_output
        )
        (
            known_invars,
            target_invars,
            args_for_propagate,
            target_tree,
            _,
        ) = _prepare_inverse_problem(
            jaxpr.jaxpr.invars,
            args,
            kwargs,
            invertible_arg,
            static_argnums,
        )
        if len(args_for_propagate) - len(known_invars) != len(jaxpr.jaxpr.outvars):
            raise ValueError("Inverse output structure does not match function outputs")
        invars = known_invars + jaxpr.jaxpr.outvars
        outvars = target_invars

        inverse_result = cast(
            tuple[list, dict, Any],
            propagate(
                jaxpr.jaxpr,
                jaxpr.consts,
                invars,
                args_for_propagate,
                outvars,
                process_eqn=processing_rule,
                stall_recovery=make_affine_recovery(
                    jaxpr,
                    known_invars,
                    args_for_propagate,
                    target_invars,
                    processing_rule,
                    recovery_cache,
                    with_logdet=True,
                ),
                cost_fn=inverse_cost_fn,
                process_all_eqns=True,
                reducer=inverse_and_logabsdet_state_reducer,
                initial_state={},
                return_state=True,
                return_env=True,
                state_namespace=INVERSE_AND_LOGABSDET_STATE_NAMESPACE,
            ),
        )
        out, log_dets, env = inverse_result
        out, complete = _materialize_inverse_targets(out, outvars, env)

        log_det = _sum_log_dets_for_vars(log_dets, outvars)
        if not complete:
            solved = _affine_fallback(jaxpr, known_invars, args_for_propagate, outvars)
            if solved is not None:
                out, log_det = solved
            else:
                log_det = jnp.asarray(jnp.nan)
        return jax.tree_util.tree_unflatten(target_tree, out), log_det

    return wrapped

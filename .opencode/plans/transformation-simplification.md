# Transformation.py Simplification Plan

## Overview
Simplify and optimize `probjax/core/transformation.py` and related files to reduce code duplication and improve performance.

## Files to Modify
- `probjax/core/registry.py`
- `probjax/core/jaxpr_propagation/engine.py`
- `probjax/core/jaxpr_propagation/pipeline.py`
- `probjax/core/transformation.py`
- `probjax/core/interpreters/ppl/joint_sample.py`

---

## Phase 1: Add `parse_processed_result` utility to registry.py

Add after `ProcessedResult` class (line 121):

```python
def parse_processed_result(
    result: Optional[ProcessedResult],
) -> Tuple[Sequence[Any], Sequence[Any], Any]:
    """Parse a ProcessedResult into (vars, vals, state) tuple.

    This is a shared utility for engine.py and pipeline.py to avoid
    duplicating the parsing logic.

    Args:
        result: ProcessedResult from a rule, or None if rule didn't apply

    Returns:
        (resolved_vars, resolved_vals, state) tuple. Returns empty sequences
        and None state if result is None.

    Raises:
        TypeError: If result is not ProcessedResult or None
    """
    if result is None:
        return (), (), None

    if not isinstance(result, ProcessedResult):
        raise TypeError(
            f"Processing rules must return ProcessedResult or None, got {type(result).__name__}"
        )

    # Normalize to sequences (handle single values)
    resolved_vars = result.resolved_vars
    resolved_vals = result.resolved_vals
    if not isinstance(resolved_vars, (list, tuple)):
        resolved_vars = [resolved_vars]
    if not isinstance(resolved_vals, (list, tuple)):
        resolved_vals = [resolved_vals]

    return tuple(resolved_vars), tuple(resolved_vals), result.state
```

---

## Phase 2: Update engine.py and pipeline.py to use shared utility

### engine.py
Replace `_parse_process_result` (lines 250-265) with import and usage:

```python
# Add to imports
from probjax.core.registry import ProcessedResult, parse_processed_result

# Remove _parse_process_result function entirely
# Update _run_process_rule to use:
# resolved_vars, resolved_vals, state = parse_processed_result(result)
```

### pipeline.py
Replace `_parse_result` (lines 57-70) with import and usage:

```python
# Add to imports
from probjax.core.registry import ProcessedResult, parse_processed_result

# Remove _parse_result function and as_sequence import if only used there
# Update code to use parse_processed_result
```

---

## Phase 3: Simplify `_merge_maps` in transformation.py

Replace lines 130-134:

```python
def _merge_maps(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for name, value in dict(update).items():
        merged[name] = value
    return merged
```

With:

```python
def _merge_maps(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two mappings, with update taking precedence."""
    return {**base, **update}
```

---

## Phase 4: Consolidate intervene/condition/substitute patterns

Add new helper function after `_set_stochastic_maps` (around line 128):

```python
def _apply_stochastic_substitution(
    fun: Callable,
    *,
    update_interventions: Optional[Mapping[str, Any]] = None,
    update_observations: Optional[Mapping[str, Any]] = None,
    update_replay: Optional[Mapping[str, Any]] = None,
) -> Callable:
    """Unified substitution logic for intervene/condition/substitute.

    Priority order: interventions > observations > replay
    """
    base_fun = _get_base_fun(fun)
    current_int, current_obs, current_replay = _collect_stochastic_maps(fun)

    # Merge with priority (interventions override observations override replay)
    merged_int = _merge_maps(current_int, update_interventions or {})
    merged_obs = {
        k: v for k, v in _merge_maps(current_obs, update_observations or {}).items()
        if k not in merged_int
    }
    merged_replay = {
        k: v for k, v in _merge_maps(current_replay, update_replay or {}).items()
        if k not in merged_int and k not in merged_obs
    }

    # Build substitutions dict with priority
    substitutions = {**merged_replay, **merged_obs, **merged_int}

    wrapped = _make_substitution_wrapper(base_fun, substitutions)
    setattr(wrapped, "_probjax_base_fun", base_fun)
    return _set_stochastic_maps(
        wrapped,
        interventions=merged_int,
        observations=merged_obs,
        replay=merged_replay,
    )
```

Then simplify `intervene`, `condition`, and `substitute` to use it:

```python
def intervene(fun: Callable, rvs: Mapping[str, Array]) -> Callable:
    """Fix stochastic sites via intervention values (causal do operation)."""
    return _apply_stochastic_substitution(fun, update_interventions=rvs)


def do(fun: Callable, interventions: Mapping[str, Array]) -> Callable:
    """Apply a causal ``do`` intervention. Alias for intervene."""
    return intervene(fun, interventions)


def condition(fun: Callable, observations: Mapping[str, Array]) -> Callable:
    """Condition a probabilistic program on observed site values."""
    return _apply_stochastic_substitution(fun, update_observations=observations)


def observe(fun: Callable, observations: Mapping[str, Array]) -> Callable:
    """Observe stochastic sites. Alias for condition."""
    return condition(fun, observations)


def substitute(
    fun: Callable, values: Mapping[str, Array], mode: str = "condition"
) -> Callable:
    """Substitute stochastic sites with fixed values."""
    normalized_mode = _normalize_substitute_mode(mode)

    if normalized_mode == "condition":
        return _apply_stochastic_substitution(fun, update_replay=values)
    if normalized_mode == "do":
        return intervene(fun, values)

    raise AssertionError("unreachable")
```

---

## Phase 5: Cache eval_shape output tree with JAXPR

Modify `_cached_jaxpr_getter` (lines 82-92) to return both JAXPR and output tree:

```python
def _cached_jaxpr_and_tree_getter(fun: Callable, static_argnums=()):
    """Create a cached getter for both JAXPR and output tree structure."""
    jaxpr_maker = jax.make_jaxpr(fun, static_argnums=static_argnums)
    cache = {}

    def get_jaxpr_and_tree(*args, **kwargs):
        key = _trace_signature(args, kwargs)
        if key not in cache:
            jaxpr = jaxpr_maker(*args, **kwargs)
            out_template = jax.eval_shape(fun, *args, **kwargs)
            out_tree = jax.tree_util.tree_structure(out_template)
            cache[key] = (jaxpr, out_tree)
        return cache[key]

    return get_jaxpr_and_tree


# Keep _cached_jaxpr_getter for backward compatibility with functions that don't need tree
def _cached_jaxpr_getter(fun: Callable, static_argnums=()):
    """Create a cached getter for JAXPR only."""
    getter = _cached_jaxpr_and_tree_getter(fun, static_argnums)

    def get_jaxpr(*args, **kwargs):
        jaxpr, _ = getter(*args, **kwargs)
        return jaxpr

    return get_jaxpr
```

Update `_make_substitution_wrapper` (lines 137-159) to use cached tree:

```python
def _make_substitution_wrapper(
    fun: Callable, substitutions: Mapping[str, Any]
) -> Callable:
    """Create a callable wrapper that replaces stochastic sites with fixed values."""
    base_fun = _get_base_fun(fun)
    get_jaxpr_and_tree = _cached_jaxpr_and_tree_getter(base_fun)

    @wraps(fun)
    def wrapped(*args, **kwargs):
        jaxpr, out_tree = get_jaxpr_and_tree(*args, **kwargs)
        out_flat = interpret(
            jaxpr.jaxpr,
            jaxpr.consts,
            jaxpr.jaxpr.invars,
            _flatten_call_inputs(args, kwargs),
            jaxpr.jaxpr.outvars,
            process_eqn=IntervenedProcessingRule(interventions=dict(substitutions)),
        )
        return jax.tree_util.tree_unflatten(out_tree, out_flat)

    return wrapped
```

---

## Phase 6: Eliminate duplicate tree_flatten calls

Optimize `_trace_signature` and `_flatten_call_inputs` to share work.

Option A: Combine into single function that returns both:

```python
def _flatten_and_signature(args, kwargs):
    """Flatten args/kwargs and compute cache signature in one pass."""
    flat, tree = jax.tree_util.tree_flatten((args, kwargs))
    sig = (tree, tuple(_leaf_signature(leaf) for leaf in flat))
    return tuple(flat), sig
```

Then update call sites to use this combined function.

Option B: Have `_cached_jaxpr_getter` accept pre-flattened inputs.

Recommend Option A for simplicity.

---

## Phase 7: Remove legacy `_probjax_interventions` mechanism

This is more complex - need to verify no external code depends on it.

### In transformation.py:
1. Remove line 126: `setattr(wrapped, "_probjax_interventions", frozenset(interventions.keys()))`
2. Remove line 181: `legacy_fixed = getattr(fun, "_probjax_interventions", None)`
3. Remove line 187: `fixed_names=legacy_fixed,`
4. Remove line 403: `legacy_interventions = getattr(fun, "_probjax_interventions", None)`
5. Simplify line 409: `intervention_config = interventions`

### In joint_sample.py:
1. Remove `fixed_names` parameter from `__init__`
2. Simplify the fixed check logic

**Note**: Only do this if confident no external code uses `_probjax_interventions`.

---

## Phase 8: Use isinstance in _leaf_signature

Replace lines 66-74:

```python
def _leaf_signature(leaf):
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return ("array", tuple(leaf.shape), str(leaf.dtype))

    try:
        hash(leaf)
        return ("py", leaf)
    except TypeError:
        return ("obj", type(leaf).__name__, repr(leaf))
```

With:

```python
def _leaf_signature(leaf):
    """Create a hashable signature for a pytree leaf for caching."""
    # Fast path for JAX arrays (most common case)
    if isinstance(leaf, jax.Array):
        return ("array", leaf.shape, leaf.dtype)
    # NumPy arrays
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return ("array", tuple(leaf.shape), str(leaf.dtype))
    # Hashable Python objects
    try:
        hash(leaf)
        return ("py", leaf)
    except TypeError:
        return ("obj", type(leaf).__name__, repr(leaf))
```

---

## Phase 9: Run tests

```bash
uv run pytest tests/test_core_inverse_and_log_det.py tests/test_core_jaxpr_backend.py -v
```

---

## Phase 10: Commit

```bash
git add -A
git commit -m "Simplify transformation.py and reduce overhead

- Add parse_processed_result utility to registry.py
- Consolidate duplicate _parse_result functions from engine.py and pipeline.py
- Simplify _merge_maps using dict unpacking
- Create _apply_stochastic_substitution to consolidate intervene/condition/substitute
- Cache eval_shape output tree with JAXPR to avoid re-tracing
- Optimize _leaf_signature with isinstance fast path
- Remove unused *args, **kwargs from intervene/do signatures"
```

---

## Expected Impact

1. **Performance**: Eliminates `jax.eval_shape` call on every wrapped function invocation
2. **Performance**: Reduces duplicate `tree_flatten` calls
3. **Maintainability**: Consolidates ~100 lines of duplicate pattern into single helper
4. **Type Safety**: Single source of truth for ProcessedResult parsing
5. **Code Quality**: Simpler, more Pythonic implementations

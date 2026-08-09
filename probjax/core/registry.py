"""
Unified Rule Registry for probjax/core interpreters.

This module provides a simple (primitive, context) -> rule mapping for
looking up transformation rules for JAX primitives.

Design goals:
  1. Simple API: registry.register(primitive, context, rule)
  2. Direct lookup: registry.get(primitive, context) -> rule or None
  3. Composable: rules are plain functions that can be combined

Usage:
    from probjax.core.registry import REGISTRY, Context

    # Get an inverse rule
    rule = REGISTRY.get(jax.lax.exp_p, Context.INVERSE)
    if rule is not None:
        result = rule(eqn, known_in, known_out)

    # Register a new rule
    @REGISTRY.rule(my_primitive, Context.INVERSE)
    def my_inverse_rule(eqn, known_in, known_out):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import jax.numpy as jnp
from jax.experimental import checkify
from jax.extend.core import ClosedJaxpr, JaxprEqn, Literal, Primitive, Var

from probjax.core.jaxpr_propagation.utils import (
    bind_primitive,
    rebind_primitive,
    sanitize_bind_params,
)

# Atom is Var | Literal but not directly exported
Atom = Union[Var, Literal]


# =============================================================================
# Type Definitions
# =============================================================================


@runtime_checkable
class RuleFunction(Protocol):
    """
    Protocol for rule functions.

    Rules take an equation and known input/output values, returning resolved
    variables and their values, or None if the rule cannot be applied.

    Args:
        eqn: The JAXPR equation being processed
        known_in: Values for input variables (None for unknowns)
        known_out: Values for output variables (None for unknowns)

    Returns:
        ProcessedResult, tuple of (resolved_vars, resolved_vals), or None if the
        rule does not apply.
    """

    def __call__(
        self,
        eqn: JaxprEqn,
        known_in: Sequence[Any],
        known_out: Sequence[Any],
    ) -> Optional["ProcessedResult"]: ...


# Rule result type
RuleResult = Union[None, "ProcessedResult"]


# =============================================================================
# Context Definitions
# =============================================================================


class Context:
    """
    Standard interpreter contexts.

    Each context defines a different mode of processing JAXPR equations:
    - FORWARD: Execute equations normally (left to right)
    - INVERSE: Compute inverse (right to left, given outputs find inputs)
      Returns: (vars, vals) or None
    - INVERSE_LOGDET: Inverse with log-determinant tracking
      Returns: (vars, vals, log_det_updates) or None
    - LOG_PROB: Compute log probability contributions
    - TRACE: Collect intermediate values during execution
    - JOINT_SAMPLE: Sample and collect all random variables
    """

    FORWARD = "forward"
    INVERSE = "inverse"
    INVERSE_LOGDET = "inverse_logdet"
    LOG_PROB = "log_prob"
    TRACE = "trace"
    JOINT_SAMPLE = "joint_sample"


# =============================================================================
# Rule Registry
# =============================================================================


@dataclass
class ProcessedResult:
    """Result from processing an equation through a rule."""

    resolved_vars: Sequence[Atom]  # Can be Var or Literal
    resolved_vals: Sequence[Any]
    state: Any = None


def parse_processed_result(
    result: Optional["ProcessedResult"],
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
            f"Processing rules must return ProcessedResult or None, "
            f"got {type(result).__name__}"
        )

    # Normalize to sequences (handle single values)
    resolved_vars = result.resolved_vars
    resolved_vals = result.resolved_vals
    if not isinstance(resolved_vars, (list, tuple)):
        resolved_vars = [resolved_vars]
    if not isinstance(resolved_vals, (list, tuple)):
        resolved_vals = [resolved_vals]

    return tuple(resolved_vars), tuple(resolved_vals), result.state


class RuleRegistry:
    """
    Central registry mapping (primitive, context) → rule.

    This replaces the scattered dispatch logic in:
    - UNIVARIATE_INVERSE_REGISTRY
    - BIVARIATE_INVERSE_REGISTRY
    - CUSTOM_INVERSE_PROCESSING_RULES
    - InverseProcessingRule method dispatch
    - LogPotentialProcessingRule

    Usage:
        registry = RuleRegistry()

        @registry.rule(jax.lax.exp_p, Context.INVERSE)
        def exp_inverse(eqn, known_in, known_out):
            import jax.numpy as jnp
            return [eqn.invars[0]], [jnp.log(known_out[0])]

        # Or register directly
        registry.register(jax.lax.log_p, Context.INVERSE, log_inverse)

        # Process an equation
        result = registry.process(eqn, known_in, known_out, Context.INVERSE)
    """

    def __init__(self):
        # {context: {primitive: rule}}
        self._rules: Dict[str, Dict[Primitive, RuleFunction]] = {}
        # Fallback rules: {context: rule}
        self._fallbacks: Dict[str, RuleFunction] = {}

    def register(
        self,
        primitive: Primitive,
        context: str,
        rule: Callable[..., RuleResult],
    ) -> None:
        """
        Register a rule for a primitive in a context.

        Args:
            primitive: The JAX primitive this rule handles
            context: The interpreter context (e.g., Context.INVERSE)
            rule: The rule function to apply
        """
        if context not in self._rules:
            self._rules[context] = {}
        self._rules[context][primitive] = rule  # type: ignore[assignment]

    def rule(
        self,
        primitive: Primitive,
        context: str,
    ) -> Callable[[RuleFunction], RuleFunction]:
        """
        Decorator to register a rule.

        Usage:
            @registry.rule(jax.lax.exp_p, Context.INVERSE)
            def exp_inverse(eqn, known_in, known_out):
                return [eqn.invars[0]], [jnp.log(known_out[0])]
        """

        def decorator(fn: RuleFunction) -> RuleFunction:
            self.register(primitive, context, fn)
            return fn

        return decorator

    def register_fallback(self, context: str, rule: RuleFunction) -> None:
        """
        Register a fallback rule for a context.

        The fallback is used when no specific rule is registered for a primitive.
        """
        self._fallbacks[context] = rule

    def get(
        self,
        primitive: Primitive,
        context: str,
    ) -> Optional[RuleFunction]:
        """
        Get the rule for a primitive in a context.

        Returns None if no rule is registered.
        """
        context_rules = self._rules.get(context, {})
        return context_rules.get(primitive)

    def get_with_fallback(
        self,
        primitive: Primitive,
        context: str,
    ) -> Optional[RuleFunction]:
        """
        Get rule for a primitive, falling back to context default if not found.
        """
        rule = self.get(primitive, context)
        if rule is not None:
            return rule
        return self._fallbacks.get(context)

    def has_rule(self, primitive: Primitive, context: str) -> bool:
        """Check if a rule exists for a primitive in a context."""
        return self.get(primitive, context) is not None

    def process(
        self,
        eqn: JaxprEqn,
        known_in: Sequence[Any],
        known_out: Sequence[Any],
        context: str,
    ) -> Optional[ProcessedResult]:
        """
        Process an equation using the appropriate rule.

        Args:
            eqn: The equation to process
            known_in: Known input values (None for unknowns)
            known_out: Known output values (None for unknowns)
            context: The interpreter context

        Returns:
            ProcessedResult with resolved variables and values, or None
        """
        rule = self.get_with_fallback(eqn.primitive, context)
        if rule is None:
            return None

        result = rule(eqn, known_in, known_out)
        if result is None:
            return None

        if not isinstance(result, ProcessedResult):
            raise TypeError(
                "Rules must return ProcessedResult or None, got "
                f"{type(result).__name__}"
            )

        return result

    def list_contexts(self) -> list[str]:
        """List all registered contexts."""
        return list(self._rules.keys())

    def list_primitives(self, context: str) -> list[Primitive]:
        """List all primitives registered for a context."""
        return list(self._rules.get(context, {}).keys())


# =============================================================================
# Global Registry Instance
# =============================================================================

REGISTRY = RuleRegistry()


def _normalize_inverse_value(value: Any, aval: Any) -> Any:
    return jnp.asarray(value, dtype=aval.dtype)


def _inverse_values_match(actual: Any, expected: Any) -> Any:
    actual = jnp.asarray(actual)
    expected = jnp.asarray(expected)
    if actual.shape != expected.shape:
        return jnp.asarray(False)
    dtype = jnp.result_type(actual.dtype, expected.dtype)
    if jnp.issubdtype(dtype, jnp.inexact):
        return jnp.isclose(actual, expected, atol=1e-6, rtol=1e-6)
    return actual == expected


def validate_inverse_value(
    value: Any,
    aval: Any,
    replayed_output: Any,
    supplied_output: Any,
    *,
    message: str,
) -> tuple[Any, Any]:
    """Validate an inverse candidate by replaying the forward primitive."""
    value = _normalize_inverse_value(value, aval)
    valid = _inverse_values_match(replayed_output, supplied_output)
    all_valid = jnp.all(valid)

    if jnp.issubdtype(aval.dtype, jnp.inexact):
        if jnp.shape(valid) != jnp.shape(value):
            valid = all_valid
        invalid = jnp.full(jnp.shape(value), jnp.nan, dtype=aval.dtype)
        return jnp.where(valid, value, invalid), all_valid

    checkify.check(all_valid, message)
    return value, all_valid


def invalid_inverse_value(aval: Any, *, message: str) -> Any:
    """Materialize an invalid inverse while preserving the target aval."""
    if jnp.issubdtype(aval.dtype, jnp.inexact):
        return jnp.full(aval.shape, jnp.nan, dtype=aval.dtype)
    checkify.check(jnp.asarray(False), message)
    return jnp.zeros(aval.shape, dtype=aval.dtype)


# =============================================================================
# Registration Helpers
# =============================================================================


def register_univariate_inverse(
    forward_prim: Primitive,
    inverse_prim_or_fn: Union[Primitive, Callable],
    registry: RuleRegistry = REGISTRY,
) -> None:
    """
    Register a univariate inverse rule.

    For primitives like exp/log, sin/asin, etc. where the inverse is a simple
    function of the output.

    Args:
        forward_prim: The forward primitive (e.g., exp_p)
        inverse_prim_or_fn: The inverse primitive (e.g., log_p) or a function
        registry: The registry to register with (defaults to global REGISTRY)

    Example:
        register_univariate_inverse(jax.lax.exp_p, jax.lax.log_p)
        register_univariate_inverse(jax.lax.sqrt_p, lambda x, **p: x**2)
    """

    def rule(eqn: JaxprEqn, known_in: Sequence[Any], known_out: Sequence[Any]):
        del known_in  # Not needed for univariate inverse
        out_val = known_out[0]
        if out_val is None:
            return None

        if isinstance(inverse_prim_or_fn, Primitive):
            inv_prim = inverse_prim_or_fn
            result = bind_primitive(inv_prim, eqn.params, out_val)
        else:
            result = inverse_prim_or_fn(out_val, **sanitize_bind_params(eqn.params))

        replayed = bind_primitive(
            forward_prim, eqn.params, result, params_from=eqn.primitive
        )
        result, _ = validate_inverse_value(
            result,
            eqn.invars[0].aval,
            replayed,
            out_val,
            message=f"invalid inverse output for {forward_prim.name}",
        )

        return ProcessedResult([eqn.invars[0]], [result])

    registry.register(forward_prim, Context.INVERSE, rule)


def register_bivariate_inverse(
    prim: Primitive,
    left_inverse: Union[Primitive, Callable],
    right_inverse: Union[Primitive, Callable],
    registry: RuleRegistry = REGISTRY,
) -> None:
    """
    Register a bivariate inverse rule.

    For binary primitives like add, mul, etc. where we can solve for one
    input given the other and the output.

    Args:
        prim: The binary primitive (e.g., add_p, mul_p)
        left_inverse: Inverse when left input is unknown: f(out, right) -> left
        right_inverse: Inverse when right input is unknown: f(out, left) -> right
        registry: The registry to register with (defaults to global REGISTRY)

    Example:
        # For add: left = out - right, right = out - left
        register_bivariate_inverse(jax.lax.add_p, jax.lax.sub_p, jax.lax.sub_p)
    """

    def rule(eqn: JaxprEqn, known_in: Sequence[Any], known_out: Sequence[Any]):
        out_val = known_out[0]
        if out_val is None:
            return None

        left_known = known_in[0] is not None
        right_known = known_in[1] is not None

        # Need exactly one known input to solve
        if left_known == right_known:
            return None

        if left_known:
            # Left is known, solve for right: right = right_inverse(out, left)
            other = known_in[0]
            inv_fn = right_inverse
            target_var = eqn.invars[1]
        else:
            # Right is known, solve for left: left = left_inverse(out, right)
            other = known_in[1]
            inv_fn = left_inverse
            target_var = eqn.invars[0]

        if isinstance(inv_fn, Primitive):
            result = bind_primitive(inv_fn, eqn.params, out_val, other)
        else:
            result = inv_fn(out_val, other, **sanitize_bind_params(eqn.params))

        if left_known:
            replayed = bind_primitive(
                prim, eqn.params, other, result, params_from=eqn.primitive
            )
        else:
            replayed = bind_primitive(
                prim, eqn.params, result, other, params_from=eqn.primitive
            )
        result, _ = validate_inverse_value(
            result,
            target_var.aval,
            replayed,
            out_val,
            message=f"invalid inverse output for {prim.name}",
        )

        return ProcessedResult([target_var], [result])

    registry.register(prim, Context.INVERSE, rule)


# =============================================================================
# INVERSE_LOGDET Registration Helpers
# =============================================================================


def register_univariate_inverse_logdet(
    forward_prim: Primitive,
    inverse_prim_or_fn: Union[Primitive, Callable],
    logdet_fn: Callable[[Any, Any, dict], Any],
    registry: RuleRegistry = REGISTRY,
) -> None:
    """
    Register a univariate inverse+logdet rule with explicit logdet.
    """
    from jax.extend.core import Literal

    def rule(eqn: JaxprEqn, known_in: Sequence[Any], known_out: Sequence[Any]):
        del known_in
        out_val = known_out[0]
        if out_val is None:
            return None

        # Compute inverse value
        if isinstance(inverse_prim_or_fn, Primitive):
            inv_prim = inverse_prim_or_fn
            in_val = bind_primitive(inv_prim, eqn.params, out_val)
        else:
            in_val = inverse_prim_or_fn(out_val, **sanitize_bind_params(eqn.params))

        replayed = bind_primitive(
            forward_prim, eqn.params, in_val, params_from=eqn.primitive
        )
        in_val, valid = validate_inverse_value(
            in_val,
            eqn.invars[0].aval,
            replayed,
            out_val,
            message=f"invalid inverse output for {forward_prim.name}",
        )

        log_abs_det = logdet_fn(out_val, in_val, eqn.params)
        log_abs_det = jnp.where(valid, log_abs_det, jnp.asarray(jnp.nan))

        # Build state updates
        updates = {}
        if not isinstance(eqn.invars[0], Literal):
            updates[eqn.invars[0]] = log_abs_det

        return ProcessedResult([eqn.invars[0]], [in_val], updates)

    registry.register(forward_prim, Context.INVERSE_LOGDET, rule)


def register_bivariate_inverse_logdet(
    prim: Primitive,
    left_inverse: Union[Primitive, Callable],
    right_inverse: Union[Primitive, Callable],
    left_logdet_fn: Callable[[Any, Any, Any, dict], Any],
    right_logdet_fn: Callable[[Any, Any, Any, dict], Any],
    registry: RuleRegistry = REGISTRY,
) -> None:
    """
    Register a bivariate inverse+logdet rule with explicit logdet.
    """
    from jax.extend.core import Literal

    def rule(eqn: JaxprEqn, known_in: Sequence[Any], known_out: Sequence[Any]):
        out_val = known_out[0]
        if out_val is None:
            return None

        left_known = known_in[0] is not None
        right_known = known_in[1] is not None

        if left_known == right_known:
            return None

        if left_known:
            other = known_in[0]
            inv_fn = right_inverse
            logdet_fn = right_logdet_fn
            target_var = eqn.invars[1]
        else:
            other = known_in[1]
            inv_fn = left_inverse
            logdet_fn = left_logdet_fn
            target_var = eqn.invars[0]

        # Compute inverse value
        if isinstance(inv_fn, Primitive):
            result = bind_primitive(inv_fn, eqn.params, out_val, other)
        else:
            result = inv_fn(out_val, other, **sanitize_bind_params(eqn.params))

        if left_known:
            replayed = bind_primitive(
                prim, eqn.params, other, result, params_from=eqn.primitive
            )
        else:
            replayed = bind_primitive(
                prim, eqn.params, result, other, params_from=eqn.primitive
            )
        result, valid = validate_inverse_value(
            result,
            target_var.aval,
            replayed,
            out_val,
            message=f"invalid inverse output for {prim.name}",
        )

        log_abs_det = logdet_fn(out_val, result, other, eqn.params)
        log_abs_det = jnp.where(valid, log_abs_det, jnp.asarray(jnp.nan))

        # Build state updates
        updates = {}
        if not isinstance(target_var, Literal):
            updates[target_var] = log_abs_det

        return ProcessedResult([target_var], [result], updates)

    registry.register(prim, Context.INVERSE_LOGDET, rule)


# =============================================================================
# Forward Fallback
# =============================================================================


def forward_rule(
    eqn: JaxprEqn,
    known_in: Sequence[Any],
    known_out: Sequence[Any],
) -> Optional[ProcessedResult]:
    """
    Default forward execution rule.

    Executes the primitive with known inputs and returns the outputs.
    """
    del known_out  # Not used in forward execution

    # Check all inputs are known
    if any(v is None for v in known_in):
        return None

    primitive = eqn.primitive
    result = rebind_primitive(primitive, eqn.params, *known_in)

    if primitive.multiple_results:
        return ProcessedResult(resolved_vars=eqn.outvars, resolved_vals=list(result))
    return ProcessedResult(resolved_vars=eqn.outvars, resolved_vals=[result])


# Register forward as the fallback for forward context
REGISTRY.register_fallback(Context.FORWARD, forward_rule)


# =============================================================================
# Lazy JAXPR Construction
# =============================================================================


@dataclass
class LazyJaxpr:
    """
    A lazily-constructed JAXPR that only traces when needed.

    This avoids the eager tracing overhead in rv_p.bind() and custom_inverse()
    when the JAXPR is not actually needed (e.g., in pure eager execution).

    Usage:
        # Instead of tracing immediately:
        jaxpr = trace_fn(...)  # Always traces

        # Use a thunk that only traces when needed:
        lazy = LazyJaxpr(lambda: trace_fn(...))
        ...
        # Later, when actually needed:
        jaxpr = lazy.get()  # Only traces now
    """

    _thunk: Callable[[], ClosedJaxpr]
    _cached: Optional[ClosedJaxpr] = field(default=None, repr=False)

    def get(self) -> ClosedJaxpr:
        """Materialize the JAXPR, caching the result."""
        if self._cached is None:
            self._cached = self._thunk()
        return self._cached

    @property
    def is_materialized(self) -> bool:
        """Check if the JAXPR has been materialized."""
        return self._cached is not None

    def __repr__(self) -> str:
        status = "materialized" if self.is_materialized else "lazy"
        return f"LazyJaxpr({status})"


def lazy_jaxpr(thunk: Callable[[], ClosedJaxpr]) -> LazyJaxpr:
    """Create a lazy JAXPR from a thunk."""
    return LazyJaxpr(_thunk=thunk)

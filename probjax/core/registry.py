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

from contextlib import contextmanager
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
import numpy as np
from jax._src import core as jax_core
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


# -----------------------------------------------------------------------------
# Guards: how a rule reports that its inverse is undefined for the given values
# -----------------------------------------------------------------------------
#
# Most inverse rules need no guard at all. An inverse that is exact wherever it
# is defined -- log for exp, sub for add, the whole transcendental family --
# reports a domain violation for free: IEEE already makes ``log(-1)`` and
# ``atanh(2)`` NaN. Re-running the forward primitive to discover that costs
# about six equations per equation and, measured on an 8-deep chain, 3.2x the
# runtime of hand-written code.
#
# A guard is for the cases IEEE cannot express: dividing by a value that is only
# zero at runtime, or a branch whose choice depends on data. Those rules declare
# a predicate, and pay two ops rather than six.
#
# Note what a guard is *not* for: ``integer_pow(x, 2) = 4`` has two roots and no
# predicate can pick between them. Replaying the forward map does not help there
# either -- it confirms ``2**2 == 4`` and accepts the principal root -- so the
# rule documents the branch it takes instead of pretending to check it.

_INVERSE_CHECKS = False


@contextmanager
def inverse_checks(enabled: bool = True):
    """Make inverse guards raise through ``checkify`` instead of only NaN-ing.

    Off by default, and it must be: ``checkify.check`` cannot be staged out by a
    plain ``jit`` -- it raises "Cannot abstractly evaluate a checkify.check which
    was not functionalized" -- and an active checkify trace is not detectable
    from inside a rule. So the default is a NaN, which is always safe, and this
    switch adds the error channel for callers who are wrapping in
    ``checkify.checkify`` anyway:

    >>> with inverse_checks():
    ...     err, out = checkify.checkify(inverse(f))(y)
    """
    global _INVERSE_CHECKS
    previous = _INVERSE_CHECKS
    _INVERSE_CHECKS = enabled
    try:
        yield
    finally:
        _INVERSE_CHECKS = previous


def inverse_checks_enabled() -> bool:
    """Whether guards should additionally emit ``checkify.check``."""
    return _INVERSE_CHECKS


def guard_is_statically_satisfied(valid: Any) -> bool:
    """Whether a guard is decidable now, and passes.

    A guard on a literal operand -- ``2.0 * exp(x)``, where the 2.0 is baked
    into the jaxpr -- is a compile-time fact. Emitting a comparison and a select
    for it would put the per-primitive cost right back into every inverse, which
    is what this module exists to remove.
    """
    if isinstance(valid, jax_core.Tracer):
        return False
    # numpy, not jnp: inside an active trace even ``jnp.all(True)`` is staged
    # out, and the whole point here is to decide without emitting anything.
    return bool(np.all(np.asarray(valid)))


def apply_inverse_guard(value: Any, aval: Any, valid: Any, *, message: str) -> Any:
    """Mask ``value`` where ``valid`` is False, the outcome of a rule's guard."""
    if guard_is_statically_satisfied(valid):
        return jnp.asarray(value, dtype=aval.dtype)

    value = jnp.asarray(value, dtype=aval.dtype)
    valid = jnp.asarray(valid)

    if _INVERSE_CHECKS:
        checkify.check(jnp.all(valid), message)

    if not jnp.issubdtype(aval.dtype, jnp.inexact):
        # No integer NaN to fall back on; the guard is only reportable through
        # the checkify channel above.
        return value

    if jnp.shape(valid) != jnp.shape(value):
        valid = jnp.all(valid)
    return jnp.where(valid, value, jnp.asarray(jnp.nan, aval.dtype))


def inverse_roundtrip_valid(replayed_output: Any, supplied_output: Any) -> Any:
    """Guard predicate: did re-applying the forward map reproduce the output?

    Only for the few rules whose validity genuinely cannot be decided any other
    way -- a linear solve that may be inconsistent, a broadcast whose copies may
    disagree, a gather whose indices may not cover the input. Elementwise rules
    must not use this: it is the expensive check this module removed.
    """
    actual = jnp.asarray(replayed_output)
    expected = jnp.asarray(supplied_output)
    if actual.shape != expected.shape:
        return jnp.asarray(False)
    if jnp.issubdtype(jnp.result_type(actual.dtype, expected.dtype), jnp.inexact):
        return jnp.isclose(actual, expected, atol=1e-6, rtol=1e-6)
    return actual == expected


def invalid_inverse_value(aval: Any, *, message: str) -> Any:
    """Materialize an invalid inverse while preserving the target aval.

    Called where the rule knows *statically* that no unique inverse exists, so
    for integer dtypes -- which have no NaN to return -- this raises at trace
    time rather than handing back zeros that would be silently wrong.
    """
    if jnp.issubdtype(aval.dtype, jnp.inexact):
        return jnp.full(aval.shape, jnp.nan, dtype=aval.dtype)
    raise NotImplementedError(
        f"{message}; the result dtype {aval.dtype} has no NaN to signal it with."
    )


# =============================================================================
# Registration Helpers
# =============================================================================


def _resolve_branch_guard(guard, solving_for_left: bool):
    """Pick the guard for the branch being solved.

    ``guard`` mirrors ``left_inverse``/``right_inverse``: a pair is
    ``(guard_when_solving_for_left, guard_when_solving_for_right)``, and a bare
    callable applies to both. ``div`` needs the asymmetry -- recovering the
    numerator is exact, recovering the denominator divides by the output.
    """
    if guard is None:
        return None
    if isinstance(guard, tuple):
        return guard[0] if solving_for_left else guard[1]
    return guard


def register_univariate_inverse(
    forward_prim: Primitive,
    inverse_prim_or_fn: Union[Primitive, Callable],
    registry: RuleRegistry = REGISTRY,
    *,
    guard: Optional[Callable] = None,
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

        if guard is not None:
            result = apply_inverse_guard(
                result,
                eqn.invars[0].aval,
                guard(out_val, eqn.params),
                message=f"{forward_prim.name} has no inverse at this value",
            )

        return ProcessedResult([eqn.invars[0]], [result])

    registry.register(forward_prim, Context.INVERSE, rule)


def register_bivariate_inverse(
    prim: Primitive,
    left_inverse: Union[Primitive, Callable],
    right_inverse: Union[Primitive, Callable],
    registry: RuleRegistry = REGISTRY,
    *,
    guard: Optional[Callable] = None,
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

        branch_guard = _resolve_branch_guard(guard, solving_for_left=not left_known)
        if branch_guard is not None:
            result = apply_inverse_guard(
                result,
                target_var.aval,
                branch_guard(out_val, other, eqn.params),
                message=f"{prim.name} has no inverse at this value",
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
    *,
    guard: Optional[Callable] = None,
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

        log_abs_det = logdet_fn(out_val, in_val, eqn.params)

        if guard is not None:
            # The value is masked elementwise -- only the offending entries have
            # no preimage. The log-determinant is one scalar for the whole map,
            # so a single bad element poisons it.
            valid = guard(out_val, eqn.params)
            in_val = apply_inverse_guard(
                in_val,
                eqn.invars[0].aval,
                valid,
                message=f"{forward_prim.name} has no inverse at this value",
            )
            if not guard_is_statically_satisfied(valid):
                log_abs_det = jnp.where(
                    jnp.all(valid), log_abs_det, jnp.asarray(jnp.nan)
                )

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
    *,
    guard: Optional[Callable] = None,
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

        log_abs_det = logdet_fn(out_val, result, other, eqn.params)

        branch_guard = _resolve_branch_guard(guard, solving_for_left=not left_known)
        if branch_guard is not None:
            valid = branch_guard(out_val, other, eqn.params)
            result = apply_inverse_guard(
                result,
                target_var.aval,
                valid,
                message=f"{prim.name} has no inverse at this value",
            )
            if not guard_is_statically_satisfied(valid):
                log_abs_det = jnp.where(
                    jnp.all(valid), log_abs_det, jnp.asarray(jnp.nan)
                )

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

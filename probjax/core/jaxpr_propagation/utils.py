import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Protocol, Sequence

from jax.extend.core import JaxprEqn, Literal
from jaxtyping import Array

if TYPE_CHECKING:
    from probjax.core.registry import ProcessedResult


def primitive_bind_params(primitive, params) -> tuple[tuple[Any, ...], dict[str, Any]]:
    bind_params = primitive.get_bind_params(params)
    if (
        isinstance(bind_params, tuple)
        and len(bind_params) == 2
        and isinstance(bind_params[1], Mapping)
    ):
        subfuns, bind_params = bind_params
        return tuple(subfuns), dict(bind_params)
    return (), dict(bind_params)


def sanitize_bind_params(params) -> dict[str, Any]:
    """Drop default-valued dtype-hint params that not all primitives accept.

    Newer JAX attaches ``out_dtype=None`` to some nary primitives (e.g. mul).
    Inverse rules forward the forward eqn's params into a *different*
    primitive's bind (e.g. div as mul's inverse); a foreign param stages an
    eqn whose JVP rule rejects it.

    Only ever apply this when re-binding a *different* primitive: dropping the
    param is not a no-op for the primitive that owns it. ``mul_p`` binds
    happily without ``out_dtype``, but stages an eqn missing it, and its
    transpose rule then requires it -- so the failure surfaces much later, in
    the backward pass of any ``grad`` through the reconstructed jaxpr.
    """
    return {k: v for k, v in params.items() if not (k == "out_dtype" and v is None)}


def bind_primitive(primitive, params, *args, params_from=None):
    """Bind ``primitive`` with ``params`` read off some equation.

    ``params_from`` is the primitive those params came from. When it is the
    primitive being bound the params are passed through untouched; otherwise
    primitive-specific hints are stripped (see :func:`sanitize_bind_params`).
    """
    if params_from is not primitive:
        params = sanitize_bind_params(params)
    subfuns, bind_params = primitive_bind_params(primitive, params)
    return primitive.bind(*subfuns, *args, **bind_params)


# =============================================================================
# Knowness Type - Tracks variable knowledge state
# =============================================================================


class KnownessLevel(Enum):
    """Level of knowledge about a variable's value.

    UNKNOWN: No value has been computed yet.
    PARTIAL: Value exists but may be incomplete (e.g., array with some NaN
             placeholders). Can be updated/merged with additional information.
    COMPLETE: Value is fully determined and authoritative. Should not be
              overwritten by forward computation.
    """

    UNKNOWN = auto()
    PARTIAL = auto()
    COMPLETE = auto()


@dataclass(frozen=True, slots=True)
class Knowness:
    """
    Represents the state of knowledge about a variable.

    This type enables more nuanced control over how variables are processed
    during inverse propagation:

    - UNKNOWN: No value yet, waiting for computation
    - PARTIAL: Has a value that can be updated (e.g., accumulating slices)
    - COMPLETE: Authoritative value, forward computation should not overwrite

    Attributes:
        level: The level of knowledge (UNKNOWN, PARTIAL, or COMPLETE)
        value: The actual value (None only if UNKNOWN)
    """

    level: KnownessLevel
    value: Optional[Any] = None

    def __post_init__(self):
        if self.level == KnownessLevel.UNKNOWN and self.value is not None:
            raise ValueError("UNKNOWN knowness must have value=None")
        if self.level != KnownessLevel.UNKNOWN and self.value is None:
            raise ValueError(f"{self.level.name} knowness must have a value")

    @classmethod
    def unknown(cls) -> "Knowness":
        """Create an UNKNOWN knowness (no value yet)."""
        return cls(KnownessLevel.UNKNOWN, None)

    @classmethod
    def partial(cls, value: Any) -> "Knowness":
        """Create a PARTIAL knowness (value can be updated/merged)."""
        return cls(KnownessLevel.PARTIAL, value)

    @classmethod
    def complete(cls, value: Any) -> "Knowness":
        """Create a COMPLETE knowness (authoritative, don't overwrite)."""
        return cls(KnownessLevel.COMPLETE, value)

    @classmethod
    def from_value(cls, value: Optional[Any]) -> "Knowness":
        """Convert a legacy None/value to Knowness.

        None becomes UNKNOWN, any value becomes COMPLETE.
        This provides backward compatibility with code that uses
        None to represent unknown values.
        """
        if value is None:
            return cls.unknown()
        return cls.complete(value)

    @property
    def is_known(self) -> bool:
        """True if any value exists (PARTIAL or COMPLETE)."""
        return self.level != KnownessLevel.UNKNOWN

    @property
    def is_complete(self) -> bool:
        """True if value is authoritative (COMPLETE only)."""
        return self.level == KnownessLevel.COMPLETE

    @property
    def is_partial(self) -> bool:
        """True if value exists but can be updated (PARTIAL only)."""
        return self.level == KnownessLevel.PARTIAL

    @property
    def can_overwrite(self) -> bool:
        """True if forward computation can overwrite this value.

        UNKNOWN and PARTIAL can be overwritten, COMPLETE cannot.
        """
        return self.level in (KnownessLevel.UNKNOWN, KnownessLevel.PARTIAL)


# =============================================================================
# Environment - Stores intermediate computations
# =============================================================================


class Environment(dict):
    """A compute environment that stores intermediate computations.

    The Environment stores both raw values and their knowness levels.
    For backward compatibility, it can be used with just values (defaulting
    to COMPLETE knowness), or with explicit Knowness objects for finer control.
    """

    def __init__(self):
        super().__init__()
        self.eqn_states: dict[Any, Any] = {}
        self.eqn_states_by_namespace: dict[Any, dict[str, Any]] = {}
        self.run_states: dict[str, Any] = {}
        self._knowness: dict[Any, KnownessLevel] = {}

    def __getitem__(self, var: Any) -> Optional[Array]:
        if isinstance(var, Literal):
            return var.val
        elif var in self:
            return super().__getitem__(var)
        else:
            return None

    def __setitem__(self, var: Any, val: Array | None) -> None:
        if not isinstance(var, Literal):
            super().__setitem__(var, val)
            # Default to COMPLETE when setting raw values
            if val is not None:
                self._knowness[var] = KnownessLevel.COMPLETE

    def read(self, var: Any) -> Array | None:
        return self[var]

    def write(self, var: Any, val: Array | None) -> None:
        self[var] = val

    def read_knowness(self, var: Any) -> Knowness:
        """Read the knowness state of a variable.

        Returns a Knowness object representing the current state of knowledge
        about the variable.
        """
        if isinstance(var, Literal):
            return Knowness.complete(var.val)
        if var not in self:
            return Knowness.unknown()
        value = super().__getitem__(var)
        level = self._knowness.get(var, KnownessLevel.COMPLETE)
        return Knowness(level, value)

    def write_knowness(self, var: Any, knowness: Knowness) -> None:
        """Write a variable with explicit knowness level.

        Args:
            var: The variable to write
            knowness: A Knowness object containing value and level
        """
        if isinstance(var, Literal):
            return
        if knowness.level == KnownessLevel.UNKNOWN:
            # Remove from environment if setting to unknown
            self.pop(var, None)
            self._knowness.pop(var, None)
        else:
            super().__setitem__(var, knowness.value)
            self._knowness[var] = knowness.level

    def get_knowness_level(self, var: Any) -> KnownessLevel:
        """Get just the knowness level for a variable."""
        if isinstance(var, Literal):
            return KnownessLevel.COMPLETE
        if var not in self:
            return KnownessLevel.UNKNOWN
        return self._knowness.get(var, KnownessLevel.COMPLETE)

    def known(self, var: Any) -> bool:
        return isinstance(var, Literal) or var in self

    def read_state(self, eqn: Any, namespace: str | None = None) -> Any:
        if namespace is None:
            value = self.eqn_states.get(eqn)
            if value is not None:
                return value
            if isinstance(eqn, int):
                return self.eqn_states.get((eqn,))
            return None

        value = self.eqn_states_by_namespace.get(eqn, {}).get(namespace)
        if value is not None:
            return value
        if isinstance(eqn, int):
            return self.eqn_states_by_namespace.get((eqn,), {}).get(namespace)
        return None

    def write_state(
        self,
        eqn: Any,
        state: Any,
        namespace: str | None = None,
    ) -> None:
        if namespace is None:
            namespace = "default"
        self.eqn_states[eqn] = state
        states = self.eqn_states_by_namespace.get(eqn)
        if states is None:
            self.eqn_states_by_namespace[eqn] = {namespace: state}
        else:
            states[namespace] = state

    def read_run_state(self, namespace: str | None = None) -> Any:
        if namespace is None:
            namespace = "default"
        return self.run_states.get(namespace)

    def write_run_state(self, state: Any, namespace: str | None = None) -> None:
        if namespace is None:
            namespace = "default"
        self.run_states[namespace] = state


class ProcessingRuleFactory(Protocol):
    def __call__(self) -> "ProcessingRule | Callable[..., ProcessedResult | None]": ...


class CostFunction(Protocol):
    def __call__(
        self,
        eqn: JaxprEqn,
        known_invars: Sequence[bool],
        known_outvars: Sequence[bool],
    ) -> float: ...


class ReducerFunction(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def supports_context_argument(func: Callable, required_positional: int) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False

    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    return has_varargs or len(positional_params) >= required_positional


class ProcessingRule(ABC):
    """A processing rule for equations.

    All processing rules should return ProcessedResult or None.
    ProcessedResult contains resolved_vars, resolved_vals, and optional state.
    """

    def __init__(self, propagator: Callable | None = None):
        self.propagator = propagator

    @abstractmethod
    def __call__(
        self,
        eqn: JaxprEqn,
        known_inputs: Sequence[Any | None],
        known_outputs: Sequence[Any | None],
    ) -> "ProcessedResult | None":
        pass


class ForwardProcessingRule(ProcessingRule):
    """Default forward execution rule that evaluates primitives with known inputs."""

    def __call__(
        self,
        eqn: JaxprEqn,
        known_inputs: Sequence[Array | None],
        _: Sequence[Array | None],
    ) -> "ProcessedResult":
        from probjax.core.registry import ProcessedResult

        primitive = eqn.primitive
        outvals = bind_primitive(
            primitive, eqn.params, *known_inputs, params_from=primitive
        )
        if not eqn.primitive.multiple_results:
            outvals = [outvals]

        return ProcessedResult(eqn.outvars, outvals)


# Helper utilities intentionally kept minimal; runtime graph construction lives
# in `extended.py` and execution logic lives in `engine.py`.

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from probjax.utils.typing import Array, PyTree


@runtime_checkable
class TraceFilter(Protocol):
    """Protocol for specifying which parts of the state to trace.

    A filter receives a PyTree with the same structure as the solver state and
    should return either a PyTree selecting the components that should be
    recorded over time or ``None`` to disable tracing entirely.
    """

    def __call__(self, state: PyTree[Array]) -> Optional[PyTree[Array]]:
        ...


class TraceEverything:
    """Filter that records the entire solver state."""

    def __call__(self, state: PyTree[Array]) -> PyTree[Array]:
        return state


class TraceNothing:
    """Filter that disables tracing to minimize memory usage."""

    def __call__(self, state: PyTree[Array]) -> None:  # type: ignore[override]
        return None


from __future__ import annotations

from typing import Protocol, runtime_checkable

from probjax.utils.typing import Array, PyTree


@runtime_checkable
class TraceFilter(Protocol):
    """Protocol for specifying which parts of the state to trace.

    A filter receives a PyTree with the same structure as the solver state
    and returns the PyTree that is recorded over time. Use
    ``collect_trace=False`` to record nothing at all.
    """

    def __call__(self, state: PyTree[Array]) -> PyTree[Array]: ...


class TraceEverything:
    """Filter that records the entire solver state."""

    def __call__(self, state: PyTree[Array]) -> PyTree[Array]:
        return state

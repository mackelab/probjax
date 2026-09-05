from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import weakref

from jax.extend.core import Jaxpr, JaxprEqn, Literal

EqnId = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExtendedEquation:
    """Static equation wrapper with stable identity."""

    eqn: JaxprEqn
    eqn_id: EqnId
    local_index: int
    depth: int

    def __getattr__(self, name: str) -> Any:
        return getattr(self.eqn, name)


def construct_extended_graph(
    equations: Sequence[ExtendedEquation],
) -> dict[Any, tuple[EqnId, ...]]:
    neighbors: dict[Any, list[EqnId]] = {}
    for extended_eqn in equations:
        for var in extended_eqn.invars + extended_eqn.outvars:
            if isinstance(var, Literal):
                continue
            if var not in neighbors:
                neighbors[var] = [extended_eqn.eqn_id]
            else:
                neighbors[var].append(extended_eqn.eqn_id)
    return {var: tuple(eqn_ids) for var, eqn_ids in neighbors.items()}


def compute_last_used(
    jaxpr: Jaxpr,
    equations: Sequence[ExtendedEquation],
) -> dict[Any, EqnId | None]:
    """Return a map var -> equation id where it is used last.

    This mirrors JAX's `last_used` behavior but stores ExtendedEquation ids.
    """

    last_used: dict[Any, EqnId | None] = {
        v: None for v in jaxpr.outvars if not isinstance(v, Literal)
    }
    for equation in reversed(equations):
        for var in equation.invars:
            if not isinstance(var, Literal) and var not in last_used:
                last_used[var] = equation.eqn_id
    return last_used


_EXTENDED_JAXPR_CACHE: weakref.WeakKeyDictionary[
    Jaxpr, dict[EqnId, "ExtendedJaxpr"]
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class ExtendedJaxpr:
    """Jaxpr wrapper with static metadata for execution."""

    jaxpr: Jaxpr
    path_prefix: EqnId
    equations: tuple[ExtendedEquation, ...]
    neighbors: Mapping[Any, tuple[EqnId, ...]]
    equation_lookup: Mapping[EqnId, ExtendedEquation]
    last_used: Mapping[Any, EqnId | None]

    @classmethod
    def from_jaxpr(
        cls,
        jaxpr: Jaxpr,
        path_prefix: EqnId = (),
        *,
        use_cache: bool = True,
    ) -> "ExtendedJaxpr":
        if use_cache:
            cached_by_prefix = _EXTENDED_JAXPR_CACHE.get(jaxpr)
            if cached_by_prefix is not None:
                cached = cached_by_prefix.get(path_prefix)
                if cached is not None:
                    return cached

        equations = tuple(
            ExtendedEquation(
                eqn=eqn,
                eqn_id=path_prefix + (index,),
                local_index=index,
                depth=len(path_prefix),
            )
            for index, eqn in enumerate(jaxpr.eqns)
        )
        neighbors = construct_extended_graph(equations)
        equation_lookup = {eqn.eqn_id: eqn for eqn in equations}
        last_used = compute_last_used(jaxpr, equations)

        extended = cls(
            jaxpr=jaxpr,
            path_prefix=path_prefix,
            equations=equations,
            neighbors=neighbors,
            equation_lookup=equation_lookup,
            last_used=last_used,
        )

        if use_cache:
            cached_by_prefix = _EXTENDED_JAXPR_CACHE.get(jaxpr)
            if cached_by_prefix is None:
                cached_by_prefix = {}
                _EXTENDED_JAXPR_CACHE[jaxpr] = cached_by_prefix
            cached_by_prefix[path_prefix] = extended

        return extended

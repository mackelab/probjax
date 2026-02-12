from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum, auto
from itertools import product
from typing import Any, Mapping, cast

try:
    pjit_mod = importlib.import_module("jax.experimental.pjit")
except ImportError:
    pjit_mod = None

pjit_p = getattr(cast(Any, pjit_mod), "pjit_p", None)
if pjit_p is None:
    # JaX 0.7
    from jax._src.pjit import jit_p as pjit_p

from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p


@dataclass(frozen=True, slots=True)
class Knownness:
    in_all: bool
    in_any: bool
    out_all: bool


class PrimitiveKind(Enum):
    CUSTOM_CALL = auto()
    CUSTOM_RULE = auto()
    PJIT = auto()
    UNIVARIATE = auto()
    BIVARIATE = auto()
    OTHER = auto()


class DispatchAction(Enum):
    CUSTOM_CALL = auto()
    CUSTOM_RULE = auto()
    PJIT_MISSING_INPUTS = auto()
    RESOLVE_CONFLICT = auto()
    UNIVARIATE = auto()
    BIVARIATE = auto()
    FORWARD = auto()
    FAIL = auto()


_DIRECT_PRIMITIVE_KIND = {
    custom_inverse_call_p: PrimitiveKind.CUSTOM_CALL,
    pjit_p: PrimitiveKind.PJIT,
}


_REGISTRY_KIND_BY_ARITY = {
    (1, 1): PrimitiveKind.UNIVARIATE,
    (2, 1): PrimitiveKind.BIVARIATE,
}


def compute_knownness(known_invars, known_outvars) -> Knownness:
    in_all = True
    in_any = False
    for value in known_invars:
        known = value is not None
        in_all = in_all and known
        in_any = in_any or known

    out_all = True
    for value in known_outvars:
        if value is None:
            out_all = False
            break

    return Knownness(in_all=in_all, in_any=in_any, out_all=out_all)


def classify_primitive(
    eqn,
    *,
    custom_rules: Mapping[Any, Any],
    univariate_registry: Mapping[Any, Any],
    bivariate_registry: Mapping[Any, Any],
) -> PrimitiveKind:
    primitive = eqn.primitive
    direct_kind = _DIRECT_PRIMITIVE_KIND.get(primitive)
    if direct_kind is not None:
        return direct_kind

    if primitive in custom_rules:
        return PrimitiveKind.CUSTOM_RULE

    arity = (len(eqn.invars), len(eqn.outvars))
    candidate_kind = _REGISTRY_KIND_BY_ARITY.get(arity)
    if candidate_kind is PrimitiveKind.UNIVARIATE and primitive in univariate_registry:
        return PrimitiveKind.UNIVARIATE
    if candidate_kind is PrimitiveKind.BIVARIATE and primitive in bivariate_registry:
        return PrimitiveKind.BIVARIATE

    return PrimitiveKind.OTHER


def _select_dispatch_action_impl(
    primitive_kind: PrimitiveKind,
    knownness: Knownness,
    *,
    prefer_resolve_conflict: bool,
) -> DispatchAction:
    if knownness.out_all and primitive_kind is PrimitiveKind.CUSTOM_CALL:
        return DispatchAction.CUSTOM_CALL
    if knownness.out_all and primitive_kind is PrimitiveKind.CUSTOM_RULE:
        return DispatchAction.CUSTOM_RULE
    if primitive_kind is PrimitiveKind.PJIT and not knownness.in_all:
        return DispatchAction.PJIT_MISSING_INPUTS
    if prefer_resolve_conflict and knownness.in_all and knownness.out_all:
        return DispatchAction.RESOLVE_CONFLICT
    if knownness.out_all and primitive_kind is PrimitiveKind.UNIVARIATE:
        return DispatchAction.UNIVARIATE
    if (
        knownness.out_all
        and knownness.in_any
        and primitive_kind is PrimitiveKind.BIVARIATE
    ):
        return DispatchAction.BIVARIATE
    if knownness.in_all:
        return DispatchAction.FORWARD
    return DispatchAction.FAIL


def _build_dispatch_table() -> dict[
    tuple[PrimitiveKind, bool, bool, bool, bool], DispatchAction
]:
    table = {}
    for primitive_kind, in_all, in_any, out_all, prefer_resolve_conflict in product(
        PrimitiveKind,
        (False, True),
        (False, True),
        (False, True),
        (False, True),
    ):
        key = (primitive_kind, in_all, in_any, out_all, prefer_resolve_conflict)
        knownness = Knownness(in_all=in_all, in_any=in_any, out_all=out_all)
        table[key] = _select_dispatch_action_impl(
            primitive_kind,
            knownness,
            prefer_resolve_conflict=prefer_resolve_conflict,
        )
    return table


_DISPATCH_TABLE = _build_dispatch_table()


def select_dispatch_action(
    primitive_kind: PrimitiveKind,
    knownness: Knownness,
    *,
    prefer_resolve_conflict: bool,
) -> DispatchAction:
    key = (
        primitive_kind,
        knownness.in_all,
        knownness.in_any,
        knownness.out_all,
        prefer_resolve_conflict,
    )
    return _DISPATCH_TABLE[key]

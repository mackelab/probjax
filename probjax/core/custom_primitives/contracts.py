from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Union

from jax.extend.core import ClosedJaxpr


@dataclass(frozen=True, slots=True)
class CustomInverseCallParams:
    forward_jaxpr: ClosedJaxpr
    inverse_jaxpr_thunk: Callable[[], ClosedJaxpr]
    in_tree: Any
    out_tree: Any
    inv_argnum: int
    target_in_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RandomVariableCallParams:
    forward_jaxpr: ClosedJaxpr
    in_tree: Any
    shape: Any
    dist: Any
    name: str
    rvs_fn: Callable[..., Any]
    logpdf_fn: Callable[..., Any]
    kwds_items: tuple[tuple[str, Any], ...]


def _require_keys(
    params: Mapping[str, Any], required: Sequence[str], where: str
) -> None:
    missing = [key for key in required if key not in params]
    if missing:
        raise KeyError(f"{where} missing required params: {missing}")


def _resolve_lazy_forward(lazy_forward) -> ClosedJaxpr:
    """Resolve lazy forward jaxpr, evaluating the thunk if needed."""
    from probjax.core.custom_primitives.common import Lazy

    if isinstance(lazy_forward, Lazy):
        forward_jaxpr, _, _ = lazy_forward.get()
        return forward_jaxpr
    # Backwards compatibility: already a ClosedJaxpr
    return lazy_forward


def parse_custom_inverse_call_params(
    params: Mapping[str, Any],
) -> CustomInverseCallParams:
    # Support both 'lazy_forward' (new) and 'forward_jaxpr' (legacy)
    has_lazy = "lazy_forward" in params
    has_forward = "forward_jaxpr" in params

    if not has_lazy and not has_forward:
        raise KeyError(
            "custom_inverse_call missing required params: "
            "['lazy_forward'] or ['forward_jaxpr']"
        )

    _require_keys(
        params,
        (
            "inverse_jaxpr_thunk",
            "in_tree",
            "out_tree",
            "inv_argnum",
            "target_in_indices",
        ),
        where="custom_inverse_call",
    )

    # Resolve forward_jaxpr from lazy_forward or use directly
    if has_lazy:
        forward_jaxpr = _resolve_lazy_forward(params["lazy_forward"])
    else:
        forward_jaxpr = params["forward_jaxpr"]

    inverse_jaxpr_thunk = params["inverse_jaxpr_thunk"]
    in_tree = params["in_tree"]
    out_tree = params["out_tree"]
    inv_argnum = params["inv_argnum"]
    target_in_indices = tuple(params["target_in_indices"])

    if not isinstance(forward_jaxpr, ClosedJaxpr):
        raise TypeError("custom_inverse_call.forward_jaxpr must be a ClosedJaxpr.")
    if not callable(inverse_jaxpr_thunk):
        raise TypeError("custom_inverse_call.inverse_jaxpr_thunk must be callable.")
    if not isinstance(inv_argnum, int):
        raise TypeError("custom_inverse_call.inv_argnum must be an int.")
    if not target_in_indices or not all(
        isinstance(index, int) for index in target_in_indices
    ):
        raise TypeError(
            "custom_inverse_call.target_in_indices must be a non-empty int tuple."
        )

    return CustomInverseCallParams(
        forward_jaxpr=forward_jaxpr,
        inverse_jaxpr_thunk=inverse_jaxpr_thunk,
        in_tree=in_tree,
        out_tree=out_tree,
        inv_argnum=inv_argnum,
        target_in_indices=target_in_indices,
    )


def parse_random_variable_call_params(
    params: Mapping[str, Any],
) -> RandomVariableCallParams:
    _require_keys(
        params,
        (
            "forward_jaxpr",
            "in_tree",
            "shape",
            "dist",
            "name",
            "rvs_fn",
            "logpdf_fn",
            "kwds_items",
        ),
        where="random_variable",
    )
    forward_jaxpr = params["forward_jaxpr"]
    in_tree = params["in_tree"]
    shape = params["shape"]
    dist = params["dist"]
    name = params["name"]
    rvs_fn = params["rvs_fn"]
    logpdf_fn = params["logpdf_fn"]
    kwds_items = params["kwds_items"]

    if not isinstance(forward_jaxpr, ClosedJaxpr):
        raise TypeError("random_variable.forward_jaxpr must be a ClosedJaxpr.")
    if not isinstance(name, str):
        raise TypeError("random_variable.name must be a string.")
    if not callable(rvs_fn):
        raise TypeError("random_variable.rvs_fn must be callable.")
    if not callable(logpdf_fn):
        raise TypeError("random_variable.logpdf_fn must be callable.")
    if not isinstance(kwds_items, tuple):
        raise TypeError("random_variable.kwds_items must be a tuple.")

    return RandomVariableCallParams(
        forward_jaxpr=forward_jaxpr,
        in_tree=in_tree,
        shape=shape,
        dist=dist,
        name=name,
        rvs_fn=rvs_fn,
        logpdf_fn=logpdf_fn,
        kwds_items=kwds_items,
    )

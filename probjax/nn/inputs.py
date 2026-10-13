"""Named model inputs.

Besides ``data``, a generative model can receive any number of named inputs
(``context``, ``local_context``, ...). They travel as one ``dict`` through
training, sampling and density evaluation, and the model forwards them to its
network as keyword arguments, so supporting a new input needs no new plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def as_inputs(context: Any = None, inputs: Mapping[str, Any] | None = None):
    """One inputs dict, with ``context`` as sugar for ``inputs["context"]``.

    Returns ``None`` when there is nothing to pass.
    """
    merged = dict(inputs or {})
    if context is not None:
        if "context" in merged:
            raise ValueError("context was given both directly and in inputs")
        merged["context"] = context
    return merged or None


def merge_inputs(bound: Mapping[str, Any] | None, call: Mapping[str, Any] | None):
    """Per-call inputs override bound ones of the same name."""
    merged = {**(bound or {}), **(call or {})}
    return merged or None


def context_only(inputs: Mapping[str, Any] | None, family: str):
    """Unwrap the ``context`` of families that accept no other named input."""
    if not inputs:
        return None
    extra = sorted(set(inputs) - {"context"})
    if extra:
        raise ValueError(f"{family} only accepts a context input, got {extra}.")
    return inputs["context"]

"""Named model inputs.

Besides ``data``, a generative model can receive any number of named inputs
(``context``, ``local_context``, ...). They travel as one ``dict`` through
training, sampling and density evaluation, and the model forwards them to its
network as keyword arguments, so supporting a new input needs no new plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def context_only(inputs: Mapping[str, Any] | None, family: str):
    """Unwrap the ``context`` of families that accept no other named input."""
    if not inputs:
        return None
    extra = sorted(set(inputs) - {"context"})
    if extra:
        raise ValueError(f"{family} only accepts a context input, got {extra}.")
    return inputs["context"]

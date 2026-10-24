"""Exact samplers built on rejection.

Each step of these kernels returns one independent, exact draw from the
target, so :func:`~probjax.inference.run` collects iid samples. They share the
kernel contract with MCMC: what tightens over time lives in the state (the
:class:`ars` envelope) or in params tuned by an adaptor (the
:class:`rejection` bound and learned proposal).
"""

from probjax.inference.rejection.adaptive import ARSInfo, ARSState, ars
from probjax.inference.rejection.proposal import (
    RejectionInfo,
    RejectionParams,
    RejectionState,
    bound_adaptor,
    rejection,
)

__all__ = [
    "ARSInfo",
    "ARSState",
    "RejectionInfo",
    "RejectionParams",
    "RejectionState",
    "ars",
    "bound_adaptor",
    "rejection",
]

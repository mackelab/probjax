"""Sequential Monte Carlo methods (lazy re-exports).

Lazy (PEP 562): importing this package does not import any SMC implementation.
``filtering.particle_filter`` imports the ``smc.ess`` and ``smc.resampling``
leaves, which must not pull in the SMC kernels with them. Submodules remain
directly importable.
"""

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_SYMBOLS = {
    "SMCInfo": "probjax.inference.smc.tempering",
    "SMCState": "probjax.inference.smc.tempering",
    "smc": "probjax.inference.smc.tempering",
    "acceptance_rate_adaptor": "probjax.inference.smc.adaptation",
    "particle_adaptor": "probjax.inference.smc.adaptation",
    "DiffusionSMCInfo": "probjax.inference.smc.diffusion",
    "DiffusionSMCState": "probjax.inference.smc.diffusion",
    "Guide": "probjax.inference.smc.diffusion",
    "diffusion_smc": "probjax.inference.smc.diffusion",
    "score_guide": "probjax.inference.smc.diffusion",
    "TemporalSMCInfo": "probjax.inference.smc.temporal",
    "TemporalSMCState": "probjax.inference.smc.temporal",
    "temporal_smc": "probjax.inference.smc.temporal",
    "population_diagnostics": "probjax.inference.smc.temporal_utils",
    "summarize_replicates": "probjax.inference.smc.temporal_utils",
}

if TYPE_CHECKING:  # lets static tools (docs, type checkers) resolve the lazy names
    from probjax.inference.smc.adaptation import (
        acceptance_rate_adaptor,
        particle_adaptor,
    )
    from probjax.inference.smc.diffusion import (
        DiffusionSMCInfo,
        DiffusionSMCState,
        Guide,
        diffusion_smc,
        score_guide,
    )
    from probjax.inference.smc.tempering import SMCInfo, SMCState, smc
    from probjax.inference.smc.temporal import (
        TemporalSMCInfo,
        TemporalSMCState,
        temporal_smc,
    )
    from probjax.inference.smc.temporal_utils import (
        population_diagnostics,
        summarize_replicates,
    )

__all__ = sorted(_LAZY_SYMBOLS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SYMBOLS:
        value = getattr(importlib.import_module(_LAZY_SYMBOLS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(list(globals().keys()) + __all__)

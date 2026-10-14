"""Sequential Monte Carlo methods (lazy re-exports).

Lazy (PEP 562): importing this package does not import any SMC implementation.
This matters because ``filtering.particle_filter`` imports the ``smc.ess`` and
``smc.resampling`` leaves; an eager ``smc/__init__`` would pull
``smc.temporal`` -> ``filtering.temporal`` during that import and recreate the
cycle the filtering refactor removed. Submodules remain directly importable.
"""

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_SYMBOLS = {
    "tuning": "probjax.inference.smc.tuning",
    "adaptive_smc": "probjax.inference.smc.adaptive",
    "adaptive_persistent_smc": "probjax.inference.smc.adaptive_persistent_smc",
    "GeometricPath": "probjax.inference.smc.path",
    "PartialPosteriorsPath": "probjax.inference.smc.path",
    "path_smc": "probjax.inference.smc.path_smc",
    "persistent_smc": "probjax.inference.smc.persistent_smc",
    "PretrainedTwistInfo": "probjax.inference.smc.pretrained_twist",
    "PretrainedTwistState": "probjax.inference.smc.pretrained_twist",
    "run_pretrained_diffusion_smc": "probjax.inference.smc.pretrained_twist",
    "RDSMCInfo": "probjax.inference.smc.rdsmc",
    "RDSMCState": "probjax.inference.smc.rdsmc",
    "run_rdsmc": "probjax.inference.smc.rdsmc",
    "acceptance_rate_adaptor": "probjax.inference.smc.tuning",
    "particle_adaptor": "probjax.inference.smc.tuning",
    "pretuned_smc": "probjax.inference.smc.ports",
    "tuned_smc": "probjax.inference.smc.ports",
    "waste_free_smc": "probjax.inference.smc.ports",
    "waste_free_strategy": "probjax.inference.smc.ports",
    "TemporalSMC": "probjax.inference.smc.temporal",
    "TemporalSMCInfo": "probjax.inference.smc.temporal",
    "TemporalSMCState": "probjax.inference.smc.temporal",
    "run_temporal_smc": "probjax.inference.smc.temporal",
    "temporal_smc": "probjax.inference.smc.temporal",
    "population_diagnostics": "probjax.inference.smc.temporal_utils",
    "summarize_replicates": "probjax.inference.smc.temporal_utils",
}

if TYPE_CHECKING:  # lets static tools (docs, type checkers) resolve the lazy names
    from probjax.inference.smc import tuning
    from probjax.inference.smc.adaptive import (
        adaptive_smc,
    )
    from probjax.inference.smc.adaptive_persistent_smc import (
        adaptive_persistent_smc,
    )
    from probjax.inference.smc.path import (
        GeometricPath,
        PartialPosteriorsPath,
    )
    from probjax.inference.smc.path_smc import (
        path_smc,
    )
    from probjax.inference.smc.persistent_smc import (
        persistent_smc,
    )
    from probjax.inference.smc.ports import (
        pretuned_smc,
        tuned_smc,
        waste_free_smc,
        waste_free_strategy,
    )
    from probjax.inference.smc.pretrained_twist import (
        PretrainedTwistInfo,
        PretrainedTwistState,
        run_pretrained_diffusion_smc,
    )
    from probjax.inference.smc.rdsmc import (
        RDSMCInfo,
        RDSMCState,
        run_rdsmc,
    )
    from probjax.inference.smc.temporal import (
        TemporalSMC,
        TemporalSMCInfo,
        TemporalSMCState,
        run_temporal_smc,
        temporal_smc,
    )
    from probjax.inference.smc.temporal_utils import (
        population_diagnostics,
        summarize_replicates,
    )
    from probjax.inference.smc.tuning import (
        acceptance_rate_adaptor,
        particle_adaptor,
    )

__all__ = sorted([
    *_LAZY_SYMBOLS,
    "smc",
    "adaptive_smc_kernel",
    "persistent_smc_kernel",
    "adaptive_persistent_smc_kernel",
])


def __getattr__(name: str) -> Any:
    if name in _LAZY_SYMBOLS:
        if name in ("tuning",):
            value = importlib.import_module(_LAZY_SYMBOLS[name])
        else:
            module = importlib.import_module(_LAZY_SYMBOLS[name])
            value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(list(globals().keys()) + __all__)


def smc(*, path=None, **kwargs):
    """Build an SMC kernel from a path object/callable."""
    from probjax.inference.smc.path import GeometricPath
    from probjax.inference.smc.path_smc import path_smc

    if path is None:
        path = GeometricPath()
    if "logprior_fn" not in kwargs:
        kwargs["logprior_fn"] = None
    if "loglikelihood_fn" not in kwargs:
        kwargs["loglikelihood_fn"] = None
    if hasattr(path, "logdensity_fn"):
        return path_smc(path=path, **kwargs)
    if callable(path):
        return path_smc(path=path, **kwargs)
    raise TypeError("path must define logdensity_fn or be callable.")


def adaptive_smc_kernel(*, path=None, **kwargs):
    from probjax.inference.smc.adaptive import adaptive_smc
    from probjax.inference.smc.path import GeometricPath

    if path is None:
        path = GeometricPath()
    return adaptive_smc(path=path, **kwargs)


def persistent_smc_kernel(**kwargs):
    """Build a persistent SMC kernel (geometric path only)."""
    from probjax.inference.smc.persistent_smc import persistent_smc

    return persistent_smc(**kwargs)


def adaptive_persistent_smc_kernel(**kwargs):
    """Build an adaptive persistent SMC kernel (geometric path only)."""
    from probjax.inference.smc.adaptive_persistent_smc import adaptive_persistent_smc

    return adaptive_persistent_smc(**kwargs)

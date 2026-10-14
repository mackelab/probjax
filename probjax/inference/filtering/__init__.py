"""Bayesian filtering, smoothing and trajectory sampling.

Lazy re-exports (PEP 562): importing this package does not import any filter
implementation. Symbols resolve on first attribute access, and submodules
(``probjax.inference.filtering.temporal`` etc.) remain directly importable.
This keeps ``particle_filter`` (which uses ``smc`` leaf utilities) free of
import cycles without function-local imports.
"""

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_SYMBOLS = {
    # base
    "FilterAPI": "probjax.inference.filtering.base",
    "FilterInfo": "probjax.inference.filtering.base",
    "FilterKernel": "probjax.inference.filtering.base",
    "FilterState": "probjax.inference.filtering.base",
    "make_filter_api": "probjax.inference.filtering.base",
    # state space models (leaf types)
    "LinearGaussianModel": "probjax.inference.filtering.state_space",
    "StateSpaceModel": "probjax.inference.filtering.state_space",
    "TemporalFilter": "probjax.inference.filtering.state_space",
    "TemporalResult": "probjax.inference.filtering.state_space",
    "TemporalTrace": "probjax.inference.filtering.state_space",
    # observation validation (leaf)
    "validate_observations": "probjax.inference.filtering.observations",
    # shared Gaussian math (leaf)
    "GaussianFilterInfo": "probjax.inference.filtering.gaussian",
    "GaussianFilterState": "probjax.inference.filtering.gaussian",
    "default_logdet": "probjax.inference.filtering.gaussian",
    "default_solve": "probjax.inference.filtering.gaussian",
    "gaussian_filter_template": "probjax.inference.filtering.gaussian",
    "gaussian_init": "probjax.inference.filtering.gaussian",
    "gaussian_unpack": "probjax.inference.filtering.gaussian",
    "innovation_loglik": "probjax.inference.filtering.gaussian",
    "kalman_update": "probjax.inference.filtering.gaussian",
    # concrete filters
    "kalman_filter": "probjax.inference.filtering.kalman_filter",
    "extended_kalman_filter": "probjax.inference.filtering.extended_kalman_filter",
    "make_continuous_transition": "probjax.inference.filtering.extended_kalman_filter",
    "make_linearized_observation": "probjax.inference.filtering.extended_kalman_filter",
    "make_linearized_transition": "probjax.inference.filtering.extended_kalman_filter",
    "ukf": "probjax.inference.filtering.unscented_kalman_filter",
    "sq_kalman_filter": "probjax.inference.filtering.square_root_kf",
    "rank_reduced_kalman_filter": "probjax.inference.filtering.rank_reduced_kalman_filter",
    "ParticleFilter": "probjax.inference.filtering.particle_filter",
    # smoothing
    "particle_smoother": "probjax.inference.filtering.smoothing",
    "rauch_tung_stribel_smoother": "probjax.inference.filtering.smoothing",
    "smooth": "probjax.inference.filtering.smoothing",
    # orchestration
    "run_temporal_filter": "probjax.inference.filtering.temporal",
    "temporal_filter": "probjax.inference.filtering.temporal",
    "scan_history": "probjax.inference.filtering.temporal",
    "particle_gibbs": "probjax.inference.filtering.trajectory",
    "sample_paths": "probjax.inference.filtering.trajectory",
    "smooth_gaussian_path": "probjax.inference.filtering.trajectory",
    "sample_joint_paths": "probjax.inference.filtering.joint",
}

if TYPE_CHECKING:  # lets static tools (docs, type checkers) resolve the lazy names
    from probjax.inference.filtering.base import (
        FilterAPI,
        FilterInfo,
        FilterKernel,
        FilterState,
        make_filter_api,
    )
    from probjax.inference.filtering.extended_kalman_filter import (
        extended_kalman_filter,
        make_continuous_transition,
        make_linearized_observation,
        make_linearized_transition,
    )
    from probjax.inference.filtering.gaussian import (
        GaussianFilterInfo,
        GaussianFilterState,
        default_logdet,
        default_solve,
        gaussian_filter_template,
        gaussian_init,
        gaussian_unpack,
        innovation_loglik,
        kalman_update,
    )
    from probjax.inference.filtering.joint import (
        sample_joint_paths,
    )
    from probjax.inference.filtering.kalman_filter import (
        kalman_filter,
    )
    from probjax.inference.filtering.observations import (
        validate_observations,
    )
    from probjax.inference.filtering.particle_filter import (
        ParticleFilter,
    )
    from probjax.inference.filtering.rank_reduced_kalman_filter import (
        rank_reduced_kalman_filter,
    )
    from probjax.inference.filtering.smoothing import (
        particle_smoother,
        rauch_tung_stribel_smoother,
        smooth,
    )
    from probjax.inference.filtering.square_root_kf import (
        sq_kalman_filter,
    )
    from probjax.inference.filtering.state_space import (
        LinearGaussianModel,
        StateSpaceModel,
        TemporalFilter,
        TemporalResult,
        TemporalTrace,
    )
    from probjax.inference.filtering.temporal import (
        run_temporal_filter,
        scan_history,
        temporal_filter,
    )
    from probjax.inference.filtering.trajectory import (
        particle_gibbs,
        sample_paths,
        smooth_gaussian_path,
    )
    from probjax.inference.filtering.unscented_kalman_filter import (
        ukf,
    )

__all__ = sorted(_LAZY_SYMBOLS)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SYMBOLS:
        module = importlib.import_module(_LAZY_SYMBOLS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(list(globals().keys()) + __all__)

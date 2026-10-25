"""Bayesian filtering, smoothing and trajectory sampling.

Every filter is a :class:`~probjax.inference.Kernel` built from a model:
``init(key, theta, t0=0.0)`` and ``step(key, state, theta, t, y,
observed=True)``, with the model parameters ``theta`` as its params, so
``run(key, filter, state, theta, len(ts), args=(ts, ys))`` filters a series
and ``state.log_likelihood`` is the log-likelihood of ``theta``.

Lazy re-exports (PEP 562): importing this package does not import any filter
implementation. Symbols resolve on first attribute access, and submodules
remain directly importable.
"""

import importlib
from typing import TYPE_CHECKING, Any

_LAZY_SYMBOLS = {
    "GaussianModel": "probjax.inference.filtering.state_space",
    "LinearGaussianModel": "probjax.inference.filtering.state_space",
    "StateSpaceModel": "probjax.inference.filtering.state_space",
    "validate_observations": "probjax.inference.filtering.observations",
    "GaussianFilterInfo": "probjax.inference.filtering.gaussian",
    "GaussianFilterState": "probjax.inference.filtering.gaussian",
    "default_logdet": "probjax.inference.filtering.gaussian",
    "default_solve": "probjax.inference.filtering.gaussian",
    "innovation_loglik": "probjax.inference.filtering.gaussian",
    "kalman_update": "probjax.inference.filtering.gaussian",
    "kalman_filter": "probjax.inference.filtering.kalman_filter",
    "extended_kalman_filter": "probjax.inference.filtering.extended_kalman_filter",
    "unscented_kalman_filter": "probjax.inference.filtering.unscented_kalman_filter",
    "merwe_sigma_point": "probjax.inference.filtering.unscented_kalman_filter",
    "julier_uhlmann_sigma_points": "probjax.inference.filtering.unscented_kalman_filter",
    "spherical_simplex_sigma_points": "probjax.inference.filtering.unscented_kalman_filter",
    "square_root_kalman_filter": "probjax.inference.filtering.square_root_kf",
    "SquareRootFilterInfo": "probjax.inference.filtering.square_root_kf",
    "SquareRootFilterState": "probjax.inference.filtering.square_root_kf",
    "rank_reduced_kalman_filter": "probjax.inference.filtering.rank_reduced_kalman_filter",
    "RankReducedKalmanFilterInfo": "probjax.inference.filtering.rank_reduced_kalman_filter",
    "RankReducedKalmanFilterState": "probjax.inference.filtering.rank_reduced_kalman_filter",
    "particle_filter": "probjax.inference.filtering.particle_filter",
    "ParticleFilterInfo": "probjax.inference.filtering.particle_filter",
    "ParticleFilterState": "probjax.inference.filtering.particle_filter",
    "particle_gibbs": "probjax.inference.filtering.trajectory",
    "sample_paths": "probjax.inference.filtering.trajectory",
    "smooth": "probjax.inference.filtering.trajectory",
    "JointPathSamples": "probjax.inference.filtering.joint",
    "sample_joint_paths": "probjax.inference.filtering.joint",
}

if TYPE_CHECKING:  # lets static tools (docs, type checkers) resolve the lazy names
    from probjax.inference.filtering.extended_kalman_filter import (
        extended_kalman_filter,
    )
    from probjax.inference.filtering.gaussian import (
        GaussianFilterInfo,
        GaussianFilterState,
        default_logdet,
        default_solve,
        innovation_loglik,
        kalman_update,
    )
    from probjax.inference.filtering.joint import (
        JointPathSamples,
        sample_joint_paths,
    )
    from probjax.inference.filtering.kalman_filter import (
        kalman_filter,
    )
    from probjax.inference.filtering.observations import (
        validate_observations,
    )
    from probjax.inference.filtering.particle_filter import (
        ParticleFilterInfo,
        ParticleFilterState,
        particle_filter,
    )
    from probjax.inference.filtering.rank_reduced_kalman_filter import (
        RankReducedKalmanFilterInfo,
        RankReducedKalmanFilterState,
        rank_reduced_kalman_filter,
    )
    from probjax.inference.filtering.square_root_kf import (
        SquareRootFilterInfo,
        SquareRootFilterState,
        square_root_kalman_filter,
    )
    from probjax.inference.filtering.state_space import (
        GaussianModel,
        LinearGaussianModel,
        StateSpaceModel,
    )
    from probjax.inference.filtering.trajectory import (
        particle_gibbs,
        sample_paths,
        smooth,
    )
    from probjax.inference.filtering.unscented_kalman_filter import (
        julier_uhlmann_sigma_points,
        merwe_sigma_point,
        spherical_simplex_sigma_points,
        unscented_kalman_filter,
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

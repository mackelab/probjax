from typing import Any


def require_ott() -> tuple[Any, ...]:
    """Import OTT components used by optional optimal-transport features."""
    try:
        # OTT is an optional extra, so it is not installed in every environment.
        from ott.geometry import costs, pointcloud  # type: ignore[import-not-found]
        from ott.initializers.linear.initializers import (  # type: ignore[import-not-found]
            GaussianInitializer,
        )
        from ott.problems.linear import linear_problem  # type: ignore[import-not-found]
        from ott.solvers.linear import sinkhorn  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError(
            "This operation requires OTT. Install it with "
            "`pip install 'probjax[wasserstein]'`."
        ) from error
    return costs, pointcloud, GaussianInitializer, linear_problem, sinkhorn


def require_lineax() -> Any:
    """Import lineax, used for optional lineax solvers in linear operators."""
    try:
        # lineax is an optional extra, so it is not installed in every environment.
        import lineax  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError(
            "This operation requires lineax. Install it with "
            "`pip install 'probjax[lineax]'`."
        ) from error
    return lineax

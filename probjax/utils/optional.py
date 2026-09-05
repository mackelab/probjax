def require_ott():
    """Import OTT components used by optional optimal-transport features."""
    try:
        from ott.geometry import costs, pointcloud
        from ott.initializers.linear.initializers import GaussianInitializer
        from ott.problems.linear import linear_problem
        from ott.solvers.linear import sinkhorn
    except ImportError as error:
        raise ImportError(
            "This operation requires OTT. Install it with "
            "`pip install 'probjax[wasserstein]'`."
        ) from error
    return costs, pointcloud, GaussianInitializer, linear_problem, sinkhorn

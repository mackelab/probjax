from probjax.inference.smc.path_smc import path_smc
from probjax.inference.smc.path import GeometricPath, PartialPosteriorsPath
from probjax.inference.smc.adaptive import adaptive_smc
from probjax.inference.smc.persistent_smc import persistent_smc
from probjax.inference.smc.adaptive_persistent_smc import adaptive_persistent_smc
from probjax.inference.smc import tuning


def smc(*, path=None, **kwargs):
    """Build an SMC kernel from a path object/callable."""
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
    if path is None:
        path = GeometricPath()
    return adaptive_smc(path=path, **kwargs)


def persistent_smc_kernel(**kwargs):
    """Build a persistent SMC kernel (geometric path only)."""
    return persistent_smc(**kwargs)


def adaptive_persistent_smc_kernel(**kwargs):
    """Build an adaptive persistent SMC kernel (geometric path only)."""
    return adaptive_persistent_smc(**kwargs)

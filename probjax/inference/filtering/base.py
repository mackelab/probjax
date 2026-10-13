from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
from jax.typing import ArrayLike

from probjax.utils.guards import has_attrs
from probjax.utils.jaxutils import API


class FilterState(NamedTuple):
    """This is a NamedTuple that represents the state of a filter.

    It contains all the information **required** to run the filter.
    """

    pass


class FilterInfo(NamedTuple):
    """This is a NamedTuple that represents the information returned by a filter.

    It contains all useful information that can be extracted from the filter.

    """

    pass


def _default_unpack(
    state: FilterState, info: FilterInfo
) -> Tuple[FilterState, FilterInfo]:
    return (state, info)


class FilterKernel(NamedTuple):
    """This is a NamedTuple that represents a filter kernel."""

    init: Callable
    step: Callable
    default_unpack: Callable = _default_unpack

    def __call__(
        self,
        state: FilterState,
        t: Optional[ArrayLike] = None,
        observed: Optional[ArrayLike] = None,
        rng_key: Optional[ArrayLike] = None,
    ) -> Tuple[FilterState, FilterInfo]:
        return self.step(state, t, observed, rng_key)


class FilterAPI(metaclass=API):
    @staticmethod
    def init(*args, **kwargs) -> FilterState:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def build_kernel(*args, **kwargs) -> FilterKernel:
        raise NotImplementedError("build_kernel method must be implemented")

    default_unpack = staticmethod(_default_unpack)

    def __new__(cls, *args, **kwargs) -> FilterKernel:
        return FilterKernel(
            cls.init, cls.build_kernel(*args, **kwargs), cls.default_unpack
        )


def make_filter_api(
    *,
    name: str,
    init_fn: Callable,
    build_kernel_fn: Callable,
    default_unpack_fn: Optional[Callable] = None,
):
    """Create a FilterAPI subclass with minimal boilerplate.

    Args:
        name: Name for the new class.
        init_fn: Function to initialize filter state.
        build_kernel_fn: Function that builds the filter step function.
        default_unpack_fn: Optional function to extract output from (state, info).
            Defaults to returning (state, info).
    """
    attrs = {
        "init": staticmethod(init_fn),
        "build_kernel": staticmethod(build_kernel_fn),
    }
    if default_unpack_fn is not None:
        attrs["default_unpack"] = staticmethod(default_unpack_fn)

    return type(name, (FilterAPI,), attrs)


def is_particle_state(states: Any) -> bool:
    """True if ``states`` carries particles + log weights."""
    return has_attrs(states, "particles", "log_weights")


def extract_mean(states: Any) -> ArrayLike:
    """Mean for Gaussian states, particles for particle states."""
    if hasattr(states, "mean"):
        return states.mean
    if hasattr(states, "particles"):
        return states.particles
    raise ValueError("Trace states have no 'mean' or 'particles' attribute.")


def extract_covariance(states: Any) -> ArrayLike:
    """Covariance for Gaussian states (``cov`` / ``std`` / factor-core)."""
    if hasattr(states, "cov"):
        return states.cov
    if hasattr(states, "std"):
        return jax.vmap(lambda s: s @ s.T)(states.std)
    if has_attrs(states, "cov_factor", "cov_core"):
        return jax.vmap(lambda u, s: u @ s @ u.T)(states.cov_factor, states.cov_core)
    raise ValueError(
        "Filtering states must have either `cov`, `std`, or (`cov_factor`, `cov_core`)."
    )


def extract_pred_covariance(infos: Any) -> ArrayLike:
    """Predicted covariance for Gaussian infos."""
    if hasattr(infos, "cov_pred"):
        return infos.cov_pred
    if hasattr(infos, "std_pred"):
        return jax.vmap(lambda s: s @ s.T)(infos.std_pred)
    if has_attrs(infos, "cov_factor_pred", "cov_core_pred"):
        return jax.vmap(lambda u, s: u @ s @ u.T)(
            infos.cov_factor_pred, infos.cov_core_pred
        )
    raise ValueError(
        "Filtering infos must have either `cov_pred`, `std_pred`, "
        "or (`cov_factor_pred`, `cov_core_pred`)."
    )

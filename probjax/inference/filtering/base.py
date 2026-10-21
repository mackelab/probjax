from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from probjax.utils.guards import has_attrs
from probjax.utils.jaxutils import KernelAPI
from probjax.utils.typing import Array, ArrayLike, RngKey


def as_time(t: Optional[ArrayLike]) -> Optional[Array]:
    """Normalize an optional time argument to an array, keeping ``None`` as ``None``."""
    return None if t is None else jnp.asarray(t)


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
        rng_key: Optional[RngKey] = None,
    ) -> Tuple[FilterState, FilterInfo]:
        return self.step(state, t, observed, rng_key)


class FilterAPI(KernelAPI):
    """Base class for filters.

    Subclass it and set ``init`` and ``build_step`` as static methods.
    ``cls(**model_functions)`` builds a :class:`FilterKernel`;
    ``cls.configure(**options)`` binds options before the model's functions
    are known.
    """

    _kernel_type = FilterKernel

    @staticmethod
    def init(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def init_params(*args: Any, **kwargs: Any) -> Any:
        """Filters have no tunable parameters."""
        return None

    @staticmethod
    def build_step(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("build_step method must be implemented")

    @staticmethod
    def build_kernel(*args: Any, **kwargs: Any) -> Any:
        """The original name of :meth:`build_step`; still accepted."""
        raise NotImplementedError("build_step method must be implemented")

    default_unpack = staticmethod(_default_unpack)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # ``build_kernel`` was the original name; ``build_step`` is the name
        # every other kernel API uses. Accept both, and prefer ``build_step``.
        if "build_kernel" in cls.__dict__ and "build_step" not in cls.__dict__:
            cls.build_step = cls.__dict__["build_kernel"]

    def __new__(cls, *args: Any, **kwargs: Any) -> FilterKernel:
        return cls.kernel_type(
            cls.init, cls.build_step(*args, **kwargs), cls.default_unpack
        )


def make_filter_api(
    *,
    name: str,
    init_fn: Callable,
    build_kernel_fn: Callable,
    default_unpack_fn: Optional[Callable] = None,
) -> type:
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
        "build_step": staticmethod(build_kernel_fn),
        "build_kernel": staticmethod(build_kernel_fn),
    }
    if default_unpack_fn is not None:
        attrs["default_unpack"] = staticmethod(default_unpack_fn)

    return type(name, (FilterAPI,), attrs)


def is_particle_state(states: Any) -> bool:
    """True if ``states`` carries particles + log weights."""
    return has_attrs(states, "particles", "log_weights")


def extract_mean(states: Any) -> Array:
    """Mean for Gaussian states, particles for particle states."""
    if hasattr(states, "mean"):
        return states.mean
    if hasattr(states, "particles"):
        return states.particles
    raise ValueError("Trace states have no 'mean' or 'particles' attribute.")


def extract_covariance(states: Any, pred: bool = False) -> Array:
    """Covariance for Gaussian states (``cov`` / ``std`` / factor-core).

    With ``pred=True`` the predicted fields (``cov_pred`` / ``std_pred`` /
    ``cov_factor_pred`` / ``cov_core_pred``) are read instead.
    """
    suffix = "_pred" if pred else ""
    if hasattr(states, f"cov{suffix}"):
        return getattr(states, f"cov{suffix}")
    if hasattr(states, f"std{suffix}"):
        return jax.vmap(lambda s: s @ s.T)(getattr(states, f"std{suffix}"))
    if has_attrs(states, f"cov_factor{suffix}", f"cov_core{suffix}"):
        return jax.vmap(lambda u, s: u @ s @ u.T)(
            getattr(states, f"cov_factor{suffix}"),
            getattr(states, f"cov_core{suffix}"),
        )
    kind = "infos" if pred else "states"
    raise ValueError(
        f"Filtering {kind} must have either `cov{suffix}`, `std{suffix}`, "
        f"or (`cov_factor{suffix}`, `cov_core{suffix}`)."
    )


def extract_pred_covariance(infos: Any) -> Array:
    """Predicted covariance for Gaussian infos."""
    return extract_covariance(infos, pred=True)

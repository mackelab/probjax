from typing import Any, Callable, NamedTuple, Optional, Tuple

from jax.typing import ArrayLike

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


def _default_unpack(state, info):
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
    def init(*args, **kwargs) -> Any:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def build_kernel(*args, **kwargs) -> Any:
        raise NotImplementedError("build_kernel method must be implemented")

    @staticmethod
    def default_unpack(state, info):
        """Default unpack function. Override in subclasses to customize."""
        return (state, info)

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

    Analogous to make_kernel_api() in the MCMC module.

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

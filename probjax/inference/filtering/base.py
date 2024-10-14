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


class FilterKernel(NamedTuple):
    """This is a NamedTuple that represents a filter kernel."""

    init: Callable
    step: Callable

    def __call__(
        self,
        state: FilterState,
        t: Optional[ArrayLike] = None,
        observed: Optional[ArrayLike] = None,
        rng: Optional[ArrayLike] = None,
    ) -> Tuple[FilterState, FilterInfo]:
        return self.step(state, t, observed, rng)


class FilterAPI(metaclass=API):
    @staticmethod
    def init(*args, **kwargs) -> Any:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def build_kernel(*args, **kwargs) -> Any:
        raise NotImplementedError("build_kernel method must be implemented")

    def __new__(cls, *args, **kwargs) -> FilterKernel:
        return FilterKernel(cls.init, cls.build_kernel(*args, **kwargs))

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, NamedTuple, Callable
from jax.typing import ArrayLike


class FilterState(NamedTuple):
    pass
    
class FilterInfo(NamedTuple):
    pass


class FilterKernel(ABC):
    

    _kernel : Callable
    

    @staticmethod
    def init_fn(*args, **kwargs) -> FilterState:
        pass
    
    @staticmethod
    def build_kernel(*args, **kwargs) -> Callable:
        pass
    
    def __init__(self, *args, **kwargs) -> None:
        self._kernel = type(self).build_kernel(*args, **kwargs)
    
    def init(self, *args, **kwargs) -> FilterState:
        return type(self).init_fn(*args, **kwargs)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._kernel(*args, **kwds)
    

    
    
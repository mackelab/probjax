from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple


from jaxtyping import Key

from probjax.utils.jaxutils import API

SMCState = NamedTuple
SMCInfo = NamedTuple


class SMCKernel(NamedTuple):
    init: Callable
    step: Callable

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)


class SMCKernelAPI(metaclass=API):
    @staticmethod
    def init(position, rng_key: Optional[Key] = None, **kwargs) -> SMCState:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def build_step(*args, **kwargs) -> Callable:
        raise NotImplementedError("build_kernel method must be implemented")

    def __new__(cls, logdensity_fn: Callable, **kwargs) -> SMCKernel:
        init = partial(cls.init, logdensity_fn=logdensity_fn)
        step = cls.build_step(logdensity_fn, **kwargs)
        return SMCKernel(init, step)

from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

from blackjax.base import Info, State
from jaxtyping import Key

from probjax.utils.jaxutils import API


def ignore_kwargs(fn: Callable, *keys) -> Callable:
    def wrapped_fn(*args, **kwargs):
        for key in keys:
            del kwargs[key]
        return fn(*args, **kwargs)

    return wrapped_fn


class Params(NamedTuple):
    """BlackJAX only distinguishes between State and Info, but we will add Params for
    convenience.

    This should contain all the parameters that can be optimized in the kernel.
    """

    pass


class MarkovKernel(NamedTuple):
    """This is a NamedTuple that represents a Markov kernel with a stationary distribution.
    given by the logdensity_fn.
    """

    logdensity_fn: Callable
    init: Callable
    step: Callable
    init_params: Callable
    adapt_params: Callable

    def __call__(
        self, key: Key, state: State, params: Optional[Params] = None
    ) -> Tuple[State, Info]:
        return self.step(key, state, params)


class MarkovKernelAPI(metaclass=API):
    @staticmethod
    def init(position, rng_key: Optional[Key] = None, **kwargs) -> State:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def init_params(position, *args, **kwargs) -> Params:
        raise NotImplementedError("init_params method must be implemented")

    @staticmethod
    def build_step(*args, **kwargs) -> Callable:
        raise NotImplementedError("build_kernel method must be implemented")

    @staticmethod
    def build_adaptation(*args, **kwargs) -> Callable:
        def no_adaptation(*args, **kwargs) -> Tuple[State, Info]:
            raise NotImplementedError("No adaption method has been implemented")

        return no_adaptation

    def __new__(cls, logdensity_fn: Callable, **kwargs) -> MarkovKernel:
        init = partial(cls.init, logdensity_fn=logdensity_fn)
        step = cls.build_step(logdensity_fn, **kwargs)
        adapt_params = cls.build_adaptation(logdensity_fn, **kwargs)

        return MarkovKernel(logdensity_fn, init, step, cls.init_params, adapt_params)

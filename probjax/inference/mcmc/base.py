from functools import partial
import inspect
from typing import Callable, NamedTuple, Optional, Tuple

from blackjax.base import Info, State
from probjax.utils.typing import RngKey

from probjax.utils.jaxutils import API


def ignore_kwargs(fn: Callable, *keys) -> Callable:
    def wrapped_fn(*args, **kwargs):
        for key in keys:
            kwargs.pop(key, None)
        return fn(*args, **kwargs)

    return wrapped_fn


class Params(NamedTuple):
    """BlackJAX only distinguishes between State and Info, but we will add Params for
    convenience.

    This should contain all the parameters that can be optimized in the kernel.
    """

    pass


class MarkovKernel(NamedTuple):
    """This is a NamedTuple that represents a Markov kernel with a stationary
    distribution given by the logdensity_fn.
    """

    logdensity_fn: Callable
    init: Callable
    step: Callable
    init_params: Callable
    fit_params: Callable

    def __call__(
        self, key: RngKey, state: State, params: Optional[Params] = None, *args
    ) -> Tuple[State, Info]:
        return self.step(key, state, params, *args)


class MarkovKernelAPI(metaclass=API):
    @staticmethod
    def init(position, rng_key: Optional[RngKey] = None, **kwargs) -> State:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def init_params(state, *args, **kwargs) -> Params:
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
        init_fn = partial(cls.init, logdensity_fn=logdensity_fn)
        raw_step = cls.build_step(logdensity_fn, **kwargs)
        fit_params = cls.build_adaptation(logdensity_fn, **kwargs)

        # Wrap step to accept (and ignore) extra *args intended for
        # logdensity_fn.  SG-MCMC kernels override step directly and
        # forward *args to the grad estimator.
        def step(key, state, params, *args):
            return raw_step(key, state, params)

        return MarkovKernel(logdensity_fn, init_fn, step, cls.init_params, fit_params)


def make_kernel_api(
    *,
    name: str,
    init_fn: Callable,
    init_params_fn: Callable,
    build_step_fn: Callable,
    build_adaptation_fn: Optional[Callable] = None,
):
    """Create a MarkovKernelAPI subclass with minimal boilerplate."""
    sig = inspect.signature(init_fn)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    accepts_rng_key = "rng_key" in sig.parameters or accepts_kwargs

    def _init_with_rng(*args, rng_key=None, **kwargs):
        if accepts_rng_key:
            return init_fn(*args, rng_key=rng_key, **kwargs)
        return init_fn(*args, **kwargs)

    attrs = {
        "init": staticmethod(_init_with_rng),
        "init_params": staticmethod(init_params_fn),
        "build_step": staticmethod(build_step_fn),
    }
    if build_adaptation_fn is not None:
        attrs["build_adaptation"] = staticmethod(build_adaptation_fn)

    return type(name, (MarkovKernelAPI,), attrs)


def make_step_from_kernel(
    logdensity_fn: Callable,
    kernel_builder: Callable,
    *,
    builder_kwargs: Optional[dict] = None,
    call_defaults: Optional[dict] = None,
) -> Callable:
    """Build a step function from a BlackJAX-style kernel builder."""
    kernel = kernel_builder(**builder_kwargs) if builder_kwargs else kernel_builder()

    def step(key: RngKey, state: State, params: Params, *args):
        params_kwargs = params._asdict() if hasattr(params, "_asdict") else {}
        call_kwargs = {}
        if call_defaults:
            call_kwargs.update(call_defaults)
        call_kwargs.update(params_kwargs)
        return kernel(key, state, logdensity_fn, **call_kwargs)

    return step

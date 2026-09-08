import inspect
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
from blackjax.smc import base as bj_smc_base

from probjax.inference.base import Kernel
from probjax.utils.jaxutils import API
from probjax.utils.typing import Array, PyTree, RngKey

SMCState = bj_smc_base.SMCState
SMCInfo = bj_smc_base.SMCInfo


SMCKernel = Kernel


class SMCKernelAPI(metaclass=API):
    @staticmethod
    def init(particles, rng_key: Optional[RngKey] = None, **kwargs) -> SMCState:
        raise NotImplementedError("init method must be implemented")

    @staticmethod
    def init_params(particles, rng_key: Optional[RngKey] = None, **kwargs) -> Dict:
        raise NotImplementedError("init_params method must be implemented")

    @staticmethod
    def build_step(*args, **kwargs) -> Callable:
        raise NotImplementedError("build_step method must be implemented")

    def __new__(cls, logprior_fn: Callable, loglikelihood_fn: Callable, **kwargs):
        init = partial(
            cls.init,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            **_filter_kwargs(cls.init, kwargs, allow_kwargs=False),
        )
        init_params = partial(
            cls.init_params, logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_fn
        )
        step = cls.build_step(logprior_fn, loglikelihood_fn, **kwargs)
        return SMCKernel(init, step, init_params)


def _filter_kwargs(
    fn: Callable, kwargs: Dict[str, Any], *, allow_kwargs: bool = True
) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    if accepts_kwargs and allow_kwargs:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _ensure_param_batch(params, *, shared: bool) -> Dict[str, Array]:
    params = _params_to_dict(params)
    batched = {}
    for k, v in params.items():
        arr = jnp.asarray(v)
        if shared:
            if arr.ndim == 0 or arr.shape[0] != 1:
                arr = arr[None, ...]
            # Ensure shared params are at least 2-D so that
            # BlackJAX's unshared_parameters_and_step_fn (which does
            # v[0, ...] on shared params) preserves the inner array.
            if arr.ndim == 1:
                arr = arr[None, ...]
        else:
            if arr.ndim == 0:
                arr = arr[None]
        batched[k] = arr
    return batched


def _params_to_dict(params) -> Dict[str, Any]:
    if params is None:
        return {}
    if hasattr(params, "_asdict"):
        return params._asdict()
    if isinstance(params, Mapping):
        return dict(params)
    if is_dataclass(params):
        return asdict(params)
    raise TypeError("mcmc_params must be a mapping, NamedTuple, or dataclass")


class _ParamsProxy:
    def __init__(self, params: Dict[str, Any]):
        self._params = params
        for k, v in params.items():
            setattr(self, k, v)

    def _asdict(self) -> Dict[str, Any]:
        return self._params


def _wrap_params(params: Dict[str, Any]):
    if params is None:
        return _ParamsProxy({})
    if hasattr(params, "_asdict"):
        return params
    if is_dataclass(params):
        return _ParamsProxy(asdict(params))
    return _ParamsProxy(params)


def make_mcmc_adapter(mcmc_kernel, **kernel_kwargs):
    """Adapt a probjax MCMC API into BlackJAX SMC mcmc_init_fn/mcmc_step_fn."""
    if not hasattr(mcmc_kernel, "build_step") or not hasattr(mcmc_kernel, "init"):
        raise TypeError("mcmc_kernel must be a probjax MCMC API (not an instance).")

    def mcmc_init_fn(position, logdensity_fn, rng_key: Optional[RngKey] = None):
        init_kwargs = _filter_kwargs(
            mcmc_kernel.init, kernel_kwargs, allow_kwargs=False
        )
        return mcmc_kernel.init(position, logdensity_fn, rng_key=rng_key, **init_kwargs)

    def mcmc_step_fn(rng_key, state, logdensity_fn, **step_parameters):
        build_kwargs = _filter_kwargs(mcmc_kernel.build_step, kernel_kwargs)
        step = mcmc_kernel.build_step(logdensity_fn, **build_kwargs)
        return step(rng_key, state, _wrap_params(step_parameters))

    return mcmc_init_fn, mcmc_step_fn


def init_mcmc_params(
    particles: PyTree,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    mcmc_kernel,
    rng_key: Optional[RngKey] = None,
    lmbda: float = 0.0,
    mcmc_kernel_kwargs: Optional[Dict[str, Any]] = None,
    **mcmc_param_kwargs,
) -> Dict[str, Array]:
    """Initialize MCMC params from the first particle at tempering level lmbda."""
    particle0 = jax.tree_util.tree_map(lambda x: x[0], particles)

    def tempered_logposterior_fn(position):
        return logprior_fn(position) + lmbda * loglikelihood_fn(position)

    mcmc_kernel_kwargs = mcmc_kernel_kwargs or {}
    mcmc_init_fn, _ = make_mcmc_adapter(mcmc_kernel, **mcmc_kernel_kwargs)
    mcmc_state = mcmc_init_fn(particle0, tempered_logposterior_fn, rng_key=rng_key)
    params = mcmc_kernel.init_params(mcmc_state, **mcmc_param_kwargs)
    return _ensure_param_batch(params, shared=True)


def make_smc_api(
    *,
    name: str,
    init_fn: Callable,
    init_params_fn: Callable,
    build_step_fn: Callable,
):
    attrs = {
        "init": staticmethod(init_fn),
        "init_params": staticmethod(init_params_fn),
        "build_step": staticmethod(build_step_fn),
    }
    return type(name, (SMCKernelAPI,), attrs)

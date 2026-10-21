from __future__ import annotations

import inspect
from functools import partial, wraps
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from blackjax.base import State
from jax.flatten_util import ravel_pytree

from probjax.inference.base import Kernel, MCMCResult
from probjax.utils.guards import has_attrs, params_to_dict
from probjax.utils.jaxutils import Configured, KernelAPI
from probjax.utils.typing import Array, RngKey, Scalar


class Params(NamedTuple):
    """BlackJAX only distinguishes between State and Info, but we will add Params for
    convenience.

    This should contain all the parameters that can be optimized in the kernel.
    """

    pass


MarkovKernel = Kernel


class MarkovKernelAPI(KernelAPI):
    """Base class for MCMC kernels.

    Subclass it and set ``init``, ``init_params`` and ``build_step`` as static
    methods. Document the kernel in the subclass docstring: ``cls(logdensity_fn,
    **options)`` forwards ``options`` to ``build_step`` and returns a
    :class:`Kernel`. Calling the class with options only --
    ``hmc(num_integration_steps=10)`` -- returns a
    :class:`~probjax.utils.jaxutils.Configured` kernel instead.
    """

    _kernel_type = MarkovKernel

    #: Optional factory returning this kernel's recommended adaptation
    #: policy (an Adaptor or a Warmup). Attached to built kernels;
    #: wrappers (e.g. pseudo-marginal) inherit it.
    default_adaptation_fn: Callable[..., Any] | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Callers always pass the rng, but many BlackJAX inits take none.
        init = cls.init
        params = inspect.signature(init).parameters.values()
        if not any(
            p.name in ("rng", "rng_key") or p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params
        ):

            @wraps(init)
            def init_with_rng(*args, rng=None, **kw):
                return init(*args, **kw)

            cls.init = staticmethod(init_with_rng)

    def __new__(
        cls, logdensity_fn: Callable | None = None, **kwargs
    ) -> MarkovKernel | Configured:
        if logdensity_fn is None:
            return Configured(cls, tuple(sorted(kwargs.items())))

        def init_fn(rng: RngKey, position: Any, **init_kwargs):
            # The API's init may be BlackJAX's (``rng_key``) or ours (``rng``).
            rng_kwarg = (
                "rng_key" if "rng_key" in inspect.signature(cls.init).parameters else "rng"
            )
            return cls.init(
                position, logdensity_fn=logdensity_fn, **{rng_kwarg: rng}, **init_kwargs
            )

        raw_step = cls.build_step(logdensity_fn, **kwargs)

        def step(rng: RngKey, state, params, *args):
            return raw_step(rng, state, params)

        return cls.kernel_type(
            init_fn, step, cls.init_params, cls.default_adaptation_fn
        )


def make_step_from_kernel(
    logdensity_fn: Callable,
    kernel_builder: Callable,
    *,
    call_defaults: dict[str, Any] | None = None,
) -> Callable:
    """Build a step function from a BlackJAX-style kernel builder."""
    kernel = kernel_builder()

    def step(rng, state: State, params: Params, *args: Any):
        params_kwargs = params_to_dict(params) if has_attrs(params, "_asdict") else {}
        call_kwargs = {}
        if call_defaults:
            call_kwargs.update(call_defaults)
        call_kwargs.update(params_kwargs)
        return kernel(rng, state, logdensity_fn, **call_kwargs)

    return step


def _select(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: getattr(value, field) for field in fields}


def ravel_position(state: Any) -> Array:
    """Flatten the position of ``state`` (or of a bare position) into one vector.

    Accepts either a kernel state carrying a ``position`` attribute or the
    position itself, and returns the flat vector of its leaves.
    """
    position = state.position if hasattr(state, "position") else state
    flat_position, _ = ravel_pytree(position)
    return flat_position


def init_hmc_like_params(state: Any, inverse_mass_matrix: Array | None) -> Array:
    """Default inverse mass matrix: identity of matching dimension."""
    dim = ravel_position(state).shape[0]
    if inverse_mass_matrix is None:
        return jnp.ones((dim,))
    if inverse_mass_matrix.shape[0] != dim:
        raise ValueError(
            "The dimension of the inverse mass matrix must match the dimension"
            " of the position."
        )
    return inverse_mass_matrix


def scale_step_size_by_grad(
    state: Any, step_size: float, inverse_mass_matrix: Array | None = None
) -> Scalar:
    """Scale ``step_size`` by the gradient norm so first steps stay sane."""
    grad = getattr(state, "logdensity_grad", None)
    if grad is None:
        return step_size
    flat_grad, _ = ravel_pytree(grad)
    if inverse_mass_matrix is not None:
        inv = inverse_mass_matrix
        if inv.ndim == 0:
            grad_norm = jnp.sqrt(inv) * jnp.linalg.norm(flat_grad)
        elif inv.ndim == 1:
            grad_norm = jnp.sqrt(jnp.sum(inv * flat_grad**2))
        elif inv.ndim == 2:
            grad_norm = jnp.sqrt(flat_grad @ (inv @ flat_grad))
        else:
            raise ValueError("Invalid inverse_mass_matrix shape")
    else:
        grad_norm = jnp.linalg.norm(flat_grad)
    return step_size / jnp.where(grad_norm == 0, 1.0, grad_norm)


def _reshape_thin_args(args: Any, num_samples: int, thin: int) -> Any:
    """Reshape leading axes of ``args`` to ``(num_samples, thin, ...)``."""
    if args is None:
        return None
    return jax.tree.map(
        lambda value: value.reshape((num_samples, thin) + value.shape[1:]),
        args,
    )


def _mcmc_collect_body(
    kernel: MarkovKernel,
    params: Params,
    step_args: Any,
    collect_info: tuple[str, ...],
    stats_fn: Callable[[Any, Any], Any] | None = None,
) -> Callable[[Any, Any], tuple[Any, tuple[Any, ...]]]:
    """Two-level scan body shared by silent and verbose ``sample``.

    Yields ``(stats, position, diagnostics)`` per collected sample, where
    ``stats`` is ``None`` when no ``stats_fn`` is supplied.
    """

    def transition(state, inner_xs):
        if step_args is None:
            step_key, current_args = inner_xs, ()
        else:
            step_key, current_args = inner_xs
        state, info = kernel(step_key, state, params, *current_args)
        return state, (info, _select(info, collect_info))

    def collect_one(state, xs):
        if step_args is None:
            sample_keys, sample_args = xs, None
        else:
            sample_keys, sample_args = xs
        inner_xs = sample_keys if sample_args is None else (sample_keys, sample_args)

        state, (step_info, selected_info) = jax.lax.scan(transition, state, inner_xs)
        diagnostics = jax.tree.map(lambda value: value[-1], selected_info)
        stats = None
        if stats_fn is not None:
            last_info = jax.tree.map(lambda value: value[-1], step_info)
            stats = stats_fn(state, last_info)
        return state, (stats, state.position, diagnostics)

    return collect_one


@partial(
    jax.jit,
    static_argnames=(
        "kernel",
        "num_samples",
        "thin",
        "collect_info",
    ),
)
def sample_kernel(
    rng,
    kernel: MarkovKernel,
    state: State,
    num_samples: int,
    params: Params,
    args: Any = None,
    *,
    thin: int = 1,
    collect_info: tuple[str, ...] = (),
) -> MCMCResult:
    """Collect positions from a compiled kernel scan."""
    keys = jax.random.split(rng, (num_samples, thin))
    step_args = _reshape_thin_args(args, num_samples, thin)
    xs = keys if step_args is None else (keys, step_args)

    collect_one = _mcmc_collect_body(kernel, params, step_args, collect_info)

    state, (_, samples, info) = jax.lax.scan(collect_one, state, xs)
    return MCMCResult(state, samples, info if collect_info else None)

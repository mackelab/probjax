from __future__ import annotations

import inspect
from functools import partial, wraps
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from probjax.inference.base import Kernel
from probjax.utils.guards import has_attrs, params_to_dict
from probjax.utils.jaxutils import Configured, KernelAPI
from probjax.utils.typing import Array, RngKey, Scalar


class MarkovKernelAPI(KernelAPI):
    """Base class for MCMC kernels.

    Subclass it and set ``init``, ``init_params`` and ``build_step`` as static
    methods. Document the kernel in the subclass docstring: ``cls(logdensity_fn,
    **options)`` forwards ``options`` to ``build_step`` and returns a
    :class:`Kernel`. Calling the class with options only --
    ``hmc(num_integration_steps=10)`` -- returns a
    :class:`~probjax.utils.jaxutils.Configured` kernel instead.
    """

    _kernel_type = Kernel

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
    ) -> Kernel | Configured:
        if logdensity_fn is None:
            return Configured(cls, tuple(sorted(kwargs.items())))

        def init_fn(rng: RngKey, position: Any, **init_kwargs):
            # The API's init may be BlackJAX's (``rng_key``) or ours (``rng``).
            rng_kwarg = (
                "rng_key"
                if "rng_key" in inspect.signature(cls.init).parameters
                else "rng"
            )
            return cls.init(
                position, logdensity_fn=logdensity_fn, **{rng_kwarg: rng}, **init_kwargs
            )

        raw_step = cls.build_step(logdensity_fn, **kwargs)

        def step(rng: RngKey, state, params, *args):
            return raw_step(rng, state, params)

        default_adaptation = bind_default_adaptation(
            cls.default_adaptation_fn, logdensity_fn=logdensity_fn, **kwargs
        )
        return cls.kernel_type(init_fn, step, cls.init_params, default_adaptation)


def bind_default_adaptation(
    factory: Callable[..., Any] | None, **options: Any
) -> Callable[..., Any] | None:
    """Bind the construction options a kernel's adaptation factory asks for.

    Policies that need the target (MCLMC's tuning) take ``logdensity_fn``;
    policies that need a build option (IMH's proposal) take that option.
    """
    if factory is None:
        return None
    parameters = inspect.signature(factory).parameters
    bound = {name: value for name, value in options.items() if name in parameters}
    return partial(factory, **bound) if bound else factory


def make_step_from_kernel(
    logdensity_fn: Callable,
    kernel_builder: Callable,
    *,
    call_defaults: dict[str, Any] | None = None,
) -> Callable:
    """Build a step function from a BlackJAX-style kernel builder."""
    kernel = kernel_builder()

    def step(rng, state: Any, params: Any, *args: Any):
        params_kwargs = params_to_dict(params) if has_attrs(params, "_asdict") else {}
        call_kwargs = {}
        if call_defaults:
            call_kwargs.update(call_defaults)
        call_kwargs.update(params_kwargs)
        return kernel(rng, state, logdensity_fn, **call_kwargs)

    return step


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


# ---------------------------------------------------------------------------
# Dynamic trajectory-length stepping (dynamic HMC and MCLMC)
# ---------------------------------------------------------------------------
#
# Both kernels pick their number of integration steps from a sequence that
# advances a ``random_generator_arg`` carried in the state. The Halton variant
# uses that argument as a sequence index; the random variant folds it into a
# PRNG key.


def halton_next_random_arg_fn(index: Array):
    return jnp.array(index + 1, dtype=jnp.int32)


def random_next_random_arg_fn(random_arg: Array):
    # The state carries an int32 counter (the same kind of argument the Halton
    # sequence uses), but an explicit sequence pair may carry a PRNG key.
    if jnp.issubdtype(jnp.result_type(random_arg), jnp.integer):
        return random_arg + jnp.asarray(1, random_arg.dtype)
    return jax.random.split(random_arg)[1]


def _as_key(random_arg: Array):
    """A PRNG key for ``random_arg``, folding an integer counter if needed."""
    if jnp.issubdtype(jnp.result_type(random_arg), jnp.integer):
        return jax.random.fold_in(jax.random.key(0), random_arg)
    return random_arg


def halton_trajectory_length_fns(length_fn, average):
    """Halton-sequence stepping with a caller-supplied length function."""

    def halton_next_integration_steps_fn(random_arg: Array, **kwargs):
        return length_fn(random_arg, average)

    return (
        halton_next_random_arg_fn,
        halton_next_integration_steps_fn,
    )


def random_trajectory_length_fns(average):
    """Random stepping with uniform lengths in ``[1, 2 * average)``."""

    def random_next_integration_steps_fn(random_arg: Array, **kwargs):
        return jax.random.randint(
            _as_key(random_arg), shape=(), minval=1, maxval=2 * average
        )

    return (
        random_next_random_arg_fn,
        random_next_integration_steps_fn,
    )


def get_dynamic_stepping(integration_steps_sequence, average, *, length_fn):
    """Resolve ``(random_arg_next_fn, integration_steps_fn)`` by sequence kind."""
    if isinstance(integration_steps_sequence, str):
        if integration_steps_sequence == "halton":
            random_arg_next_fn, integration_steps_fn = halton_trajectory_length_fns(
                length_fn, average
            )
        elif integration_steps_sequence == "random":
            random_arg_next_fn, integration_steps_fn = random_trajectory_length_fns(
                average
            )
        else:
            raise ValueError(
                "integration_steps_sequence must be 'halton', 'random', or a tuple"
            )
    else:
        random_arg_next_fn, integration_steps_fn = integration_steps_sequence

    return random_arg_next_fn, integration_steps_fn


def init_dynamic_arg(rng, integration_steps_sequence):
    """Initial random-generator argument for dynamic stepping.

    Both sequences carry an int32 counter: the Halton step uses it as the
    sequence index, and the random step folds it into a key. A PRNG key cannot
    be stored here because it reports an int32 scalar type inside a jitted
    kernel, which makes ``jax.random`` reject it (see ``_as_key``).
    """
    del integration_steps_sequence
    return jax.random.randint(
        rng, shape=(), minval=0, maxval=2**31 - 1, dtype=jnp.int32
    )

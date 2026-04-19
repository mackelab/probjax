from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from probjax.utils.typing import Array, ArrayLike, Callable, PyTree


def _replace_positional_arg(
    args: tuple[Any, ...], index: int, value: Any
) -> tuple[Any, ...]:
    if index < 0 or index >= len(args):
        raise IndexError(
            f"Argument index {index} is out of range for {len(args)} arguments."
        )
    return args[:index] + (value,) + args[index + 1 :]


@dataclass(frozen=True)
class split_drift:
    """Callable marker for drifts of the form c(t) * y + N(t, y)."""

    lin_coeff: Callable[[ArrayLike], Array]
    nonlin: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, x: PyTree, *args, **kwargs) -> PyTree:
        coeff = self.lin_coeff(t)
        linear = jax.tree_util.tree_map(lambda xi: coeff * xi, x)
        nonlin = self.nonlin(t, x, *args, **kwargs)
        return jax.tree_util.tree_map(lambda li, ni: li + ni, linear, nonlin)

    def bind_args(self, *args, **kwargs) -> split_drift:
        if not args and not kwargs:
            return self

        def nonlin_bound(t: ArrayLike, x: PyTree):
            return self.nonlin(t, x, *args, **kwargs)

        return split_drift(lin_coeff=self.lin_coeff, nonlin=nonlin_bound)

    def ravel_arg(
        self, unravel: Callable[[Array], PyTree], index: int = 1
    ) -> split_drift:
        from probjax.utils.jaxutils import ravel_args

        def nonlin_raveled(*args, **kwargs):
            args = _replace_positional_arg(args, index, unravel(args[index]))
            value = self.nonlin(*args, **kwargs)
            value_flat, _ = ravel_args(value)
            return value_flat

        return split_drift(lin_coeff=self.lin_coeff, nonlin=nonlin_raveled)


@dataclass(frozen=True)
class state_drift:
    """Callable marker for autonomous drifts of the form f(y)."""

    drift: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args, **kwargs) -> PyTree:
        del t
        return self.drift(y, *args, **kwargs)

    def bind_args(self, *args, **kwargs) -> state_drift:
        if not args and not kwargs:
            return self

        def drift_bound(y: PyTree):
            return self.drift(y, *args, **kwargs)

        return state_drift(drift=drift_bound)

    def ravel_arg(self, unravel: Callable[[Array], PyTree], index: int = 1) -> Callable:
        from probjax.utils.jaxutils import ravel_args

        def drift_raveled(*args, **kwargs):
            args = _replace_positional_arg(args, index, unravel(args[index]))
            value = self(*args, **kwargs)
            value_flat, _ = ravel_args(value)
            return value_flat

        return drift_raveled


@dataclass(frozen=True)
class affine_drift:
    """Callable marker for affine drifts of the form A(t, y) + b(t)."""

    linear: Callable[..., PyTree]
    bias: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args, **kwargs) -> PyTree:
        linear_part = self.linear(t, y, *args, **kwargs)
        bias_part = self.bias(t, *args, **kwargs)
        return jax.tree_util.tree_map(lambda li, bi: li + bi, linear_part, bias_part)

    def bind_args(self, *args, **kwargs) -> affine_drift:
        if not args and not kwargs:
            return self

        def linear_bound(t: ArrayLike, y: PyTree):
            return self.linear(t, y, *args, **kwargs)

        def bias_bound(t: ArrayLike):
            return self.bias(t, *args, **kwargs)

        return affine_drift(linear=linear_bound, bias=bias_bound)

    def ravel_arg(self, unravel: Callable[[Array], PyTree], index: int = 1) -> Callable:
        from probjax.utils.jaxutils import ravel_args

        def drift_raveled(*args, **kwargs):
            args = _replace_positional_arg(args, index, unravel(args[index]))
            value = self(*args, **kwargs)
            value_flat, _ = ravel_args(value)
            return value_flat

        return drift_raveled


@dataclass(frozen=True)
class additive_diffusion:
    """Callable marker for state-independent diffusions of the form g(t)."""

    diffusion: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args, **kwargs) -> PyTree:
        del y
        return self.diffusion(t, *args, **kwargs)

    def bind_args(self, *args, **kwargs) -> additive_diffusion:
        if not args and not kwargs:
            return self

        def diffusion_bound(t: ArrayLike):
            return self.diffusion(t, *args, **kwargs)

        return additive_diffusion(diffusion=diffusion_bound)


@dataclass(frozen=True)
class linear_drift:
    """Callable marker for linear drift dy/dt = A(t) y + b(t).

    If A is constant (array), the matrix exponential is computed once.
    If A is time-dependent (callable), it's evaluated at each step.
    """

    A: Array | Callable[[ArrayLike], Array]
    b: Callable[..., Array] | None = None

    def __call__(self, t: ArrayLike, y: PyTree, *args, **kwargs) -> PyTree:
        import jax.numpy as jnp

        if callable(self.A):
            A_t = self.A(t)
        else:
            A_t = self.A

        # Handle scalar A matrices (can't use @ operator on scalars)
        A_concrete = jnp.asarray(A_t)
        if A_concrete.ndim == 0:
            linear_part = jax.tree_util.tree_map(lambda yi: A_t * yi, y)
        else:
            linear_part = jax.tree_util.tree_map(lambda yi: A_t @ yi, y)

        if self.b is None:
            return linear_part
        bias_part = self.b(t, *args, **kwargs)
        return jax.tree_util.tree_map(lambda li, bi: li + bi, linear_part, bias_part)

    def bind_args(self, *args, **kwargs) -> linear_drift:
        if self.b is None or (not args and not kwargs):
            return self

        b_fn = self.b

        def b_bound(t: ArrayLike):
            return b_fn(t, *args, **kwargs)

        return linear_drift(A=self.A, b=b_bound)

    def ravel_arg(
        self, unravel: Callable[[Array], PyTree], index: int = 1
    ) -> linear_drift:
        del unravel, index
        return self

    @property
    def is_constant(self) -> bool:
        return not callable(self.A)

    def __hash__(self):
        # Make hashable for use with jit and custom_inverse
        if callable(self.A):
            A_hash = hash(self.A)
        else:
            # Hash the array by converting to bytes
            A_hash = hash(self.A.tobytes())
        b_hash = hash(self.b) if self.b is not None else 0
        return hash((A_hash, b_hash))

    def __eq__(self, other):
        if not isinstance(other, linear_drift):
            return False
        if callable(self.A) != callable(other.A):
            return False
        if callable(self.A):
            A_eq = self.A == other.A
        else:
            import jax.numpy as jnp

            # Both self.A and other.A are Arrays (not callables) at this point
            A_eq = bool(jnp.array_equal(self.A, other.A))  # type: ignore
        b_eq = self.b == other.b
        return A_eq and b_eq


@dataclass(frozen=True)
class const_diffusion:
    """Callable marker for constant diffusion g(y) = G (matrix/vector)."""

    G: Array

    def __call__(self, t: ArrayLike, y: PyTree, *args, **kwargs) -> PyTree:
        del t, args, kwargs
        return self.G

    def bind_args(self, *args, **kwargs) -> const_diffusion:
        del args, kwargs
        return self

    def __hash__(self):
        # Make hashable for use with jit and custom_inverse
        return hash(self.G.tobytes())

    def __eq__(self, other):
        if not isinstance(other, const_diffusion):
            return False
        import jax.numpy as jnp

        return bool(jnp.array_equal(self.G, other.G))

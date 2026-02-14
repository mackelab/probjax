from __future__ import annotations

from dataclasses import dataclass

import jax

from probjax.utils.typing import Array, ArrayLike, Callable, PyTree


@jax.tree_util.register_pytree_node_class
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

    def tree_flatten(self):
        return (), (self.lin_coeff, self.nonlin)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        lin_coeff, nonlin = aux_data
        return cls(lin_coeff=lin_coeff, nonlin=nonlin)


@dataclass(frozen=True)
class state_drift:
    """Callable marker for autonomous drifts of the form f(y)."""

    drift: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args, **kwargs) -> PyTree:
        del t
        return self.drift(y, *args, **kwargs)


@dataclass(frozen=True)
class affine_drift:
    """Callable marker for affine drifts of the form A(t, y) + b(t)."""

    linear: Callable[..., PyTree]
    bias: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args, **kwargs) -> PyTree:
        linear_part = self.linear(t, y, *args, **kwargs)
        bias_part = self.bias(t, *args, **kwargs)
        return jax.tree_util.tree_map(lambda li, bi: li + bi, linear_part, bias_part)


@dataclass(frozen=True)
class additive_diffusion:
    """Callable marker for state-independent diffusions of the form g(t)."""

    diffusion: Callable[..., PyTree]

    def __call__(self, t: ArrayLike, y: PyTree, *args, **kwargs) -> PyTree:
        del y
        return self.diffusion(t, *args, **kwargs)


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

    def __hash__(self):
        # Make hashable for use with jit and custom_inverse
        return hash(self.G.tobytes())

    def __eq__(self, other):
        if not isinstance(other, const_diffusion):
            return False
        import jax.numpy as jnp

        return bool(jnp.array_equal(self.G, other.G))

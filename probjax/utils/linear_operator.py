from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike


# decorator to make a function a linear operator
def linear_operator(
    fn: Callable[[ArrayLike], ArrayLike], in_dim, out_dim
) -> 'LinearOperator':
    return LinearOperator(fn, in_dim, out_dim)


class LinearOperator:
    def __init__(
        self,
        operator: Callable[[ArrayLike], ArrayLike],
        in_dim: int,
        out_dim: int,
        dtype=None,
    ):
        self.operator = operator
        self.in_dim = in_dim
        self.out_dim = out_dim
        if dtype is None:
            # Get default current default dtype i.e. float32 or float64
            dtype = jnp.result_type(jnp.ones(1))
        self._dtype = dtype

    @property
    def shape(self) -> Tuple[int, int]:
        return self.in_dim, self.out_dim

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def ndim(self) -> int:
        return 2

    @property
    def T(self):
        transposed_fn = jax.linear_transpose(self.operator, jnp.ones((self.in_dim,)))

        def adjoint_operator(x: ArrayLike) -> ArrayLike:
            return transposed_fn(x)[0]

        return LinearOperator(adjoint_operator, self.out_dim, self.in_dim, self.dtype)

    def __add__(self, other: 'LinearOperator' | ArrayLike) -> 'LinearOperator':
        if isinstance(other, LinearOperator):
            assert self.in_dim == other.in_dim, f"{self.in_dim} != {other.in_dim}"
            assert self.out_dim == other.out_dim, f"{self.out_dim} != {other.out_dim}"

            def sum_fn(x: ArrayLike) -> ArrayLike:
                return self.operator(x) + other.operator(x)

            return LinearOperator(sum_fn, self.in_dim, self.out_dim, self.dtype)
        else:
            other = jnp.asarray(other)
            assert other.ndim == 2, f"Can only add 2D arrays, got {other.ndim}D"

            def sum_fn(x: ArrayLike) -> ArrayLike:
                return self.operator(x) + other @ x

            return LinearOperator(sum_fn, self.in_dim, self.out_dim, self.dtype)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self) -> 'LinearOperator':
        def neg_fn(x: ArrayLike) -> ArrayLike:
            return -self.operator(x)

        return LinearOperator(neg_fn, self.in_dim, self.out_dim, self.dtype)

    def __sub__(self, other: 'LinearOperator') -> 'LinearOperator':
        return self + (-other)

    def __mul__(self, other: ArrayLike) -> 'LinearOperator':
        other = jnp.asarray(other)

        def scalar_mul_fn(x: ArrayLike) -> ArrayLike:
            return other * self.operator(x)

        return LinearOperator(scalar_mul_fn, self.in_dim, self.out_dim, self.dtype)

    def __matmul__(self, other: 'LinearOperator' | ArrayLike) -> 'LinearOperator':
        if isinstance(other, LinearOperator):
            assert self.in_dim == other.out_dim, f"{self.out_dim} != {other.in_dim}"

            def matmul_fn(x: ArrayLike) -> ArrayLike:
                return self.operator(other.operator(x))

            return LinearOperator(matmul_fn, other.in_dim, self.out_dim, self.dtype)
        elif isinstance(other, ArrayLike):
            other = jnp.asarray(other)
            if other.ndim == 1:
                return self.operator(other)
            else:

                def matmul_fn(x: ArrayLike) -> ArrayLike:
                    return other @ self.operator(x)

                return LinearOperator(matmul_fn, self.in_dim, other.shape[-2])
        else:
            raise NotImplementedError(
                f"Multiplication with {type(other)} not implemented"
            )

    def __rmatmul__(self, other: 'LinearOperator' | ArrayLike) -> 'LinearOperator':
        if isinstance(other, LinearOperator):
            return other @ self
        else:
            other = jnp.asarray(other)
            if other.ndim == 1:
                return self.operator(other)
            else:

                def composed(x: ArrayLike) -> ArrayLike:
                    return other @ self.operator(x)

                return LinearOperator(composed, self.in_dim, other.shape[-2])

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.operator(x)

    def as_array(self) -> ArrayLike:
        """Materialize this LinearOperator as a dense array."""
        return LinearOperator.to_array(self.operator, self.in_dim, self._dtype)

    @staticmethod
    def to_array(
        operator: Callable[[ArrayLike], ArrayLike], in_dim: int, dtype=None
    ) -> ArrayLike:
        if dtype is None:
            dtype = jnp.result_type(jnp.ones(1))
        matrix = jax.vmap(operator)(jnp.eye(in_dim, dtype=dtype))
        return matrix.T

    @staticmethod
    def from_array(matrix: ArrayLike) -> 'LinearOperator':
        matrix = jnp.atleast_2d(matrix)

        def operator(x: ArrayLike) -> ArrayLike:
            return jnp.dot(matrix, x)

        return LinearOperator(operator, matrix.shape[1], matrix.shape[0])

    # Pytree registration
    def tree_flatten(self):
        return (), (self.operator, self.in_dim, self.out_dim, self._dtype)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        op_fn, in_dim, out_dim, dtype = aux_data
        return cls(op_fn, in_dim, out_dim, dtype)


tree_util.register_pytree_node(
    LinearOperator, LinearOperator.tree_flatten, LinearOperator.tree_unflatten
)

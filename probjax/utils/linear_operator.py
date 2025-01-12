from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
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
        elif isinstance(other, ArrayLike) and other.ndim == 2:
            other = jnp.asarray(other)
            operator = LinearOperator.from_array(other)
            return self + operator
        else:
            raise NotImplementedError(f"Addition with {type(other)} not implemented")

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
                operator = LinearOperator.from_array(other)
                return self @ operator
        else:
            raise NotImplementedError(
                f"Multiplication with {type(other)} not implemented"
            )

    def __rmatmul__(self, other: 'LinearOperator' | ArrayLike) -> 'LinearOperator':
        if isinstance(other, LinearOperator):
            return other @ self
        elif isinstance(other, ArrayLike):
            other = jnp.asarray(other)
            if other.ndim == 1:
                return self.operator(other)
            else:
                operator = LinearOperator.from_array(other)
                return operator @ self
        else:
            raise NotImplementedError(
                f"Multiplication with {type(other)} not implemented"
            )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.operator(x)

    @jax.util.cache()
    def __jax_array__(self) -> ArrayLike:
        with jax.ensure_compile_time_eval():
            matrix = LinearOperator.to_array(self.operator, self.in_dim, self.dtype)
        return matrix

    def as_array(self) -> ArrayLike:
        return self.__jax_array__()

    @staticmethod
    def to_array(
        operator: Callable[[ArrayLike], ArrayLike], in_dim: int, dtype=jnp.float32
    ) -> ArrayLike:
        matrix = jax.vmap(operator)(jnp.eye(in_dim, dtype=dtype))
        return matrix.T

    @staticmethod
    def from_array(matrix: ArrayLike) -> 'LinearOperator':
        matrix = jnp.atleast_2d(matrix)

        def operator(x: ArrayLike) -> ArrayLike:
            return jnp.dot(matrix, x)

        return LinearOperator(operator, matrix.shape[1], matrix.shape[0])

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike

from probjax.utils.linalg import batched_pcg_solve, lanczos_logdet


def _global_default_dtype():
    """Current global default floating dtype (float32, or float64 under x64)."""
    return jnp.result_type(jnp.ones(1))


class LinearOperator:
    """Matrix-free linear operator ``x -> A @ x`` with dense-oracle algebra.

    Instances are treated as immutable; arithmetic dunders return new
    operators. ``dtype`` is the dtype the operator's output is materialized
    and traced with — inferred by probing the callable once at construction
    when not given explicitly.
    """

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
            dtype = self._infer_dtype(operator, in_dim)
        self._dtype = dtype

    @staticmethod
    def _infer_dtype(
        operator: Callable[[ArrayLike], ArrayLike], in_dim: int
    ) -> Any:
        """Infer the output dtype by probing the operator once on a zero vector.

        If probing is unsafe for the callable (it raises on a zero input),
        fall back to the global default dtype. The attribute is then
        *assumed*, not inferred.
        """
        try:
            probe = operator(jnp.zeros((in_dim,), dtype=_global_default_dtype()))
            return jnp.result_type(probe)
        except Exception:
            return _global_default_dtype()

    @property
    def shape(self) -> Tuple[int, int]:
        return self.out_dim, self.in_dim

    @property
    def dtype(self) -> Any:
        return self._dtype

    @property
    def ndim(self) -> int:
        return 2

    @cached_property
    def T(self) -> 'LinearOperator':
        """Adjoint operator, via ``jax.linear_transpose`` (cached per instance)."""
        transposed_fn = jax.linear_transpose(
            self.operator, jnp.ones((self.in_dim,), dtype=self.dtype)
        )

        def adjoint_operator(x: ArrayLike) -> ArrayLike:
            return transposed_fn(x)[0]

        return LinearOperator(adjoint_operator, self.out_dim, self.in_dim, self.dtype)

    def __add__(self, other: 'LinearOperator' | ArrayLike) -> 'LinearOperator':
        if isinstance(other, LinearOperator):
            if self.in_dim != other.in_dim or self.out_dim != other.out_dim:
                raise ValueError(
                    f"Incompatible shapes for addition: {self.shape} and {other.shape}"
                )
            other_dtype = other.dtype

            def sum_fn(x: ArrayLike) -> ArrayLike:
                return self.operator(x) + other.operator(x)

        else:
            other = jnp.asarray(other)
            if other.ndim == 0:
                # A 0-d operand acts as c * I.
                other_dtype = other.dtype

                def sum_fn(x: ArrayLike) -> ArrayLike:
                    return self.operator(x) + other * x

            elif other.ndim == 2:
                if other.shape != (self.out_dim, self.in_dim):
                    raise ValueError(
                        "Can only add arrays of shape "
                        f"{(self.out_dim, self.in_dim)}, got {other.shape}"
                    )
                other_dtype = other.dtype

                def sum_fn(x: ArrayLike) -> ArrayLike:
                    return self.operator(x) + other @ x

            else:
                raise ValueError(
                    "Can only add 0-d or (out_dim, in_dim) arrays, "
                    f"got shape {other.shape}"
                )

        return LinearOperator(
            sum_fn,
            self.in_dim,
            self.out_dim,
            jnp.result_type(self.dtype, other_dtype),
        )

    def __radd__(self, other):
        # Note: this only ever fires for Python scalars (e.g. ``1.0 + A``).
        # For a jnp.ndarray left operand, JAX's own __add__ raises before
        # Python gets a chance to dispatch to __radd__, so ``arr + A`` stays
        # unsupported — use ``A + arr`` instead.
        return self.__add__(other)

    def __neg__(self) -> 'LinearOperator':
        def neg_fn(x: ArrayLike) -> ArrayLike:
            return -self.operator(x)

        return LinearOperator(neg_fn, self.in_dim, self.out_dim, self.dtype)

    def __sub__(self, other: 'LinearOperator' | ArrayLike) -> 'LinearOperator':
        neg = -other if isinstance(other, LinearOperator) else -jnp.asarray(other)
        return self + neg

    def __rsub__(self, other):
        # Note: same dispatch caveat as __radd__ — only fires for Python
        # scalars; ``arr - A`` with a jnp.ndarray raises inside JAX first.
        if isinstance(other, LinearOperator):
            return other + (-self)
        other = jnp.asarray(other)
        if other.ndim == 0:
            # A 0-d operand acts as c * I.
            def rsub_fn(x: ArrayLike) -> ArrayLike:
                return other * x - self.operator(x)

        elif other.ndim == 2:
            if other.shape != (self.out_dim, self.in_dim):
                raise ValueError(
                    "Can only subtract arrays of shape "
                    f"{(self.out_dim, self.in_dim)}, got {other.shape}"
                )

            def rsub_fn(x: ArrayLike) -> ArrayLike:
                return other @ x - self.operator(x)

        else:
            raise ValueError(
                "Can only subtract 0-d or (out_dim, in_dim) arrays, "
                f"got shape {other.shape}"
            )

        return LinearOperator(
            rsub_fn,
            self.in_dim,
            self.out_dim,
            jnp.result_type(self.dtype, other.dtype),
        )

    def __mul__(self, other: ArrayLike) -> 'LinearOperator':
        other = jnp.asarray(other)
        if other.ndim != 0 and other.shape != (self.out_dim,):
            raise ValueError(
                "Can only multiply by a scalar or an array of shape "
                f"{(self.out_dim,)}, got shape {other.shape}"
            )

        def scaled_fn(x: ArrayLike) -> ArrayLike:
            return other * self.operator(x)

        return LinearOperator(
            scaled_fn,
            self.in_dim,
            self.out_dim,
            jnp.result_type(self.dtype, other.dtype),
        )

    def __rmul__(self, other: ArrayLike) -> 'LinearOperator':
        # Note: same dispatch caveat as __radd__ — only fires for Python
        # scalars; ``arr * A`` with a jnp.ndarray raises inside JAX first.
        return self.__mul__(other)

    def __truediv__(self, other: ArrayLike) -> 'LinearOperator':
        other = jnp.asarray(other)
        if other.ndim != 0 and other.shape != (self.out_dim,):
            raise ValueError(
                "Can only divide by a scalar or an array of shape "
                f"{(self.out_dim,)}, got shape {other.shape}"
            )

        def div_fn(x: ArrayLike) -> ArrayLike:
            return self.operator(x) / other

        # True division promotes integer inputs to float, which
        # jnp.result_type alone would not capture; probe the composed
        # function so the dtype attribute stays truthful.
        out_dtype = jnp.result_type(
            div_fn(jnp.zeros((self.in_dim,), dtype=self.dtype))
        )
        return LinearOperator(div_fn, self.in_dim, self.out_dim, out_dtype)

    def __matmul__(self, other: 'LinearOperator' | ArrayLike):
        if isinstance(other, LinearOperator):
            if self.in_dim != other.out_dim:
                raise ValueError(
                    f"Incompatible matrix shapes: {self.shape} and {other.shape}"
                )

            def composed(x):
                return self.operator(other.operator(x))

            return LinearOperator(
                composed,
                other.in_dim,
                self.out_dim,
                jnp.result_type(self.dtype, other.dtype),
            )

        # ArrayLike is a typing alias, not a runtime test for JAX tracers.
        other = jnp.asarray(other)
        if other.ndim not in (1, 2) or other.shape[0] != self.in_dim:
            raise ValueError(
                f"Incompatible matrix shapes: {self.shape} and {other.shape}"
            )
        if other.ndim == 1:
            return self.operator(other)

        def composed(x):
            return self.operator(other @ x)

        return LinearOperator(
            composed,
            other.shape[1],
            self.out_dim,
            jnp.result_type(self.dtype, other.dtype),
        )

    def __rmatmul__(self, other: 'LinearOperator' | ArrayLike):
        if isinstance(other, LinearOperator):
            return other @ self
        other = jnp.asarray(other)
        if other.ndim not in (1, 2) or other.shape[-1] != self.out_dim:
            raise ValueError(
                f"Incompatible matrix shapes: {other.shape} and {self.shape}"
            )
        if other.ndim == 1:
            return self.T.operator(other)

        def composed(x):
            return other @ self.operator(x)

        return LinearOperator(
            composed,
            self.in_dim,
            other.shape[0],
            jnp.result_type(self.dtype, other.dtype),
        )

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
            dtype = _global_default_dtype()
        matrix = jax.vmap(operator)(jnp.eye(in_dim, dtype=dtype))
        return matrix.T

    @staticmethod
    def from_array(matrix: ArrayLike) -> 'LinearOperator':
        matrix = jnp.atleast_2d(matrix)

        def operator(x: ArrayLike) -> ArrayLike:
            return jnp.dot(matrix, x)

        return LinearOperator(operator, matrix.shape[1], matrix.shape[0], matrix.dtype)

    def solve(
        self,
        rhs: ArrayLike,
        *,
        method: str = "auto",
        dense_mem_limit: float = 200,
        assume_a: str | None = None,
    ) -> ArrayLike:
        """Solve ``self @ x = rhs`` for ``x``.

        Dispatches between a dense factorization and matrix-free (batched)
        conjugate gradients based on the memory needed to materialize the
        operator, mirroring the Kalman filter's historical dispatch rule.
        The iterative path assumes a symmetric positive-definite operator.

        Args:
            rhs: Right-hand side, shape ``(out_dim,)`` or ``(out_dim, nrhs)``.
            method: ``"auto"`` (choose by memory), ``"dense"`` or
                ``"iterative"``.
            dense_mem_limit: Max MB for the materialized operator before
                ``"auto"`` switches to the iterative path.
            assume_a: Passed to ``jax.scipy.linalg.solve`` on the dense path
                (e.g. ``"pos"`` for SPD systems).

        Returns:
            Solution with the same shape as ``rhs``.
        """
        if method not in ("auto", "dense", "iterative"):
            raise ValueError(
                f"Unknown solve method {method!r}; expected 'auto', 'dense' or 'iterative'"
            )
        if self.in_dim != self.out_dim:
            raise ValueError(
                f"Can only solve square operators, got shape {self.shape}"
            )
        rhs = jnp.asarray(rhs)
        if rhs.ndim == 1 or rhs.ndim == 2:
            if rhs.shape[0] != self.out_dim:
                raise ValueError(
                    f"Incompatible rhs shape {rhs.shape} for operator of shape {self.shape}"
                )
        else:
            raise ValueError(
                f"rhs must be a vector or a matrix, got shape {rhs.shape}"
            )

        dim = self.out_dim
        # Estimate materialized memory in MB (f64 = 8 bytes as upper bound).
        mem_mb = dim * dim * 8 / (1024 * 1024)
        iterative = method == "iterative" or (
            method == "auto" and mem_mb > dense_mem_limit
        )
        if iterative:
            if rhs.ndim == 1:
                return jax.scipy.sparse.linalg.cg(self.operator, rhs, tol=1e-4)[0]
            solution, _info = batched_pcg_solve(
                self.operator, rhs, tol=1e-4, block_size=128
            )
            return solution
        dense = self.as_array()
        if assume_a is None:
            return jnp.linalg.solve(dense, rhs)
        return jax.scipy.linalg.solve(dense, rhs, assume_a=assume_a)

    def logdet(
        self,
        *,
        method: str = "auto",
        dense_mem_limit: float = 200,
        num_steps: int = 500,
    ) -> ArrayLike:
        """Compute ``log det(self)`` for a symmetric positive-definite operator.

        Uses the same memory-based dispatch as :meth:`solve`: materialize and
        call ``slogdet`` when cheap, otherwise estimate with multi-probe
        Lanczos quadrature (:func:`probjax.utils.linalg.lanczos_logdet`).

        Args:
            method: ``"auto"`` (choose by memory), ``"dense"`` or
                ``"iterative"``.
            dense_mem_limit: Max MB for the materialized operator before
                ``"auto"`` switches to the iterative path.
            num_steps: Lanczos quadrature depth on the iterative path
                (capped at the operator dimension).

        Returns:
            Scalar log determinant.
        """
        if method not in ("auto", "dense", "iterative"):
            raise ValueError(
                f"Unknown logdet method {method!r}; expected 'auto', 'dense' or 'iterative'"
            )
        if self.in_dim != self.out_dim:
            raise ValueError(
                f"Can only compute the logdet of square operators, got shape {self.shape}"
            )
        dim = self.out_dim
        mem_mb = dim * dim * 8 / (1024 * 1024)
        if method == "iterative" or (method == "auto" and mem_mb > dense_mem_limit):
            return lanczos_logdet(self, num_steps=min(num_steps, dim))
        return jnp.linalg.slogdet(self.as_array()).logabsdet

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

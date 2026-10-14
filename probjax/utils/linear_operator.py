"""Structured linear operators with pluggable solvers.

Three structures cover the library's needs: :class:`Diagonal` (including a
scalar ``c * I``), :class:`Dense` and matrix-free :class:`Function` (plus
:class:`Jacobian` for matrix-free derivatives). The
structure, not the problem size, decides how ``solve``/``logdet``/``expm`` are
computed, and cheap structure is preserved under ``+`` and ``@``.

Operators map arrays of ``in_shape`` to arrays of ``out_shape``. ``shape`` and
:meth:`LinearOperator.as_array` use the flattened ``(out_size, in_size)``
matrix view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probjax.utils.linalg import lanczos_logdet
from probjax.utils.optional import require_lineax
from probjax.utils.special.phi1 import phi1_scalar

Shape = tuple[int, ...]

_ITERATIVE_SOLVERS = {
    "cg": jax.scipy.sparse.linalg.cg,
    "gmres": jax.scipy.sparse.linalg.gmres,
    "bicgstab": jax.scipy.sparse.linalg.bicgstab,
}


def _static(**kwargs):
    return field(metadata=dict(static=True), **kwargs)


def _as_shape(shape) -> Shape:
    from probjax.utils.guards import ensure_tuple

    return ensure_tuple("shape", shape)


def is_operator(x: Any) -> bool:
    """True if ``x`` is a structured :class:`LinearOperator`."""
    return isinstance(x, LinearOperator)


def as_array(x: Any) -> jax.Array:
    """Materialize operators via ``as_array()``; arrays via ``jnp.asarray``."""
    if isinstance(x, LinearOperator):
        return x.as_array()
    return jnp.asarray(x)


def densify_if_small(S: Any, dense_mem_limit: float = 200) -> Any:
    """Materialize ``S`` when its dense matrix fits in ``dense_mem_limit`` MB."""
    if not isinstance(S, LinearOperator):
        return jnp.asarray(S)
    mem_mb = S.shape[0] * S.shape[1] * jnp.dtype(S.dtype).itemsize / 1024**2
    return S.as_dense() if mem_mb <= dense_mem_limit else S


def is_array_or_operator(x: Any) -> bool:
    """True for dense arrays or :class:`LinearOperator` values."""
    from probjax.utils.guards import is_array_like

    return isinstance(x, LinearOperator) or is_array_like(x)


class LinearOperator:
    """Base class: shared shape logic, arithmetic and solver entry points.

    Subclasses provide ``in_shape``, ``out_shape``, ``dtype``, ``_mv`` (apply
    to a single input of exactly ``in_shape``) and ``T``.
    """

    @property
    def in_size(self) -> int:
        return math.prod(self.in_shape)

    @property
    def out_size(self) -> int:
        return math.prod(self.out_shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_size, self.in_size

    def mv(self, x: ArrayLike) -> jax.Array:
        """Apply to ``x`` of ``in_shape``, or ``in_shape + (k,)`` for k inputs."""
        x = jnp.asarray(x)
        if x.shape == self.in_shape:
            return self._mv(x)
        if x.shape[:-1] == self.in_shape:
            return jax.vmap(self._mv, in_axes=-1, out_axes=-1)(x)
        raise ValueError(f"Expected input of shape {self.in_shape}, got {x.shape}")

    __call__ = mv

    def as_array(self) -> jax.Array:
        """Materialize the flattened ``(out_size, in_size)`` matrix."""
        basis = jnp.eye(self.in_size, dtype=self.dtype)
        return jax.vmap(self._flat_mv)(basis).T

    def as_dense(self) -> "Dense":
        return Dense(self.as_array(), self.in_shape, self.out_shape)

    def _flat_mv(self, v: jax.Array) -> jax.Array:
        return self._mv(v.reshape(self.in_shape)).ravel()

    # Arithmetic -------------------------------------------------------------

    def _coerce(self, other) -> "LinearOperator":
        """Scalars act as ``c * I``; 2-D arrays as the flattened matrix."""
        if isinstance(other, LinearOperator):
            return other
        other = jnp.asarray(other)
        if other.ndim == 0:
            return Diagonal(other, self.out_shape)
        if other.shape == self.shape:
            return Dense(other, self.in_shape, self.out_shape)
        raise ValueError(
            f"Cannot combine shape {other.shape} with operator {self.shape}"
        )

    def __add__(self, other) -> "LinearOperator":
        other = self._coerce(other)
        if (other.in_shape, other.out_shape) != (self.in_shape, self.out_shape):
            raise ValueError(
                f"Incompatible operators for addition: {self.in_shape} -> "
                f"{self.out_shape} and {other.in_shape} -> {other.out_shape}"
            )
        return _add(self, other)

    __radd__ = __add__

    def __neg__(self) -> "LinearOperator":
        return self * -1

    def __sub__(self, other) -> "LinearOperator":
        return self + (-self._coerce(other))

    def __rsub__(self, other) -> "LinearOperator":
        return (-self) + other

    def __mul__(self, scalar) -> "LinearOperator":
        scalar = jnp.asarray(scalar)
        if scalar.ndim != 0:
            raise ValueError(
                "Operators can only be scaled by scalars; use Diagonal @ A"
            )
        return Diagonal(scalar, self.out_shape) @ self

    __rmul__ = __mul__

    def __truediv__(self, scalar) -> "LinearOperator":
        return self * (1 / jnp.asarray(scalar))

    def __matmul__(self, other):
        """``A @ B`` composes; ``A @ x`` with ``x`` of ``in_shape`` applies."""
        if isinstance(other, LinearOperator):
            return _compose(self, other)
        other = jnp.asarray(other)
        if other.shape == self.in_shape:
            return self._mv(other)
        # A 2-D array right of a vector-input operator is a matrix to compose.
        if len(self.in_shape) == 1 and other.ndim == 2:
            return _compose(self, Dense(other))
        raise ValueError(
            f"Incompatible shapes: operator {self.shape} and {other.shape}"
        )

    def __rmatmul__(self, other):
        other = jnp.asarray(other)
        if other.shape == self.out_shape:
            return self.T._mv(other)
        if len(self.out_shape) == 1 and other.ndim == 2:
            return _compose(Dense(other), self)
        raise ValueError(
            f"Incompatible shapes: {other.shape} and operator {self.shape}"
        )

    # Linear algebra ---------------------------------------------------------

    def solve(self, b: ArrayLike, solver: Any = "auto", **kwargs) -> jax.Array:
        return solve(self, b, solver, **kwargs)

    def logdet(self, **kwargs) -> jax.Array:
        return logdet(self, **kwargs)

    def expm(self, t: ArrayLike = 1.0) -> "LinearOperator":
        """``exp(t * A)``. Matrix-free operators must call ``as_dense()`` first."""
        raise NotImplementedError(
            f"{type(self).__name__} has no expm; call as_dense() first"
        )

    def phi(self, t: ArrayLike = 1.0, k: int = 1) -> list["LinearOperator"]:
        """``[phi_0(tA), ..., phi_k(tA)]`` with ``phi_j(z) = sum_m z^m / (m+j)!``."""
        raise NotImplementedError(
            f"{type(self).__name__} has no phi; call as_dense() first"
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Diagonal(LinearOperator):
    """Elementwise ``x -> diag * x``; a 0-d ``diag`` is ``diag * I``."""

    diag: jax.Array
    in_shape: Shape = _static(default=None)

    def __post_init__(self):
        in_shape = jnp.shape(self.diag) if self.in_shape is None else self.in_shape
        object.__setattr__(self, "in_shape", _as_shape(in_shape))

    @property
    def out_shape(self) -> Shape:
        return self.in_shape

    @property
    def dtype(self):
        return jnp.result_type(self.diag)

    @property
    def T(self) -> "Diagonal":
        return self

    def _mv(self, x):
        return self.diag * x

    def _full_diag(self) -> jax.Array:
        return jnp.broadcast_to(self.diag, self.in_shape).ravel()

    def as_array(self):
        return jnp.diag(self._full_diag())

    def expm(self, t=1.0):
        return Diagonal(jnp.exp(t * self.diag), self.in_shape)

    def phi(self, t=1.0, k=1):
        z = t * jnp.asarray(self.diag)
        if k == 1:
            values = [jnp.exp(z), phi1_scalar(z)]
        else:
            # Higher phi_k by a tiny block expm per entry; the scalar
            # recursion phi_{k+1} = (phi_k - 1/k!) / z cancels badly near 0.
            per_entry = jax.vmap(
                lambda zi: jnp.stack(phi_matrices(zi.reshape(1, 1), k)).ravel()
            )
            values = [v.reshape(z.shape) for v in per_entry(z.ravel()).T]
        return [Diagonal(v, self.in_shape) for v in values]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Dense(LinearOperator):
    """A materialized ``(out_size, in_size)`` matrix acting on shaped arrays."""

    matrix: jax.Array
    in_shape: Shape = _static(default=None)
    out_shape: Shape = _static(default=None)

    def __post_init__(self):
        rows, cols = jnp.shape(self.matrix)
        in_shape = (cols,) if self.in_shape is None else self.in_shape
        out_shape = (rows,) if self.out_shape is None else self.out_shape
        object.__setattr__(self, "in_shape", _as_shape(in_shape))
        object.__setattr__(self, "out_shape", _as_shape(out_shape))

    @classmethod
    def jacfwd(cls, fn: Callable, x: ArrayLike) -> "Dense":
        """Materialized Jacobian of ``fn`` at ``x``."""
        x = jnp.asarray(x)
        jac = jax.jacfwd(fn)(x)
        out_shape = jac.shape[: jac.ndim - x.ndim]
        return cls(jac.reshape(math.prod(out_shape), x.size), x.shape, out_shape)

    @property
    def dtype(self):
        return self.matrix.dtype

    @property
    def T(self) -> "Dense":
        return Dense(self.matrix.T, self.out_shape, self.in_shape)

    def _mv(self, x):
        return (self.matrix @ x.ravel()).reshape(self.out_shape)

    def as_array(self):
        return self.matrix

    def expm(self, t=1.0):
        return Dense(
            jax.scipy.linalg.expm(t * self.matrix), self.in_shape, self.in_shape
        )

    def phi(self, t=1.0, k=1):
        return [
            Dense(m, self.in_shape, self.in_shape)
            for m in phi_matrices(t * self.matrix, k)
        ]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Function(LinearOperator):
    """Matrix-free operator given by a linear ``matvec`` on ``in_shape`` arrays.

    ``out_shape``/``dtype`` are inferred with ``jax.eval_shape`` (no FLOPs)
    when omitted. ``matvec`` is static pytree data, so build Function
    operators inside the traced function that uses them rather than passing
    them across ``jit``/``scan`` boundaries.
    """

    matvec: Callable = _static()
    in_shape: Shape = _static()
    out_shape: Shape = _static(default=None)
    dtype: Any = _static(default=None)

    def __post_init__(self):
        in_shape = _as_shape(self.in_shape)
        object.__setattr__(self, "in_shape", in_shape)
        if self.out_shape is not None and self.dtype is not None:
            object.__setattr__(self, "out_shape", _as_shape(self.out_shape))
            return
        probe_dtype = jnp.result_type(float) if self.dtype is None else self.dtype
        out = jax.eval_shape(self.matvec, jax.ShapeDtypeStruct(in_shape, probe_dtype))
        if self.out_shape is not None and _as_shape(self.out_shape) != out.shape:
            raise ValueError(
                f"matvec returns shape {out.shape}, expected {_as_shape(self.out_shape)}"
            )
        object.__setattr__(self, "out_shape", out.shape)
        if self.dtype is None:
            object.__setattr__(self, "dtype", out.dtype)

    @property
    def T(self) -> "LinearOperator":
        return _Transpose(self)

    def _mv(self, x):
        return self.matvec(x)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Jacobian(LinearOperator):
    """Matrix-free Jacobian of ``fn(x, *args)`` with respect to ``x``.

    Each application costs one JVP. ``x`` and ``args`` are pytree leaves and
    ``fn`` is static, so rebuilding the Jacobian at a new point inside a
    jitted function does not retrace.
    """

    fn: Callable = _static()
    x: jax.Array
    args: tuple = ()
    out_shape: Shape = _static(default=None)
    dtype: Any = _static(default=None)

    def __post_init__(self):
        if self.out_shape is None or self.dtype is None:
            out = jax.eval_shape(self.fn, self.x, *self.args)
            object.__setattr__(self, "out_shape", out.shape)
            object.__setattr__(self, "dtype", out.dtype)

    @property
    def in_shape(self) -> Shape:
        return jnp.shape(self.x)

    @property
    def T(self) -> "LinearOperator":
        return _Transpose(self)

    def _mv(self, v):
        return jax.jvp(lambda x: self.fn(x, *self.args), (self.x,), (v,))[1]


# Composite operators are pytree nodes over their parts, never closures: a new
# closure per composition would be fresh static data and retrace every jit.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Sum(LinearOperator):
    a: LinearOperator
    b: LinearOperator

    @property
    def in_shape(self) -> Shape:
        return self.a.in_shape

    @property
    def out_shape(self) -> Shape:
        return self.a.out_shape

    @property
    def dtype(self):
        return jnp.result_type(self.a.dtype, self.b.dtype)

    @property
    def T(self) -> "LinearOperator":
        return _Sum(self.a.T, self.b.T)

    def _mv(self, x):
        return self.a._mv(x) + self.b._mv(x)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Product(LinearOperator):
    """``a @ b``."""

    a: LinearOperator
    b: LinearOperator

    @property
    def in_shape(self) -> Shape:
        return self.b.in_shape

    @property
    def out_shape(self) -> Shape:
        return self.a.out_shape

    @property
    def dtype(self):
        return jnp.result_type(self.a.dtype, self.b.dtype)

    @property
    def T(self) -> "LinearOperator":
        return _Product(self.b.T, self.a.T)

    def _mv(self, x):
        return self.a._mv(self.b._mv(x))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Transpose(LinearOperator):
    op: LinearOperator

    @property
    def in_shape(self) -> Shape:
        return self.op.out_shape

    @property
    def out_shape(self) -> Shape:
        return self.op.in_shape

    @property
    def dtype(self):
        return self.op.dtype

    @property
    def T(self) -> "LinearOperator":
        return self.op

    def _mv(self, y):
        primal = jax.ShapeDtypeStruct(self.op.in_shape, self.op.dtype)
        return jax.linear_transpose(self.op._mv, primal)(y)[0]


def as_operator(A: LinearOperator | ArrayLike, shape: Shape) -> LinearOperator:
    """Interpret ``A`` as a square operator on arrays of ``shape``.

    Operators pass through (their domain must equal ``shape``); a 0-d array
    or 1-d array broadcastable to ``shape`` is a diagonal, and a square 2-d
    array a dense matrix on vector states of matching dimension. Anything
    else raises a clear error: the exact solvers need a flattened square
    system, so dense drifts support vector states only (evaluate the drift
    itself for matmul-broadcast behavior on shaped states).
    """
    shape = _as_shape(shape)
    if isinstance(A, LinearOperator):
        if tuple(A.in_shape) != tuple(shape) or tuple(A.out_shape) != tuple(shape):
            raise ValueError(
                f"Operator with domain {A.in_shape} -> {A.out_shape} cannot act"
                f" on states of shape {shape}."
            )
        return A
    A = jnp.asarray(A)
    if A.ndim <= 1:
        try:
            jnp.broadcast_shapes(A.shape, shape)
        except ValueError:
            raise ValueError(
                f"Diagonal of shape {A.shape} cannot broadcast to state"
                f" shape {shape}."
            ) from None
        return Diagonal(A, shape)
    if A.ndim == 2:
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Dense drift must be square, got shape {A.shape}.")
        if tuple(shape) != (A.shape[0],):
            raise ValueError(
                f"Dense drift of shape {A.shape} needs vector states of shape"
                f" ({A.shape[0]},), got {shape}."
            )
        return Dense(A, shape, shape)
    raise ValueError(f"Expected a 0-d, 1-d or 2-d array, got shape {A.shape}")


def phi_matrices(z: jax.Array, k: int) -> list[jax.Array]:
    """``[phi_0(Z), ..., phi_k(Z)]`` from one block matrix exponential.

    expm of the ``(k+1)``-block-bidiagonal augmentation (``Z`` in the first
    block, identities on the first superdiagonal) has ``phi_j(Z)`` in its
    j-th top-row block.
    """
    n = z.shape[0]
    aug = jnp.zeros((n * (k + 1), n * (k + 1)), dtype=z.dtype)
    aug = aug.at[:n, :n].set(z)
    aug = aug.at[: n * k, n:].add(jnp.eye(n * k, dtype=z.dtype))
    top = jax.scipy.linalg.expm(aug)[:n]
    return [top[:, j * n : (j + 1) * n] for j in range(k + 1)]


def _add(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    if isinstance(a, Diagonal) and isinstance(b, Diagonal):
        return Diagonal(a.diag + b.diag, a.in_shape)
    if isinstance(a, (Dense, Diagonal)) and isinstance(b, (Dense, Diagonal)):
        return Dense(a.as_array() + b.as_array(), a.in_shape, a.out_shape)
    return _Sum(a, b)


def _compose(a: LinearOperator, b: LinearOperator) -> LinearOperator:
    """``a @ b``, keeping Diagonal/Dense structure where it is cheap."""
    if a.in_shape != b.out_shape:
        raise ValueError(
            f"Incompatible operators for composition: {a.in_shape} != {b.out_shape}"
        )
    if isinstance(a, Diagonal) and isinstance(b, Diagonal):
        return Diagonal(a.diag * b.diag, b.in_shape)
    if isinstance(a, Diagonal) and isinstance(b, Dense):
        return Dense(a._full_diag()[:, None] * b.matrix, b.in_shape, a.out_shape)
    if isinstance(a, Dense) and isinstance(b, Diagonal):
        return Dense(a.matrix * b._full_diag()[None, :], b.in_shape, a.out_shape)
    if isinstance(a, Dense) and isinstance(b, Dense):
        return Dense(a.matrix @ b.matrix, b.in_shape, a.out_shape)
    return _Product(a, b)


def _symmetric_part(matrix: jax.Array) -> jax.Array:
    # Cholesky reads one triangle only; factoring the symmetric part keeps the
    # gradient symmetric instead of concentrating it in that triangle.
    return (matrix + matrix.T) / 2


def _check_square(op: LinearOperator):
    if op.in_shape != op.out_shape:
        raise ValueError(
            f"Expected a square operator, got {op.in_shape} -> {op.out_shape}"
        )


def solve(
    op: LinearOperator,
    b: ArrayLike,
    solver: Any = "auto",
    *,
    psd: bool = False,
    tol: float = 1e-5,
    maxiter: int | None = None,
) -> jax.Array:
    """Solve ``op @ x = b`` for ``x``.

    Args:
        op: Square operator (``in_shape == out_shape``).
        b: Right-hand side of ``out_shape``, or ``out_shape + (k,)`` for k
            right-hand sides.
        solver: ``"auto"``, ``"lu"``, ``"cholesky"``, ``"cg"``, ``"gmres"``,
            ``"bicgstab"`` or any lineax solver instance. ``"auto"`` picks by
            structure: elementwise for Diagonal, Cholesky (``psd``) or LU for
            Dense, CG (``psd``) or GMRES for Function. ``"cholesky"`` and
            ``"cg"`` assume a symmetric positive-definite operator.
        psd: Whether ``op`` is symmetric positive-definite.
        tol: Relative tolerance of the iterative solvers.
        maxiter: Iteration cap of the iterative solvers.

    Returns:
        Solution of ``in_shape`` (``+ (k,)``).
    """
    _check_square(op)
    b = jnp.asarray(b)
    batch_shape = b.shape[len(op.out_shape) :]
    if b.shape[: len(op.out_shape)] != op.out_shape or len(batch_shape) > 1:
        raise ValueError(
            f"Expected rhs of shape {op.out_shape} (+ (k,)), got {b.shape}"
        )
    rhs = b.reshape(op.out_size, -1)

    if not isinstance(solver, str):
        lx = require_lineax()
        lx_op = to_lineax(op, psd)
        x = jax.vmap(
            lambda column: lx.linear_solve(lx_op, column, solver).value,
            in_axes=1,
            out_axes=1,
        )(rhs)
        return x.reshape(op.in_shape + batch_shape)

    if solver == "auto":
        if isinstance(op, Diagonal):
            solver = "diagonal"
        elif isinstance(op, Dense):
            solver = "cholesky" if psd else "lu"
        else:
            solver = "cg" if psd else "gmres"

    if solver == "diagonal":
        x = rhs / op._full_diag()[:, None]
    elif solver == "lu":
        x = jnp.linalg.solve(op.as_array(), rhs)
    elif solver == "cholesky":
        factor = jax.scipy.linalg.cho_factor(_symmetric_part(op.as_array()))
        x = jax.scipy.linalg.cho_solve(factor, rhs)
    elif solver in _ITERATIVE_SOLVERS:
        method = _ITERATIVE_SOLVERS[solver]

        def unit_scale_solve(matvec, column):
            # Krylov methods square residual norms, which over/underflows
            # float32 for extreme |b|; solve at unit scale instead.
            scale = jnp.max(jnp.abs(column))
            scale = jnp.where(scale > 0, scale, 1)
            return method(matvec, column / scale, tol=tol, maxiter=maxiter)[0] * scale

        def solve_column(column):
            # The rescaling is nonlinear in b; declaring the solve linear keeps
            # it transposable (e.g. inside lax.custom_root tangent solves).
            return jax.lax.custom_linear_solve(
                op._mv,
                column,
                unit_scale_solve,
                transpose_solve=unit_scale_solve,
                symmetric=solver == "cg",
            )

        # The jax.scipy solvers accept shaped vectors, so no flattening is needed.
        columns = rhs.reshape(op.out_shape + (-1,))
        x = jax.vmap(solve_column, in_axes=-1, out_axes=-1)(columns)
    else:
        raise ValueError(f"Unknown solver {solver!r}")
    return x.reshape(op.in_shape + batch_shape)


def to_lineax(op: LinearOperator, psd: bool = False):
    """Convert to a lineax operator on the flattened ``in_size`` vector space."""
    lx = require_lineax()
    tags = (lx.symmetric_tag, lx.positive_semidefinite_tag) if psd else ()
    if isinstance(op, Diagonal):
        return lx.DiagonalLinearOperator(op._full_diag())
    if isinstance(op, Dense):
        return lx.MatrixLinearOperator(op.matrix, tags)
    return lx.FunctionLinearOperator(
        op._flat_mv, jax.ShapeDtypeStruct((op.in_size,), op.dtype), tags
    )


def logdet(op: LinearOperator, *, psd: bool = False, **lanczos_kwargs) -> jax.Array:
    """``log |det op|`` of a square operator.

    Function operators need ``psd=True`` and use Lanczos quadrature
    (:func:`probjax.utils.linalg.lanczos_logdet`, which receives
    ``lanczos_kwargs``); call ``as_dense()`` first for an exact result.
    """
    _check_square(op)
    if isinstance(op, Diagonal):
        return jnp.sum(jnp.log(jnp.abs(op._full_diag())))
    if isinstance(op, Dense):
        if psd:
            factor = jnp.linalg.cholesky(_symmetric_part(op.matrix))
            return 2 * jnp.sum(jnp.log(jnp.diagonal(factor)))
        return jnp.linalg.slogdet(op.matrix).logabsdet
    if not psd:
        raise ValueError(
            "Matrix-free logdet requires psd=True; call as_dense() otherwise"
        )
    flat = Function(op._flat_mv, (op.in_size,), (op.out_size,), op.dtype)
    return lanczos_logdet(flat, **lanczos_kwargs)

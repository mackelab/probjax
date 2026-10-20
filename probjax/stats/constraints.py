"""
Constraints (:mod:`probjax.stats.constraints`)
===========================================

This module implements various constraints for distribution parameters.
"""

from __future__ import annotations

from abc import abstractmethod
from functools import total_ordering
from typing import Any, Union

import jax.numpy as jnp
from jax.tree_util import tree_flatten

from probjax.utils.typing import Array, ArrayLike, PyTree

ConstraintLike = Union[Array, "Constraint"]

# TODO Maybe add differentiable _call methods


@total_ordering
class Constraint:
    """A constraint checks if a value satisfies the constraint."""

    def __contains__(self, val: PyTree[Any]) -> bool:
        # Should transform the value to satisfy the constraint.
        val_flatten, _ = tree_flatten(val)
        return all(bool(self._is_contained(x)) for x in val_flatten)

    def __eq__(self, __value: object) -> bool:
        return self.__class__ == __value.__class__

    def __lt__(self, __value: object) -> bool:
        return issubclass(self.__class__, __value.__class__)

    def _is_contained(self, x: ConstraintLike) -> bool:
        """Dispatch on runtime kind; subclasses only implement the two checks."""
        if isinstance(x, Array):
            return self._check_array(x)
        if isinstance(x, Constraint):
            return self._check_constraint(x)
        return False

    def _check_array(self, x: Array) -> bool:
        return False

    @abstractmethod
    def _check_constraint(self, x: Constraint) -> bool:
        pass

    def __repr__(self) -> str:
        return type(self).__name__.lower()

    def __str__(self) -> str:
        return self.__repr__()


class Distribution(Constraint):
    """A constraint that checks if a value is a distribution."""

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, Distribution)


class Real(Constraint):
    """A constraint that checks if a value is real."""

    def _check_array(self, x: Array) -> bool:
        return bool(jnp.all(jnp.isreal(x)))

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, Real)


class Integer(Real):
    """A constraint that checks if a value is an integer."""

    def _check_array(self, x: Array) -> bool:
        return bool(jnp.issubdtype(x.dtype, jnp.integer))

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, (Integer, Boolean))


class Boolean(Integer):
    """A constraint that checks if a value is boolean."""

    def _check_array(self, x: Array) -> bool:
        return bool(jnp.issubdtype(x.dtype, jnp.bool_))

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, Boolean)


class Interval(Real):
    """A constraint that checks if a value is in an interval."""

    def __init__(
        self,
        lower: float | ArrayLike,
        upper: float | ArrayLike,
        closed_left: bool = True,
        closed_right: bool = True,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.closed_left = closed_left
        self.closed_right = closed_right

    def _check_array(self, x: Array) -> bool:
        term1 = x >= self.lower if self.closed_left else x > self.lower
        term2 = x <= self.upper if self.closed_right else x < self.upper
        return (
            Real._check_array(self, x) and bool(jnp.all(term1)) and bool(jnp.all(term2))
        )

    def _check_constraint(self, x: Constraint) -> bool:
        if not isinstance(x, Interval):
            return False
        lower = jnp.asarray(self.lower)
        upper = jnp.asarray(self.upper)
        term1 = x.lower >= lower if self.closed_left else x.lower > lower
        term2 = x.upper <= upper if self.closed_right else x.upper < upper
        return bool(term1) and bool(term2)


class UnitInterval(Interval):
    """A constraint that checks if a value is in the unit interval [0, 1]."""

    def __init__(self) -> None:
        super().__init__(0, 1)


class Simplex(UnitInterval):
    """A constraint that checks if a value is in the simplex."""

    def _check_array(self, x: Array) -> bool:
        if not UnitInterval._check_array(self, x):
            return False
        return bool(jnp.isclose(jnp.sum(x), 1.0))

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, Simplex)


class Positive(Interval):
    """A constraint that checks if a value is positive."""

    def __init__(self) -> None:
        super().__init__(0, jnp.inf)


class StrictPositive(Interval):
    """A constraint that checks if a value is strictly positive."""

    def __init__(self) -> None:
        super().__init__(0, jnp.inf, closed_left=False)


class Negative(Interval):
    """A constraint that checks if a value is negative."""

    def __init__(self) -> None:
        super().__init__(-jnp.inf, 0)


class StrictNegative(Interval):
    """A constraint that checks if a value is strictly negative."""

    def __init__(self) -> None:
        super().__init__(-jnp.inf, 0, closed_right=False)


class IntegerInterval(Integer, Interval):
    """A constraint that checks if a value is in an integer interval."""

    def __init__(self, lower: float, upper: float) -> None:
        self.lower = lower
        self.upper = upper

    def _check_array(self, x: Array) -> bool:
        lower_ok = bool(jnp.all(x >= self.lower))
        upper_ok = bool(jnp.all(x <= self.upper))
        return Integer._check_array(self, x) and lower_ok and upper_ok

    def _check_constraint(self, x: Constraint) -> bool:
        if not isinstance(x, IntegerInterval):
            return False
        return x.lower >= self.lower and x.upper <= self.upper


class PositiveInteger(IntegerInterval):
    """A constraint that checks if a value is a positive integer."""

    def __init__(self) -> None:
        super().__init__(0, jnp.inf)


class NonNegativeInteger(IntegerInterval):
    """A constraint that checks if a value is a non-negative integer."""

    def __init__(self) -> None:
        super().__init__(0, jnp.inf)


class NegativeInteger(IntegerInterval):
    """A constraint that checks if a value is a negative integer."""

    def __init__(self) -> None:
        super().__init__(-jnp.inf, 0)


class StrictPositiveInteger(IntegerInterval):
    """A constraint that checks if a value is a strictly positive integer."""

    def __init__(self) -> None:
        super().__init__(1, jnp.inf)


class StrictNegativeInteger(IntegerInterval):
    """A constraint that checks if a value is a strictly negative integer."""

    def __init__(self) -> None:
        super().__init__(-jnp.inf, -1)


class FiniteSet(Constraint):
    """A constraint that checks if a value is in a finite set."""

    def __init__(self, values: Array) -> None:
        self.values = values

    def _check_array(self, x: Array) -> bool:
        return bool(jnp.all(jnp.isin(x, jnp.asarray(self.values))))

    def _check_constraint(self, x: Constraint) -> bool:
        values = jnp.asarray(self.values)
        if isinstance(x, FiniteSet):
            return bool(jnp.all(jnp.isin(x.values, values)))
        if isinstance(x, Interval):
            min_val = float(jnp.min(values))
            max_val = float(jnp.max(values))
            lower = jnp.asarray(x.lower)
            upper = jnp.asarray(x.upper)
            return bool(lower >= min_val) and bool(upper <= max_val)
        return False


class UnitSquare(Interval):
    def __init__(self) -> None:
        super().__init__(-1, 1)


class Matrix(Real):
    def _check_array(self, x: Array) -> bool:
        return Real._check_array(self, x) and x.ndim >= 2

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, Matrix)


class SquareMatrix(Matrix):
    def _check_array(self, x: Array) -> bool:
        return Matrix._check_array(self, x) and x.shape[-1] == x.shape[-2]

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, SquareMatrix)


class SymmetricMatrix(SquareMatrix):
    def _check_array(self, x: Array) -> bool:
        is_square = SquareMatrix._check_array(self, x)
        return bool(is_square and jnp.allclose(x, jnp.swapaxes(x, -1, -2)))

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, SymmetricMatrix)


class SymmetricPositiveDefiniteMatrix(SymmetricMatrix):
    """A constraint that checks if a value is a symmetric positive definite matrix."""

    def _check_array(self, x: Array) -> bool:
        if not SymmetricMatrix._check_array(self, x):
            return False
        eigvals = jnp.linalg.eigvalsh(x)
        return bool(jnp.all(eigvals > 0))

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, SymmetricPositiveDefiniteMatrix)


def _is_orthonormal_columns(x: Array) -> bool:
    gram = jnp.swapaxes(x, -1, -2) @ x
    identity = jnp.eye(x.shape[-1], dtype=x.dtype)
    return bool(jnp.allclose(gram, identity))


class Spherical(Constraint):
    """A constraint that checks if a value lies on a unit sphere."""

    def _check_array(self, x: Array) -> bool:
        squared_norm = jnp.sum(x**2, axis=-1)
        return bool(jnp.allclose(squared_norm, 1.0))

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, Spherical)


class Stiefel(Constraint):
    """A constraint that checks if a value is a Stiefel matrix (orthogonal matrix)."""

    def _check_array(self, x: Array) -> bool:
        return _is_orthonormal_columns(x)

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, Stiefel)


class Grassmannian(Constraint):
    """A constraint that checks if a value is a Grassmannian matrix (subspace)."""

    def _check_array(self, x: Array) -> bool:
        return _is_orthonormal_columns(x)

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, Grassmannian)


class Lorentz(Constraint):
    """A constraint that checks if a value lies on the Lorentz manifold."""

    def _check_array(self, x: Array) -> bool:
        time_component = x[..., 0] ** 2
        spatial_component = jnp.sum(x[..., 1:] ** 2, axis=-1)
        return bool(jnp.allclose(time_component - spatial_component, 1.0))

    def _check_constraint(self, x: Constraint) -> bool:
        return isinstance(x, Lorentz)


# Numerical constraints
real = Real()
integer = Integer()
boolean = Boolean()
positive = Positive()
positive_integer = PositiveInteger()
non_negative_integer = NonNegativeInteger()
negative_integer = NegativeInteger()
strict_positive = StrictPositive()
strict_negative = StrictNegative()
strict_positive_integer = StrictPositiveInteger()
strict_negative_integer = StrictNegativeInteger()
negative = Negative()
interval = Interval
finit_set = FiniteSet
unit_interval = UnitInterval()
unit_square = UnitSquare()
unit_integer_interval = IntegerInterval(0, 1)
simplex = Simplex()
matrix = Matrix()
square_matrix = SquareMatrix()
symmetric_matrix = SymmetricMatrix()
symmetric_positive_definite_matrix = SymmetricPositiveDefiniteMatrix()


# Other constraints
distribution = Distribution()

# Create singleton instances
spherical = Spherical()
stiefel = Stiefel()
grassmannian = Grassmannian()
lorentz = Lorentz()

__all__ = [
    "real",
    "integer",
    "boolean",
    "positive",
    "positive_integer",
    "non_negative_integer",
    "negative",
    "negative_integer",
    "strict_negative_integer",
    "interval",
    "finit_set",
    "unit_interval",
    "unit_square",
    "unit_integer_interval",
    "simplex",
    "matrix",
    "square_matrix",
    "symmetric_matrix",
    "symmetric_positive_definite_matrix",
    "distribution",
    "spherical",
    "stiefel",
    "grassmannian",
    "lorentz",
]

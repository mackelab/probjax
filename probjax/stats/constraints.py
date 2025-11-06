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

from probjax.utils.typing import Array, PyTree

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

    @abstractmethod
    def _is_contained(self, x: ConstraintLike) -> bool:
        pass

    def __repr__(self) -> str:
        return type(self).__name__.lower()

    def __str__(self) -> str:
        return self.__repr__()


class Distribution(Constraint):
    """A constraint that checks if a value is a distribution."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Constraint):
            return isinstance(x, Distribution)
        return False


class Real(Constraint):
    """A constraint that checks if a value is real."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            return bool(jnp.all(jnp.isreal(x)))
        if isinstance(x, Constraint):
            return isinstance(x, Real)
        raise TypeError(f"Cannot check if {x} of type {type(x)} is real.")


class Integer(Real):
    """A constraint that checks if a value is an integer."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            return bool(jnp.issubdtype(x.dtype, jnp.integer))
        if isinstance(x, Constraint):
            return isinstance(x, (Integer, Boolean))
        return False


class Boolean(Integer):
    """A constraint that checks if a value is boolean."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            return bool(jnp.issubdtype(x.dtype, jnp.bool_))
        if isinstance(x, Constraint):
            return isinstance(x, Boolean)
        return False


class Interval(Real):
    """A constraint that checks if a value is in an interval."""

    def __init__(
        self,
        lower: float,
        upper: float,
        closed_left: bool = True,
        closed_right: bool = True,
    ) -> None:
        self.lower = lower
        self.upper = upper
        self.closed_left = closed_left
        self.closed_right = closed_right

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            term1 = x >= self.lower if self.closed_left else x > self.lower
            term2 = x <= self.upper if self.closed_right else x < self.upper
            return (
                Real._is_contained(self, x)
                and bool(jnp.all(term1))
                and bool(jnp.all(term2))
            )
        if isinstance(x, Constraint):
            if not isinstance(x, Interval):
                return False
            term1 = x.lower >= self.lower if self.closed_left else x.lower > self.lower
            term2 = x.upper <= self.upper if self.closed_right else x.upper < self.upper
            return term1 and term2
        return False


class UnitInterval(Interval):
    """A constraint that checks if a value is in the unit interval [0, 1]."""

    def __init__(self) -> None:
        super().__init__(0, 1)


class Simplex(UnitInterval):
    """A constraint that checks if a value is in the simplex."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            if not UnitInterval._is_contained(self, x):
                return False
            sum_to_one = jnp.isclose(jnp.sum(x), 1.0)
            return bool(sum_to_one)
        if isinstance(x, Constraint):
            return isinstance(x, Simplex)
        return False


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

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            lower_ok = bool(jnp.all(x >= self.lower))
            upper_ok = bool(jnp.all(x <= self.upper))
            return Integer._is_contained(self, x) and lower_ok and upper_ok
        if isinstance(x, Constraint):
            if not isinstance(x, IntegerInterval):
                return False
            return x.lower >= self.lower and x.upper <= self.upper
        return False


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

    def _is_contained(self, x: ConstraintLike) -> bool:
        values = jnp.asarray(self.values)
        if isinstance(x, Array):
            return bool(jnp.all(jnp.isin(x, values)))
        if isinstance(x, Constraint):
            if isinstance(x, FiniteSet):
                return bool(jnp.all(jnp.isin(x.values, values)))
            if isinstance(x, Interval):
                min_val = float(jnp.min(values))
                max_val = float(jnp.max(values))
                return x.lower >= min_val and x.upper <= max_val
        return False


class UnitSquare(Interval):
    def __init__(self) -> None:
        super().__init__(-1, 1)


class Matrix(Real):
    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            return Real._is_contained(self, x) and x.ndim >= 2
        if isinstance(x, Constraint):
            return isinstance(x, Matrix)
        return False


class SquareMatrix(Matrix):
    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            return Matrix._is_contained(self, x) and x.shape[-1] == x.shape[-2]
        if isinstance(x, Constraint):
            return isinstance(x, SquareMatrix)
        return False


class SymmetricMatrix(SquareMatrix):
    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            is_square = SquareMatrix._is_contained(self, x)
            return bool(is_square and jnp.allclose(x, jnp.swapaxes(x, -1, -2)))
        if isinstance(x, Constraint):
            return isinstance(x, SymmetricMatrix)
        return False


class SymmetricPositiveDefiniteMatrix(SymmetricMatrix):
    """A constraint that checks if a value is a symmetric positive definite matrix."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            if not SymmetricMatrix._is_contained(self, x):
                return False
            eigvals = jnp.linalg.eigvalsh(x)
            return bool(jnp.all(eigvals > 0))
        if isinstance(x, Constraint):
            return isinstance(x, SymmetricPositiveDefiniteMatrix)
        return False


class Spherical(Constraint):
    """A constraint that checks if a value lies on a unit sphere."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            squared_norm = jnp.sum(x**2, axis=-1)
            return bool(jnp.allclose(squared_norm, 1.0))
        if isinstance(x, Constraint):
            return isinstance(x, Spherical)
        return False


class Stiefel(Constraint):
    """A constraint that checks if a value is a Stiefel matrix (orthogonal matrix)."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            gram = jnp.swapaxes(x, -1, -2) @ x
            identity = jnp.eye(x.shape[-1], dtype=x.dtype)
            return bool(jnp.allclose(gram, identity))
        if isinstance(x, Constraint):
            return isinstance(x, Stiefel)
        return False


class Grassmannian(Constraint):
    """A constraint that checks if a value is a Grassmannian matrix (subspace)."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            gram = jnp.swapaxes(x, -1, -2) @ x
            identity = jnp.eye(x.shape[-1], dtype=x.dtype)
            return bool(jnp.allclose(gram, identity))
        if isinstance(x, Constraint):
            return isinstance(x, Grassmannian)
        return False


class Lorentz(Constraint):
    """A constraint that checks if a value lies on the Lorentz manifold."""

    def _is_contained(self, x: ConstraintLike) -> bool:
        if isinstance(x, Array):
            time_component = x[..., 0] ** 2
            spatial_component = jnp.sum(x[..., 1:] ** 2, axis=-1)
            return bool(jnp.allclose(time_component - spatial_component, 1.0))
        if isinstance(x, Constraint):
            return isinstance(x, Lorentz)
        return False


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

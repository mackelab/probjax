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
from jaxtyping import Array, PyTree

# TODO Maybe add differentiable _call methods


@total_ordering
class Constraint:
    """A constraint checks if a value satisfies the constraint."""

    def __contains__(self, val: PyTree[Union[Array, "Constraint"]]) -> bool:
        # Should transform the value to satisfy the constraint.
        val_flatten, _ = tree_flatten(val)
        return all(self._is_contained(x) for x in val_flatten)

    def __eq__(self, __value: object) -> bool:
        return self.__class__ == __value.__class__

    def __lt__(self, __value: object) -> bool:
        return issubclass(self.__class__, __value.__class__)

    @abstractmethod
    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        pass

    def __repr__(self) -> str:
        return type(self).__name__.lower()

    def __str__(self) -> str:
        return self.__repr__()


class Distribution(Constraint):
    """A constraint that checks if a value is a distribution."""

    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        return isinstance(x, Distribution)


class Real(Constraint):
    """A constraint that checks if a value is real."""

    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        if isinstance(x, Array):
            return jnp.isreal(x).all()
        elif isinstance(x, Constraint):
            return x == self
        else:
            raise TypeError(f"Cannot check if {x} of type {type(x)} is real.")


class Integer(Real):
    """A constraint that checks if a value is an integer."""

    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        if isinstance(x, Array):
            return jnp.issubdtype(x.dtype, jnp.integer)
        else:
            return isinstance(x, (Integer, Boolean))


class Boolean(Integer):
    """A constraint that checks if a value is boolean."""

    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        if isinstance(x, Array):
            return jnp.issubdtype(x.dtype, jnp.bool_)
        else:
            return isinstance(x, Boolean)


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

    def _is_contained(self, x: Array) -> bool:
        if isinstance(x, Array):
            term1 = x >= self.lower if self.closed_left else x > self.lower
            term2 = x <= self.upper if self.closed_right else x < self.upper
            return super()._is_contained(x) and all(term1) and all(term2)
        else:
            is_real = super()._is_contained(x)
            is_interval = isinstance(x, Interval)
            term1 = x.lower >= self.lower if self.closed_left else x.lower > self.lower
            term2 = x.upper <= self.upper if self.closed_right else x.upper < self.upper
            return is_real and is_interval and term1 and term2


class UnitInterval(Interval):
    """A constraint that checks if a value is in the unit interval [0, 1]."""

    def __init__(self) -> None:
        super().__init__(0, 1)


class Simplex(UnitInterval):
    """A constraint that checks if a value is in the simplex."""

    def _is_contained(self, x: Array) -> bool:
        return super()._is_contained(x) and jnp.sum(x) == 1


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

    def __init__(self, lower: int, upper: int) -> None:
        self.lower = lower
        self.upper = upper

    def _is_contained(self, x: Array) -> bool:
        if isinstance(x, Array):
            return (
                super()._is_contained(x) and all(x > self.lower) and all(x < self.upper)
            )
        else:
            is_integer = super()._is_contained(x)
            is_interval = isinstance(x, IntegerInterval)
            return (
                is_integer
                and is_interval
                and x.lower >= self.lower
                and x.upper <= self.upper
            )


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

    def _is_contained(self, x: Array) -> bool:
        if isinstance(x, Array):
            return x in self.values
        else:
            if isinstance(x, FiniteSet):
                return all([v in self.values for v in x.values])
            elif isinstance(x, Interval):
                min = jnp.min(self.values)
                max = jnp.max(self.values)
                return x.lower >= min and x.upper <= max
            else:
                return False


class UnitSquare(Interval):
    def __init__(self) -> None:
        super().__init__(-1, 1)


class Matrix(Real):
    def _is_contained(self, x: Any | Constraint) -> bool:
        return super()._is_contained(x) and len(x.shape) >= 2


class SquareMatrix(Matrix):
    def _is_contained(self, x: Any | Constraint) -> bool:
        return super()._is_contained(x) and x.shape[-1] == x.shape[-2]


class SymmetricMatrix(SquareMatrix):
    def _is_contained(self, x: Any | Constraint) -> bool:
        return super()._is_contained(x) and jnp.allclose(x, jnp.transpose(x, (-2, -1)))


class PositiveDefiniteMatrix(SymmetricMatrix):
    def _is_contained(self, x: Any | Constraint) -> bool:
        return super()._is_contained(x) and jnp.all(jnp.linalg.eigvals(x) > 0)


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
positive_definite_matrix = PositiveDefiniteMatrix()

# Other constraints
distribution = Distribution()


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
    "distribution",
]

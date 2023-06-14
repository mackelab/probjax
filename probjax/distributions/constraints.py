from typing import Any
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

from jaxtyping import PyTree, Array, Float, Int, Bool
from typing import Union

from abc import abstractmethod
from functools import total_ordering


# TODO Maybe add differentiable _call methods


@total_ordering
class Constraint:
    """A constraint checks if a value satisfies the constraint."""

    def __call__(self, val: PyTree[Array]) -> PyTree[Array]:
        # Should transform the value to satisfy the constraint.
        return jax.tree_map(self._call, val)

    def __contains__(self, val: PyTree[Union[Array, "Constraint"]]) -> bool:
        # Should transform the value to satisfy the constraint.
        val_flatten, _ = tree_flatten(val)
        return all(self._is_contained(x) for x in val_flatten)

    def __eq__(self, __value: object) -> bool:
        return self.__class__ == __value.__class__

    def __lt__(self, __value: object) -> bool:
        return issubclass(__value.__class__, self.__class__)

    @abstractmethod
    def _call(self, x: Array) -> Array:
        pass

    @abstractmethod
    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        pass


class Real(Constraint):
    """A constraint that checks if a value is real."""

    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        if isinstance(x, Array):
            return jnp.isreal(x).all()
        else:
            return (
                isinstance(x, Real) or isinstance(x, Integer) or isinstance(x, Boolean)
            )

    def _call(self, x: Array) -> Array:
        return jnp.real(x)

    def __repr__(self) -> str:
        return type(self).__name__.lower()

    __str__ = __repr__


class Integer(Real):
    """A constraint that checks if a value is an integer."""

    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        if isinstance(x, Array):
            return jnp.issubdtype(x.dtype, jnp.integer)
        else:
            return isinstance(x, Integer) or isinstance(x, Boolean)

    def _call(self, x: Array) -> Array:
        return jnp.round(x)


class Boolean(Integer):
    """A constraint that checks if a value is boolean."""

    def _is_contained(self, x: Union[Array, "Constraint"]) -> bool:
        if isinstance(x, Array):
            return jnp.issubdtype(x.dtype, jnp.bool_)
        else:
            return isinstance(x, Boolean)

    def _call(self, x: Array) -> Array:
        return jnp.round(jnp.clip(x, 0, 1)).astype(jnp.bool_)


class Interval(Real):
    """A constraint that checks if a value is in an interval."""

    def __init__(self, lower: float, upper: float) -> None:
        self.lower = lower
        self.upper = upper

    def _is_contained(self, x: Array) -> bool:
        if isinstance(x, Array):
            return (
                super()._is_contained(x)
                and (x > self.lower).all()
                and (x < self.upper).all()
            )
        else:
            is_real = super()._is_contained(x)
            is_interval = isinstance(x, Interval)
            return (
                is_real
                and is_interval
                and x.lower >= self.lower
                and x.upper <= self.upper
            )

    def _call(self, x: Array) -> Array:
        return jnp.clip(x, self.lower, self.upper)


class IntegerInterval(Integer, Interval):
    """A constraint that checks if a value is in an interval."""

    def __init__(self, lower: int, upper: int) -> None:
        self.lower = lower
        self.upper = upper

    def _is_contained(self, x: Array) -> bool:
        if isinstance(x, Array):
            return (
                super()._is_contained(x)
                and (x > self.lower).all()
                and (x < self.upper).all()
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

    def _call(self, x: Array) -> Array:
        return jnp.clip(x, self.lower, self.upper)


class Positive(Interval):
    def __init__(self) -> None:
        super().__init__(0, jnp.inf)

    def _call(self, x: Array) -> Array:
        return super()._call(x)


class Negative(Interval):
    def __init__(self) -> None:
        super().__init__(-jnp.inf, 0)

    def _call(self, x: Array) -> Array:
        return super()._call(x)


real = Real()
integer = Integer()
boolean = Boolean()
positive = Positive()
negative = Negative()
interval = Interval
unit_interval = Interval(0, 1)
unit_square = Interval(-1, 1)


__all__ = [
    "real",
    "integer",
    "boolean",
    "positive",
    "negative",
    "interval",
    "unit_interval",
    "unit_square",
]

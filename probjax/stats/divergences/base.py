"""
Base classes and utilities for statistical divergences.
"""

import warnings
from typing import Callable, Dict, Optional, Tuple, Type

import jax

from probjax.stats.base import rv_generic

__all__ = ["register_divergence", "divergence"]

_DIV_REGISTRY: Dict[str, Dict[Tuple[Type, Type], Callable]] = {}
_DIV_MEMOIZE: Dict[str, Dict[Tuple[Type, Type], Callable]] = {}


class Match:
    """A class for finding the most specific match in a registry of functions.

    This class is used to find the most specific match for a given set of types
    in a registry of functions. The most specific match is determined by the
    inheritance hierarchy of the types.

    Attributes:
        registry (Dict[Tuple[Type, ...], Callable]): A dictionary mapping type tuples to functions.
    """

    def __init__(self):
        self.registry = {}

    def register(self, types: Tuple[Type, ...], func: Callable) -> None:
        """Register a function for a given tuple of types.

        Args:
            types (Tuple[Type, ...]): A tuple of types to register the function for.
            func (Callable): The function to register.
        """
        self.registry[types] = func

    def find_match(self, types: Tuple[Type, ...]) -> Optional[Callable]:
        """Find the most specific match for a given tuple of types.

        Args:
            types (Tuple[Type, ...]): A tuple of types to find a match for.

        Returns:
            Optional[Callable]: The most specific matching function, or None if no match is found.
        """
        if types in self.registry:
            return self.registry[types]

        # Find all matches
        matches = []
        for reg_types, func in self.registry.items():
            if len(reg_types) != len(types):
                continue
            if all(issubclass(t, rt) for t, rt in zip(types, reg_types, strict=False)):
                matches.append((reg_types, func))

        if not matches:
            return None

        # Find the most specific match
        best_match = matches[0]
        for match in matches[1:]:
            if all(issubclass(rt1, rt2) for rt1, rt2 in zip(match[0], best_match[0], strict=False)):
                best_match = match

        return best_match[1]


def register_divergence(name: str, type_p: Type, type_q: Type):
    """
    Decorator to register a pairwise function with :meth:`divergence`.
    Usage::

        @register_divergence("kl", Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised.

    Args:
        name (str): Name of the divergence to register
        type_p (type): A subclass of :class:`~probjax.stats.base.rv_generic` or an instance of such a class.
        type_q (type): A subclass of :class:`~probjax.stats.base.rv_generic` or an instance of such a class.
    """
    # Get the actual class type if an instance is passed
    if not isinstance(type_p, type):
        type_p = type(type_p)
    if not isinstance(type_q, type):
        type_q = type(type_q)

    if not issubclass(type_p, rv_generic):
        raise TypeError(
            "Expected type_p to be a rv_generic subclass but got {}".format(type_p)
        )
    if not issubclass(type_q, rv_generic):
        raise TypeError(
            "Expected type_q to be a rv_generic subclass but got {}".format(type_q)
        )

    if name not in _DIV_REGISTRY:
        _DIV_REGISTRY[name] = {}
        _DIV_MEMOIZE[name] = {}

    def decorator(fun):
        _DIV_REGISTRY[name][type_p, type_q] = fun
        _DIV_MEMOIZE[name].clear()  # reset since lookup order may have changed
        return fun

    return decorator


def _dispatch(name: str, type_p: Type, type_q: Type):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    matches = [
        (super_p, super_q)
        for super_p, super_q in _DIV_REGISTRY[name]
        if issubclass(type_p, super_p) and issubclass(type_q, super_q)
    ]
    if not matches:
        return NotImplemented
    # Check that the left- and right- lexicographic orders agree.
    left_p, left_q = min(Match(*m) for m in matches).types
    right_q, right_p = min(Match(*reversed(m)) for m in matches).types
    left_fun = _DIV_REGISTRY[name][left_p, left_q]
    right_fun = _DIV_REGISTRY[name][right_p, right_q]
    if left_fun is not right_fun:
        warnings.warn(
            "Ambiguous divergence({}, {}). Please register_divergence({}, {})".format(
                type_p.__name__, type_q.__name__, left_p.__name__, right_q.__name__
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    return left_fun


def divergence(
    name: str,
    p: rv_generic,
    q: rv_generic,
    mc_samples: int = 0,
    key: Optional[jax.Array] = None,
    **kwargs,
) -> jax.Array:
    """Compute divergence between two distributions.

    Args:
        name (str): Name of the divergence to compute
        p (rv_generic): First distribution
        q (rv_generic): Second distribution
        mc_samples (int): Number of samples to use for Monte Carlo approximation.
            Defaults to 0. Then only analytic expressions.
        key (jax.Array): Key for random number generation.
            Defaults to None. Only required if mc_samples > 0.
        **kwargs: Additional keyword arguments passed to the divergence function.

    Returns:
        jax.Array: A batch of divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_divergence`.
    """
    try:
        fun = _DIV_MEMOIZE[name][type(p), type(q)]
    except KeyError:
        fun = _dispatch(name, type(p), type(q))
        _DIV_MEMOIZE[name][type(p), type(q)] = fun
    if fun is NotImplemented:
        raise NotImplementedError(
            "No divergence is implemented for p type {} and q type {}".format(
                p.__class__.__name__, q.__class__.__name__
            )
        )
    return fun(p, q, mc_samples=mc_samples, key=key, **kwargs)

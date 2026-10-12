"""
Base classes and utilities for statistical divergences.
"""

import warnings
from functools import total_ordering
from typing import Callable, Dict, Optional, Tuple, Type

import jax

from probjax.stats.base import rv_generic

__all__ = ["register_divergence", "divergence"]

_DIV_REGISTRY: Dict[str, Dict[Tuple[Type, Type], Callable]] = {}
_DIV_MEMOIZE: Dict[str, Dict[Tuple[Type, Type], Callable]] = {}


@total_ordering
class Match:
    """Orders (type_p, type_q) registrations by how specific they are.

    ``a <= b`` when each type of ``a`` is a subclass of the matching type of
    ``b``, comparing left to right and stopping at the first strictly more
    specific type. ``_dispatch`` takes the ``min`` of these.
    """

    __slots__ = ["types"]

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types, strict=True):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


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

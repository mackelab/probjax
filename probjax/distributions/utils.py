import jax

from jax.tree_util import tree_flatten, tree_unflatten

from functools import total_ordering


@total_ordering
class Match:
    # Subclass ordering...
    __slots__ = ["types"]

    def __init__(self, *types):
        self.types = types

    def __eq__(self, other):
        return self.types == other.types

    def __le__(self, other):
        for x, y in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True

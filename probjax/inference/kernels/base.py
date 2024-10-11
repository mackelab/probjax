from typing import Any, NamedTuple, Tuple

from blackjax.base import Info, State
from chex import PRNGKey
from jaxtyping import PyTree


class Params(NamedTuple):
    pass


class MCMCKernel:
    params: Params

    def init_state(self, position: PyTree) -> State:
        raise NotImplementedError("init_state method must be implemented")

    def init_params(self, position: PyTree):
        raise NotImplementedError("init_params method must be implemented")

    def adapt_params(
        self, key: PRNGKey, position: PyTree, num_steps: int = 100, **kwargs: Any
    ) -> Tuple[State, Info]:
        raise NotImplementedError("adapt_params method must be implemented")

    def __call__(self, key: PRNGKey, state: State) -> Tuple[State, Info]:
        raise NotImplementedError("__call__ method must be implemented")

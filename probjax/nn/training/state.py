"""The pytrees that travel through the training loop.

Three of them, and each answers one question: what is being trained
(:class:`FitState`), what one update produced (:class:`FitInfo`), and what a
whole run produced (:class:`FitResult`).
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
from jaxtyping import Array

from probjax.utils.typing import PyTree

__all__ = ["FitInfo", "FitResult", "FitState"]


class FitState(NamedTuple):
    """A training snapshot. Every leaf stays on the device.

    ``step`` counts completed updates. ``params`` always holds the raw
    optimizer parameters, even when the averaged ones are the ones selected for
    inference; ``ema_params`` holds the average and is ``None`` when no
    averaging was asked for.

    ``aux`` is an opaque pytree carried through the loop untouched. It is how
    a module's non-parameter state -- BatchNorm statistics, counters -- survives
    training, and the functional API leaves it alone.
    """

    params: PyTree[Array]
    opt_state: Any
    step: Array
    rng: Array
    ema_params: PyTree[Array] | None = None
    aux: Any = None


class FitInfo(NamedTuple):
    """What one update produced: the loss, and any auxiliary metrics.

    Both describe the loss *before* the update that follows them.
    """

    loss: Array
    metrics: Any = None


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("state", "losses", "metrics"),
    meta_fields=("use_ema",),
)
@dataclass(frozen=True)
class FitResult:
    """The outcome of a run: the final state and the stacked history.

    ``losses`` holds one entry per update that actually ran, so its length is
    the number of updates. ``metrics`` is the same history for whatever the
    loss reported alongside it. ``use_ema`` is static metadata: it says which
    parameters :attr:`params` selects.
    """

    state: FitState
    losses: Array
    metrics: Any = None
    use_ema: bool = False

    @property
    def params(self):
        """The parameters to use for inference: averaged if selected, else raw."""
        return self.state.ema_params if self.use_ema else self.state.params

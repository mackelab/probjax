"""Training configuration: the knobs, and the contract with the data.

:class:`FitConfig` holds everything about *how* to train and *how the data
reaches the loss*. It refers to no model and no dataset, so one configuration
describes any run.

A configuration only overrides the fields it names. The rest keep the value of
the configuration it is merged into -- a model's own recipe, or the defaults
below -- so a caller who wants one thing changed does not have to restate the
whole recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional, Union

import optax

from probjax.utils.guards import ensure_int

__all__ = ["DataSpec", "FitConfig", "Schedule"]


#: A named built-in, any optax schedule ``step -> learning_rate``, or ``None``
#: for a constant learning rate.
Schedule = Union[
    Literal["onecycle_cosine", "warmup_cosine", "constant"], optax.Schedule, None
]


#: Sentinel for a field the caller did not name. A configuration that is
#: merged into another one keeps the base's value for these.
_UNSET: Any = object()


#: The values a configuration falls back to when it names nothing.
_DEFAULTS: dict[str, Any] = {
    "max_steps": 1000,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "schedule": "onecycle_cosine",
    "clip_norm": 10.0,
    "ema_decay": None,
    "use_ema": False,
    "callback_every": 1,
    "optimizer": None,
}


@dataclass(frozen=True)
class DataSpec:
    """How the data is served to the loss.

    A batch is a mapping from names to arrays. The entry named by ``primary``
    is handed to the loss positionally and every other entry as a keyword
    argument; ``primary=None`` hands over the whole mapping instead.

    ``stream`` marks a source that *yields* batches rather than being one
    batch. A stream decides its own batch size, so the loop never resizes it.
    """

    primary: Optional[str] = None
    stream: bool = False


@dataclass(frozen=True, init=False)
class FitConfig:
    """How to train, and how the data is served to the loss.

    Only the fields named at construction are overridden; the rest keep the
    value of the configuration this one is merged into. ``data`` is the
    exception: it has no default, so a configuration that is not merged into
    anything must say what the data looks like.
    """

    data: DataSpec
    max_steps: int = 1000
    batch_size: Optional[int] = 512
    learning_rate: float = 1e-3
    schedule: Schedule = "onecycle_cosine"
    clip_norm: Optional[float] = 10.0
    ema_decay: Optional[float] = None
    use_ema: bool = False
    callback_every: int = 1
    optimizer: Optional[optax.GradientTransformation] = None

    #: The names of the fields that were set at construction. Not part of the
    #: configuration's value: two configurations that differ only in which
    #: fields were named are still different configurations.
    _explicit: frozenset = field(default=frozenset(), compare=False, repr=False)

    def __init__(
        self,
        data: DataSpec = _UNSET,
        max_steps: int = _UNSET,
        batch_size: Optional[int] = _UNSET,
        learning_rate: float = _UNSET,
        schedule: Schedule = _UNSET,
        clip_norm: Optional[float] = _UNSET,
        ema_decay: Optional[float] = _UNSET,
        use_ema: bool = _UNSET,
        callback_every: int = _UNSET,
        optimizer: Optional[optax.GradientTransformation] = _UNSET,
    ) -> None:
        given = {
            "data": data,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "schedule": schedule,
            "clip_norm": clip_norm,
            "ema_decay": ema_decay,
            "use_ema": use_ema,
            "callback_every": callback_every,
            "optimizer": optimizer,
        }
        object.__setattr__(
            self,
            "_explicit",
            frozenset(name for name, value in given.items() if value is not _UNSET),
        )
        for name, value in given.items():
            if value is _UNSET:
                value = _DEFAULTS.get(name, _UNSET)
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def explicit(self) -> frozenset:
        """The names of the fields that were set at construction."""
        return self._explicit

    def default_optimizer(self) -> optax.GradientTransformation:
        """The optimizer this configuration describes.

        Adam with the configured learning rate, schedule and clipping. A run
        needs a built optimizer: this is the one built from the configuration's
        own settings, and a caller who wants a different one passes it as
        ``optimizer=``.
        """
        schedule = self.schedule
        if callable(schedule):
            lr = schedule
        elif schedule is None or schedule == "constant" or self.max_steps <= 1:
            lr = self.learning_rate
        elif schedule == "onecycle_cosine":
            lr = optax.cosine_onecycle_schedule(
                transition_steps=max(1, self.max_steps),
                peak_value=self.learning_rate,
            )
        else:
            lr = optax.warmup_cosine_decay_schedule(
                init_value=self.learning_rate / 100.0,
                peak_value=self.learning_rate,
                warmup_steps=max(1, self.max_steps // 20),
                decay_steps=self.max_steps,
                end_value=self.learning_rate / 1000.0,
            )
        adam = optax.adam(lr)
        if self.clip_norm is None:
            return adam
        return optax.chain(optax.clip_by_global_norm(self.clip_norm), adam)

    def merged(self, **changes) -> FitConfig:
        """A copy of this configuration with some fields replaced.

        The replaced fields become explicit; the rest keep their status, so a
        merged configuration still overrides only what it names.
        """
        unknown = set(changes) - set(_DEFAULTS) - {"data"}
        if unknown:
            raise TypeError(f"unknown configuration fields {sorted(unknown)}.")
        return merge_config(self, FitConfig(**changes))

    def __post_init__(self) -> None:
        if self.max_steps is not _UNSET:
            ensure_int("max_steps", self.max_steps, min_value=0)
        if self.batch_size is not _UNSET and self.batch_size is not None:
            ensure_int("batch_size", self.batch_size, min_value=1)
        if self.learning_rate is not _UNSET and not self.learning_rate > 0:
            raise ValueError(
                f"learning_rate must be positive; got {self.learning_rate!r}."
            )
        if (
            self.clip_norm is not _UNSET
            and self.clip_norm is not None
            and self.clip_norm <= 0
        ):
            raise ValueError(
                f"clip_norm must be positive or None; got {self.clip_norm!r}."
            )
        if (
            self.ema_decay is not _UNSET
            and self.ema_decay is not None
            and not 0 <= self.ema_decay < 1
        ):
            raise ValueError(
                f"ema_decay must be in [0, 1) or None; got {self.ema_decay!r}."
            )
        if self.callback_every is not _UNSET:
            ensure_int("callback_every", self.callback_every, min_value=1)
        if self.schedule is not _UNSET and not (
            callable(self.schedule)
            or self.schedule is None
            or self.schedule in ("onecycle_cosine", "warmup_cosine", "constant")
        ):
            raise ValueError(
                "schedule must be 'onecycle_cosine', 'warmup_cosine', "
                "'constant', None or an optax schedule; got "
                f"{self.schedule!r}."
            )
        if self.data is not _UNSET and not isinstance(self.data, DataSpec):
            raise TypeError(f"data must be a DataSpec; got {type(self.data).__name__}.")
        if self.optimizer is None:
            # No optimizer was brought along, so the one this configuration
            # describes is the one the run gets.
            object.__setattr__(self, "optimizer", self.default_optimizer())


def merge_config(base: FitConfig, override: FitConfig) -> FitConfig:
    """The fields ``override`` names, on top of ``base``.

    A field the override does not name keeps the base's value, so a caller
    who changes one setting does not silently drop the rest of a model's
    recipe.
    """
    values = {name: getattr(base, name) for name in _DEFAULTS}
    values["data"] = base.data
    for name in override.explicit:
        values[name] = getattr(override, name)
    if "optimizer" not in base.explicit and "optimizer" not in override.explicit:
        # Neither side brought an optimizer, so the default follows the
        # recipe: build it again from the merged settings.
        values["optimizer"] = None
    merged = FitConfig(**values)
    object.__setattr__(merged, "_explicit", base.explicit | override.explicit)
    return merged


def check_config(config: FitConfig) -> None:
    """The checks that need the whole configuration, not one field."""
    if config.data is _UNSET:
        raise ValueError(
            "FitConfig needs data=DataSpec(...) to say how the data is served "
            "to the loss: DataSpec(primary='data') for a mapping whose 'data' "
            "entry is the training target, DataSpec() to hand over the whole "
            "batch."
        )
    if config.use_ema and config.ema_decay is None:
        raise ValueError("use_ema requires ema_decay.")
    if not isinstance(config.optimizer, optax.GradientTransformation):
        raise TypeError(
            "optimizer must be an optax.GradientTransformation; got "
            f"{type(config.optimizer).__name__}."
        )


def config_fields() -> tuple[str, ...]:
    """The names of the configuration's fields."""
    return tuple(f.name for f in fields(FitConfig) if not f.name.startswith("_"))

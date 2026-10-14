"""Shared runtime type guards and small normalizers.

Centralizes the ``isinstance`` / ``hasattr`` / ``callable`` checks that were
previously copy-pasted across ``stats``, ``inference``, ``nn`` and ``core``.
Keep this module dependency-free (only ``jax``, stdlib) so every subpackage
can import it without cycles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from typing import Any, TypeGuard

import jax
import jax.numpy as jnp


def is_array_like(x: Any) -> bool:
    """Duck-type check: does ``x`` look like a JAX array (incl. tracers)?"""
    return hasattr(x, "shape") and hasattr(x, "dtype")


def has_attrs(x: Any, *names: str) -> bool:
    """True if ``x`` has all of the given attributes."""
    return all(hasattr(x, name) for name in names)


def is_mapping(x: Any) -> TypeGuard[Mapping]:
    return isinstance(x, Mapping)


def get_field(params: Any, name: str) -> Any:
    """Read ``params[name]`` for mappings, else ``params.<name>``."""
    if isinstance(params, Mapping):
        if name not in params:
            raise ValueError(f"params must have a {name} field")
        return params[name]
    if not hasattr(params, name):
        raise ValueError(f"params must have a {name} field")
    return getattr(params, name)


def replace_field(params: Any, **updates: Any) -> Any:
    """Replace fields in a mapping, NamedTuple, or dataclass."""
    if isinstance(params, Mapping):
        return type(params)(params, **updates)
    if hasattr(params, "_replace"):
        return params._replace(**updates)
    try:
        return replace(params, **updates)
    except TypeError as error:
        raise TypeError("params must be a mapping, NamedTuple, or dataclass") from error


def params_to_dict(params: Any) -> dict:
    """Normalize mapping / NamedTuple / dataclass params to a plain dict."""
    if params is None:
        return {}
    if hasattr(params, "_asdict"):
        return params._asdict()
    if isinstance(params, Mapping):
        return dict(params)
    if is_dataclass(params):
        return asdict(params)
    raise TypeError("params must be a mapping, NamedTuple, or dataclass")


def ensure_int(name: str, value: Any, *, min_value: int | None = None) -> int:
    """Validate an integer argument, rejecting bools (since ``bool < int``)."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}.")
    return value


def ensure_tuple(name: str, value: int | Any, *, positive: bool = False) -> tuple:
    """Normalize ``int`` or iterable-of-ints to a ``tuple[int, ...]``."""
    if isinstance(value, int) and not isinstance(value, bool):
        if positive and value <= 0:
            raise ValueError(f"{name} must be positive.")
        return (value,)
    items = tuple(int(v) for v in value)
    if not items:
        raise ValueError(f"{name} must be non-empty.")
    if positive and any(v <= 0 for v in items):
        raise ValueError(f"All entries in {name} must be positive.")
    return items


def ensure_array(name: str, value: Any) -> jax.Array:
    """Require a non-None param and return it as an array."""
    if value is None:
        raise ValueError(f"{name} must be provided.")
    return jnp.asarray(value)


def call_if_callable(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn(*args, **kwargs)`` if callable, else return ``fn``."""
    return fn(*args, **kwargs) if callable(fn) else fn


def is_tracing(x: Any) -> bool:
    """True if ``x`` is a JAX tracer (staged value inside jit/vmap/grad)."""
    return isinstance(x, jax.core.Tracer)


def as_concrete_bool(x: Any) -> bool:
    """``bool(x)`` guarded against tracers; raises a clear error if staged."""
    if is_tracing(x):
        raise ValueError(f"Expected a concrete value, got tracer {x!r}.")
    return bool(x)


def normalize_rng(rng: Any) -> jax.Array:
    """Accept an int seed or key array; always return a PRNG key."""
    if isinstance(rng, int) and not isinstance(rng, bool):
        return jax.random.PRNGKey(int(rng))
    return rng


def normalize_devices(devices: Any) -> list:
    """Accept None / single device / list-or-tuple; always return a list."""
    if devices is None:
        try:
            cpu = jax.devices("cpu")
        except Exception:
            cpu = []
        return [cpu[0] if cpu else jax.devices()[0]]
    if isinstance(devices, (list, tuple)):
        return list(devices)
    return [devices]

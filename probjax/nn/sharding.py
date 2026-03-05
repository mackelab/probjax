from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence

import jax
from flax import nnx
from jax.sharding import Mesh, PartitionSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _partition_spec_or_none(*axes: str | None) -> PartitionSpec | None:
    """Build a PartitionSpec unless every axis is None."""
    if all(axis is None for axis in axes):
        return None
    return PartitionSpec(*axes)


def _filter_constructor_kwargs(ctor, **kwargs):
    try:
        param_names = inspect.signature(ctor).parameters.keys()
    except (ValueError, TypeError):
        return kwargs
    return {key: kwargs[key] for key in kwargs if key in param_names}


def _resolve_init_fn(ctor, name: str, default_fn):
    if isinstance(ctor, partial) and ctor.keywords and name in ctor.keywords:
        return ctor.keywords[name]
    return default_fn


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearShardingSpec:
    """Partition specs for a single linear (dense) layer."""

    kernel: PartitionSpec | None = None
    bias: PartitionSpec | None = None
    activation: PartitionSpec | None = None


@dataclass(frozen=True)
class NormShardingSpec:
    """Partition specs for a normalization layer (LayerNorm, RMSNorm, etc.).

    Defaults to fully replicated (empty PartitionSpec) since norm parameters
    are small and benefit from being available on every device.
    """

    scale: PartitionSpec = PartitionSpec()
    bias: PartitionSpec = PartitionSpec()


# ---------------------------------------------------------------------------
# Base ShardingCfg – mesh resolution + fundamental operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardingCfg:
    """Minimal sharding configuration: mesh resolution and core operations.

    Subclass or mix-in additional ``*ShardingMixin`` classes to add
    layer-specific partition spec generation.

    **Convenience API** — modules should call methods on ``self.sharding_cfg``
    rather than using free functions:

    - ``constrain(value, spec)`` — reshard an activation
    - ``constrain_for_rank(value, rank)`` — reshard by tensor rank
    - ``norm_kwargs(ctor)`` — kwargs for norm layer constructors
    - ``linear_kwargs(ctor, spec)`` — kwargs for linear layer constructors
    - ``make_linear_ctor(base_ctor, spec)`` — constructor with baked-in sharding
    - ``partitioned_init(init_fn, spec)`` — wrap initializer with partitioning
    - ``as_type(target_cls)`` — upcast to a composed config subclass
    - ``resolve_or_noop(cfg)`` — class method returning cfg or ``_NOOP``
    """

    mesh: Mesh | None = None
    data_axis: str = "data"
    model_axis: str = "model"

    # -- Class-level resolution -------------------------------------------

    @classmethod
    def resolve(cls, cfg: ShardingCfg | None) -> ShardingCfg | None:
        """Return *cfg* if given, else auto-detect from ``jax.sharding.get_mesh()``."""
        if cfg is not None:
            return cfg
        mesh = jax.sharding.get_mesh()
        axis_names = tuple(getattr(mesh, "axis_names", ()))
        return cls(mesh=mesh) if axis_names else None

    @classmethod
    def resolve_or_noop(cls, cfg: ShardingCfg | None) -> ShardingCfg:
        """Return *cfg* if given, auto-detect, or the noop singleton.

        Guarantees a non-``None`` return so callers never need null checks.
        """
        resolved = cls.resolve(cfg)
        if resolved is not None:
            return resolved
        # _NOOP is defined after this class; use module-level lookup
        return globals()["_NOOP"]

    # -- Mesh helpers ------------------------------------------------------

    def resolved_mesh(self) -> Mesh | None:
        if self.mesh is not None:
            return self.mesh
        mesh = jax.sharding.get_mesh()
        axis_names = tuple(getattr(mesh, "axis_names", ()))
        return mesh if axis_names else None

    def has_axis(self, axis_name: str, mesh: Mesh | None = None) -> bool:
        mesh = self.resolved_mesh() if mesh is None else mesh
        return mesh is not None and axis_name in tuple(getattr(mesh, "axis_names", ()))

    def data_axis_name(self, mesh: Mesh | None = None) -> str | None:
        return self.data_axis if self.has_axis(self.data_axis, mesh) else None

    def model_axis_name(self, mesh: Mesh | None = None) -> str | None:
        mesh = self.resolved_mesh() if mesh is None else mesh
        if mesh is None or not self.has_axis(self.model_axis, mesh):
            return None
        return self.model_axis if mesh.shape[self.model_axis] > 1 else None

    # -- Core operations ---------------------------------------------------

    def apply_activation(
        self,
        value,
        spec: PartitionSpec | None,
        mesh: Mesh | None = None,
    ):
        """Reshard *value* according to *spec*."""
        resolved_mesh = self.resolved_mesh() if mesh is None else mesh
        if spec is None or resolved_mesh is None:
            return value
        try:
            return jax.sharding.reshard(
                value,
                jax.sharding.NamedSharding(resolved_mesh, spec),
            )
        except Exception:
            return value

    # -- Convenience methods (used by NN modules) --------------------------

    def constrain(self, value, spec: PartitionSpec | None = None):
        """Reshard *value* according to *spec* (alias for ``apply_activation``)."""
        return self.apply_activation(value, spec)

    def constrain_for_rank(self, value, rank: int):
        """Reshard *value* using a default spec derived from tensor *rank*.

        Uses ``linear_spec`` for rank-2 and ``mha_spec`` for rank-3.
        """
        mesh = self.resolved_mesh()
        if mesh is None:
            return value
        spec = activation_spec_for_rank(self, mesh, rank)
        return self.constrain(value, spec)

    def norm_kwargs(self, ctor) -> dict:
        """Build keyword arguments for a norm layer constructor."""
        mesh = self.resolved_mesh()
        if mesh is None:
            return {}
        spec = self._resolve_norm_spec(mesh)
        if spec is None:
            return {}
        kwargs = {}
        if spec.scale is not None:
            init_fn = _resolve_init_fn(ctor, "scale_init", nnx.initializers.ones)
            kwargs["scale_init"] = self.partitioned_init(init_fn, spec.scale)
        if spec.bias is not None:
            init_fn = _resolve_init_fn(ctor, "bias_init", nnx.initializers.zeros)
            kwargs["bias_init"] = self.partitioned_init(init_fn, spec.bias)
        return _filter_constructor_kwargs(ctor, **kwargs)

    def linear_kwargs(self, ctor, spec: LinearShardingSpec | None = None) -> dict:
        """Build keyword arguments for a linear layer constructor."""
        if spec is None:
            return {}
        mesh = self.resolved_mesh()
        kwargs = {}
        if spec.kernel is not None:
            init_fn = _resolve_init_fn(
                ctor, "kernel_init", nnx.initializers.lecun_normal()
            )
            kwargs["kernel_init"] = self.partitioned_init(init_fn, spec.kernel)
        if spec.bias is not None:
            init_fn = _resolve_init_fn(ctor, "bias_init", nnx.initializers.zeros)
            kwargs["bias_init"] = self.partitioned_init(init_fn, spec.bias)
        return _filter_constructor_kwargs(ctor, **kwargs)

    def make_linear_ctor(self, base_ctor, spec: LinearShardingSpec | None = None):
        """Return a linear layer constructor with baked-in sharding kwargs."""
        kwargs = self.linear_kwargs(base_ctor, spec)
        if not kwargs:
            return base_ctor

        def ctor(in_features, out_features):
            return base_ctor(in_features, out_features, **kwargs)

        return ctor

    def partitioned_init(
        self, init_fn: Callable, spec: PartitionSpec | None = None
    ) -> Callable:
        """Wrap *init_fn* with ``nnx.with_partitioning`` for *spec*."""
        mesh = self.resolved_mesh()
        if spec is None or mesh is None:
            return init_fn
        return nnx.with_partitioning(init_fn, sharding=tuple(spec), mesh=mesh)

    def as_type(self, target_cls: type) -> ShardingCfg:
        """Upcast to *target_cls* if not already an instance.

        Preserves existing fields and creates a new instance of *target_cls*
        using the base ``mesh``, ``data_axis``, ``model_axis`` fields.
        """
        if isinstance(self, target_cls):
            return self
        return target_cls(
            mesh=self.mesh, data_axis=self.data_axis, model_axis=self.model_axis
        )

    # -- Internal helpers --------------------------------------------------

    def _resolve_norm_spec(self, mesh: Mesh | None = None) -> NormShardingSpec | None:
        """Resolve a NormShardingSpec from this config."""
        if isinstance(self, NormShardingMixin):
            return self.norm_spec(mesh)
        resolved_mesh = self.resolved_mesh() if mesh is None else mesh
        if resolved_mesh is None:
            return None
        return NormShardingSpec()


# ---------------------------------------------------------------------------
# Mixins – opt-in spec generation for specific layer families
# ---------------------------------------------------------------------------


class LinearShardingMixin:
    """Adds ``linear_spec`` and ``mha_spec`` to a :class:`ShardingCfg`."""

    # Satisfy type-checkers; the concrete class supplies these via ShardingCfg.
    linear: LinearShardingSpec | None
    mha: LinearShardingSpec | None
    data_axis_name: Callable
    model_axis_name: Callable

    def linear_spec(self, mesh: Mesh | None = None) -> LinearShardingSpec | None:
        if self.linear is not None:  # type: ignore[attr-defined]
            return self.linear  # type: ignore[attr-defined]
        data_axis = self.data_axis_name(mesh)
        spec = LinearShardingSpec(
            kernel=None,
            bias=None,
            activation=_partition_spec_or_none(data_axis, None),
        )
        return spec if spec.activation is not None else None

    def mha_spec(self, mesh: Mesh | None = None) -> LinearShardingSpec | None:
        if self.mha is not None:  # type: ignore[attr-defined]
            return self.mha  # type: ignore[attr-defined]
        data_axis = self.data_axis_name(mesh)
        spec = LinearShardingSpec(
            kernel=None,
            bias=None,
            activation=_partition_spec_or_none(data_axis, None, None),
        )
        return spec if spec.activation is not None else None


class TransformerShardingMixin:
    """Adds transformer-specific partition specs."""

    transformer_hidden_activation: PartitionSpec | None
    transformer_input_activation: PartitionSpec | None
    transformer_mlp_column: LinearShardingSpec | None
    transformer_mlp_row: LinearShardingSpec | None
    data_axis_name: Callable

    def transformer_hidden_spec(self, mesh: Mesh | None = None) -> PartitionSpec | None:
        if self.transformer_hidden_activation is not None:  # type: ignore[attr-defined]
            return self.transformer_hidden_activation  # type: ignore[attr-defined]
        data_axis = self.data_axis_name(mesh)
        return _partition_spec_or_none(data_axis, None, None)

    def transformer_input_spec(self, mesh: Mesh | None = None) -> PartitionSpec | None:
        if self.transformer_input_activation is not None:  # type: ignore[attr-defined]
            return self.transformer_input_activation  # type: ignore[attr-defined]
        data_axis = self.data_axis_name(mesh)
        return _partition_spec_or_none(data_axis, None, None)

    def transformer_mlp_column_spec(
        self, mesh: Mesh | None = None
    ) -> LinearShardingSpec | None:
        if self.transformer_mlp_column is not None:  # type: ignore[attr-defined]
            return self.transformer_mlp_column  # type: ignore[attr-defined]
        activation = self.transformer_hidden_spec(mesh)
        spec = LinearShardingSpec(kernel=None, bias=None, activation=activation)
        return spec if spec.activation is not None else None

    def transformer_mlp_row_spec(
        self, mesh: Mesh | None = None
    ) -> LinearShardingSpec | None:
        if self.transformer_mlp_row is not None:  # type: ignore[attr-defined]
            return self.transformer_mlp_row  # type: ignore[attr-defined]
        activation = self.transformer_hidden_spec(mesh)
        spec = LinearShardingSpec(kernel=None, bias=None, activation=activation)
        return spec if spec.activation is not None else None


class NormShardingMixin:
    """Adds ``norm_spec`` to a :class:`ShardingCfg`."""

    norm: NormShardingSpec | None
    data_axis_name: Callable

    def norm_spec(self, mesh: Mesh | None = None) -> NormShardingSpec | None:
        """Return the norm sharding spec.

        Returns the explicit ``norm`` field if set, otherwise defaults to
        fully replicated (empty PartitionSpec for both scale and bias) when
        a mesh is available.
        """
        if self.norm is not None:  # type: ignore[attr-defined]
            return self.norm  # type: ignore[attr-defined]
        # Default: replicate norm params (they're small).
        return NormShardingSpec()


class SpatialShardingMixin:
    """Adds spatial activation spec (for UNet-style architectures)."""

    spatial_activation: PartitionSpec | None
    data_axis_name: Callable

    def spatial_activation_spec(self, mesh: Mesh | None = None) -> PartitionSpec | None:
        if self.spatial_activation is not None:  # type: ignore[attr-defined]
            return self.spatial_activation  # type: ignore[attr-defined]
        data_axis = self.data_axis_name(mesh)
        return _partition_spec_or_none(data_axis, None, None, None)


# ---------------------------------------------------------------------------
# Composed configs – ready-made combinations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearShardingCfg(ShardingCfg, LinearShardingMixin, NormShardingMixin):
    """ShardingCfg with linear/MHA/norm layer support."""

    linear: LinearShardingSpec | None = None
    mha: LinearShardingSpec | None = None
    norm: NormShardingSpec | None = None


@dataclass(frozen=True)
class TransformerShardingCfg(LinearShardingCfg, TransformerShardingMixin):
    """ShardingCfg with full transformer support (includes linear/MHA)."""

    transformer_hidden_activation: PartitionSpec | None = None
    transformer_input_activation: PartitionSpec | None = None
    transformer_mlp_column: LinearShardingSpec | None = None
    transformer_mlp_row: LinearShardingSpec | None = None


@dataclass(frozen=True)
class SpatialShardingCfg(LinearShardingCfg, SpatialShardingMixin):
    """ShardingCfg with spatial activation support (for UNets)."""

    spatial_activation: PartitionSpec | None = None


# ---------------------------------------------------------------------------
# Noop singleton – null-object pattern
# ---------------------------------------------------------------------------


class NoopShardingCfg(ShardingCfg):
    """No-op sharding config.  Every method is safe to call but does nothing.

    Modules store ``self.sharding_cfg = ShardingCfg.resolve_or_noop(cfg)``
    so they never need ``if self.sharding_cfg is not None:`` checks.
    """

    # Prevent dataclass inheritance from adding fields
    __slots__ = ()

    def __init__(self):
        # Bypass frozen-dataclass __init__; set fields via object.__setattr__
        object.__setattr__(self, "mesh", None)
        object.__setattr__(self, "data_axis", "data")
        object.__setattr__(self, "model_axis", "model")

    # -- Mesh helpers (all return None / False) ----------------------------
    def resolved_mesh(self) -> None:
        return None

    def has_axis(self, axis_name: str, mesh: Mesh | None = None) -> bool:
        return False

    def data_axis_name(self, mesh: Mesh | None = None) -> None:
        return None

    def model_axis_name(self, mesh: Mesh | None = None) -> None:
        return None

    # -- Core operations (all no-ops) --------------------------------------
    def apply_activation(self, value, spec=None, mesh=None):
        return value

    def constrain(self, value, spec=None):
        return value

    def constrain_for_rank(self, value, rank: int):
        return value

    def norm_kwargs(self, ctor) -> dict:
        return {}

    def linear_kwargs(self, ctor, spec=None) -> dict:
        return {}

    def make_linear_ctor(self, base_ctor, spec=None):
        return base_ctor

    def partitioned_init(self, init_fn, spec=None):
        return init_fn

    def as_type(self, target_cls):
        return self

    def _resolve_norm_spec(self, mesh=None):
        return None

    # -- Mixin method stubs (all return None) ------------------------------
    def linear_spec(self, mesh=None):
        return None

    def mha_spec(self, mesh=None):
        return None

    def transformer_hidden_spec(self, mesh=None):
        return None

    def transformer_input_spec(self, mesh=None):
        return None

    def transformer_mlp_column_spec(self, mesh=None):
        return None

    def transformer_mlp_row_spec(self, mesh=None):
        return None

    def spatial_activation_spec(self, mesh=None):
        return None

    def norm_spec(self, mesh=None):
        return None

    # Singleton-friendly repr
    def __repr__(self) -> str:
        return "NoopShardingCfg()"


# Module-level singleton — used by resolve_or_noop when no sharding is active.
_NOOP = NoopShardingCfg()


# ---------------------------------------------------------------------------
# MLP-specific sharding helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLPShardingSpec:
    sharding_cfg: ShardingCfg | None = None
    per_layer: Sequence[LinearShardingSpec | None] | LinearShardingSpec | None = None
    default: LinearShardingSpec | None = None

    def normalize(self, num_layers: int):
        """Resolve per-layer specs for an MLP with *num_layers* linear layers.

        Returns ``(mesh, default_spec, per_layer_specs)`` where
        *per_layer_specs* is a list of length *num_layers* or ``None``.
        """
        cfg = self.sharding_cfg or ShardingCfg.resolve(None)
        mesh = cfg.resolved_mesh() if cfg is not None else None
        default_spec = self.default
        if (
            default_spec is None
            and cfg is not None
            and isinstance(cfg, LinearShardingMixin)
        ):
            default_spec = cfg.linear_spec(mesh)
        elif default_spec is None and cfg is not None:
            # Wrap in LinearShardingCfg to get linear_spec
            linear_cfg = LinearShardingCfg(
                mesh=cfg.mesh,
                data_axis=cfg.data_axis,
                model_axis=cfg.model_axis,
            )
            default_spec = linear_cfg.linear_spec(mesh)
        per_layer = self.per_layer

        if per_layer is None:
            per_layer_specs = (
                [default_spec for _ in range(num_layers)]
                if default_spec is not None
                else None
            )
        elif isinstance(per_layer, LinearShardingSpec):
            per_layer_specs = [per_layer for _ in range(num_layers)]
        else:
            if len(per_layer) != num_layers:
                raise ValueError(
                    "sharding_cfg per_layer must have length "
                    f"{num_layers}, got {len(per_layer)}"
                )
            per_layer_specs = [
                spec if spec is not None else default_spec for spec in per_layer
            ]
        return mesh, default_spec, per_layer_specs


# ---------------------------------------------------------------------------
# Public helpers consumed by layer/net modules
# ---------------------------------------------------------------------------


def normalize_mlp_sharding(
    sharding_cfg: ShardingCfg | MLPShardingSpec | None,
    num_layers: int,
):
    """Normalize MLP sharding config to (mesh, default_spec, per_layer_specs).

    Accepts a ``ShardingCfg``, ``MLPShardingSpec``, or ``None``.
    """
    if sharding_cfg is None:
        return MLPShardingSpec().normalize(num_layers)
    if isinstance(sharding_cfg, MLPShardingSpec):
        return sharding_cfg.normalize(num_layers)
    if isinstance(sharding_cfg, ShardingCfg):
        return MLPShardingSpec(sharding_cfg=sharding_cfg).normalize(num_layers)
    raise TypeError(
        "sharding_cfg must be a ShardingCfg or MLPShardingSpec, "
        f"got {type(sharding_cfg)}"
    )


def filter_sharding_kwargs(ctor, **kwargs):
    """Filter *kwargs* to only those accepted by *ctor*."""
    return _filter_constructor_kwargs(ctor, **kwargs)


def activation_spec_for_rank(
    cfg: ShardingCfg | None,
    mesh: Mesh | None,
    rank: int,
) -> PartitionSpec | None:
    """Return a default activation PartitionSpec for the given tensor rank.

    Uses ``linear_spec`` for rank-2 and ``mha_spec`` for rank-3.
    """
    if cfg is None or mesh is None:
        return None
    # Wrap in LinearShardingCfg if needed to access linear_spec/mha_spec
    if isinstance(cfg, LinearShardingMixin):
        linear_cfg = cfg
    else:
        linear_cfg = LinearShardingCfg(
            mesh=cfg.mesh, data_axis=cfg.data_axis, model_axis=cfg.model_axis
        )
    if rank == 2:
        spec = linear_cfg.linear_spec(mesh)
        return spec.activation if spec is not None else None
    if rank == 3:
        spec = linear_cfg.mha_spec(mesh)
        return spec.activation if spec is not None else None
    return None

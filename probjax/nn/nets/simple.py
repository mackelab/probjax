from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.layers.fuse import AffineFuse
from probjax.nn.layers.masked import MaskedLinear
from probjax.nn.sharding import (
    LinearShardingSpec,
    MLPShardingSpec,
    linear_sharding_kwargs,
    make_sharded_linear_ctor,
    mesh_context,
    normalize_mlp_sharding,
)
from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
)
from probjax.utils.typing import (
    Array,
    ArrayLike,
    DTypeLike,
    ModuleLike,
    ModuleLikeType,
    PrecisionLike,
    PyTree,
)


class Sequential(nnx.Module):
    def __init__(
        self,
        *layers,
        sharding: jax.sharding.Mesh | None = None,
    ):
        """Sequential module.

        Args:
            layers (nnx.Module): List of layers.
        """
        self._mesh = sharding
        self.layers = nnx.List(layers)

    def __call__(self, x, *args, **kwargs) -> Array:
        with mesh_context(self._mesh):
            for layer in self.layers:
                x = layer(x, *args, **kwargs)
            return x


class MLP(nnx.Module):
    """Multi-layer perceptron (MLP) module with configurable layers and activation."""

    def __init__(
        self,
        feature_dims: Sequence[int],
        *,
        activation=jax.nn.gelu,
        activate_final: bool = False,
        context_dim: Optional[int] = None,
        # Accept alias used elsewhere in the codebase
        context_features: Optional[int] = None,
        precision: PrecisionLike | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        norm_cls: ModuleLikeType | None = None,
        linear_cls: ModuleLikeType | Sequence[ModuleLikeType] = nnx.Linear,
        context_fuse_cls: ModuleLikeType = AffineFuse,
        sharding: jax.sharding.Mesh | MLPShardingSpec | None = None,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        """Initialize MLP module.

        Args:
            dims: Sequence of layer dimensions. Length must be >= 2.
            rngs: Random number generators.
            linear: Linear layer module to use. Defaults to nnx.Linear.
            norm: Optional normalization layer. If provided, applied after
                each hidden layer (not output layer).
            activation: Activation function. Defaults to GELU.
            activate_final: Whether to apply activation to final layer output.
                Defaults to False.
            precision: Computation precision.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            preferred_element_dtype: Preferred element dtype.
            **kwargs: Additional arguments passed to linear layers.

        Raises:
            ValueError: If dims has fewer than 2 elements or contains
                non-positive values.
        """
        if len(feature_dims) < 2:
            raise ValueError(
                f"dims must have at least 2 elements, got {len(feature_dims)}"
            )
        if any(dim <= 0 for dim in feature_dims):
            raise ValueError(f"All dimensions must be positive, got {feature_dims}")

        self.feature_dims = feature_dims
        # Prefer explicit context_dim, fallback to alias if provided
        self.context_dim = context_dim if context_dim is not None else context_features
        self._mesh, _, per_layer_sharding = normalize_mlp_sharding(
            sharding, len(feature_dims) - 1
        )
        if per_layer_sharding is not None:
            self._activation_shardings = [
                spec.activation if spec is not None else None
                for spec in per_layer_sharding
            ]
        else:
            self._activation_shardings = None

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        num_layers = len(feature_dims) - 1
        if isinstance(linear_cls, Sequence) and not isinstance(linear_cls, type):
            if len(linear_cls) != num_layers:
                raise ValueError(
                    f"linear_cls sequence must have length {num_layers}, got {len(linear_cls)}"
                )
            base_linears = [
                partial(
                    lcls,
                    rngs=rngs,
                    **filter_precision_kwargs(lcls, **precision_kwargs),
                    **kwargs,
                )
                for lcls in linear_cls
            ]
        else:
            filtered = filter_precision_kwargs(linear_cls, **precision_kwargs)
            base_ctor = partial(linear_cls, rngs=rngs, **filtered, **kwargs)
            base_linears = [base_ctor for _ in range(num_layers)]

        layers = []
        norm_layers = []
        context_fuses = []
        _ctx_dim = self.context_dim
        with mesh_context(self._mesh):
            for i in range(num_layers):
                ctor = base_linears[i]
                if per_layer_sharding is not None:
                    sharding_kwargs = linear_sharding_kwargs(
                        ctor, per_layer_sharding[i]
                    )
                    if sharding_kwargs:
                        ctor = make_sharded_linear_ctor(ctor, sharding_kwargs)
                layers.append(ctor(feature_dims[i], feature_dims[i + 1]))

                if norm_cls is not None and i < num_layers - 1:
                    norm_layers.append(norm_cls(feature_dims[i + 1], rngs=rngs))
                if _ctx_dim is not None:
                    context_fuses.append(
                        context_fuse_cls(feature_dims[i + 1], _ctx_dim, rngs=rngs)
                    )

        self.layers = nnx.List(layers)
        self.norm_layers = nnx.List(norm_layers) if norm_cls is not None else None
        self.context_fuses = nnx.List(context_fuses) if _ctx_dim is not None else None
        self.activation = activation
        self.activate_final = activate_final

    def __call__(self, x: Array, context: Array | None = None) -> Array:
        """Forward pass through the MLP.

        Args:
            x: Input array of shape [..., input_dim].

        Returns:
            Output array of shape [..., output_dim].
        """
        with mesh_context(self._mesh):
            h = self.layers[0](x)
            h = self.activation(h)
            if self._activation_shardings is not None:
                spec = self._activation_shardings[0]
                if spec is not None:
                    h = jax.lax.with_sharding_constraint(h, spec)
            for i in range(1, len(self.layers) - 1):
                h = self.layers[i](h)
                if self.norm_layers is not None:
                    h = self.norm_layers[i - 1](h)
                h = self.activation(h)
                if self.context_fuses is not None:
                    h = self.context_fuses[i - 1](h, context)
                if self._activation_shardings is not None:
                    spec = self._activation_shardings[i]
                    if spec is not None:
                        h = jax.lax.with_sharding_constraint(h, spec)

            out = self.layers[-1](h) if len(self.layers) > 1 else h

            if self.activate_final:
                out = self.activation(out)
                if self._activation_shardings is not None:
                    spec = self._activation_shardings[-1]
                    if spec is not None:
                        out = jax.lax.with_sharding_constraint(out, spec)
            return out


class MaskedMLP(MLP):
    def __init__(
        self,
        dims: Sequence[int],
        masks: Sequence[ArrayLike],
        rngs: nnx.Rngs,
        **kwargs,
    ):
        if len(masks) != len(dims) - 1:
            raise ValueError(f"Expected {len(dims) - 1} masks, got {len(masks)}")
        # Build per-layer masked linear constructors via partial so MLP can handle them
        masked_linears = [
            partial(MaskedLinear, mask=masks[i]) for i in range(len(dims) - 1)
        ]
        # Delegate to MLP with a sequence of constructors
        super().__init__(
            feature_dims=dims, linear_cls=masked_linears, rngs=rngs, **kwargs
        )


class ResNet(nnx.Module):
    """Residual neural network with optional context conditioning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_dim: int = 50,
        num_hidden_layers: int = 2,
        context_dim: Optional[int] = None,
        activation=jax.nn.gelu,
        activate_final: bool = False,
        precision: PrecisionLike | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        context_fuse_cls: ModuleLikeType = AffineFuse,
        norm_cls: ModuleLikeType | None = None,
        linear_cls: ModuleLikeType = nnx.Linear,
        sharding: jax.sharding.Mesh | MLPShardingSpec | None = None,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        """Initialize ResNet module.

        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            hidden_dim: Hidden layer dimension. Defaults to 50.
            num_hidden_layers: Number of hidden layers. Defaults to 2.
            context_dim: Optional context dimension for conditioning.
            activation: Activation function. Defaults to GELU.
            activate_final: Whether to apply activation to final layer output.
                Defaults to False.
            precision: Computation precision.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            preferred_element_dtype: Preferred element dtype.
            linear_cls: Linear layer module to use. Defaults to nnx.Linear.
            context_fuse_cls: Context fusion module. Defaults to AffineFuse.
            norm_cls: Optional normalization layer.
            rngs: Random number generators.
            **kwargs: Additional arguments passed to linear layers.

        Raises:
            ValueError: If input/output dimensions or hidden dimensions
                are not positive.
        """
        if in_features <= 0:
            raise ValueError(f"in_dim must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_dim must be positive, got {out_features}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_hidden_layers < 0:
            raise ValueError(
                f"num_hidden_layers must be non-negative, got {num_hidden_layers}"
            )

        self.in_dim = in_features
        self.out_dim = out_features
        if context_dim is not None and context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        self.context_dim = context_dim
        num_layers = num_hidden_layers + 2
        self._mesh, _, per_layer_sharding = normalize_mlp_sharding(
            sharding, num_layers
        )
        if per_layer_sharding is not None:
            self._activation_shardings = [
                spec.activation if spec is not None else None
                for spec in per_layer_sharding
            ]
        else:
            self._activation_shardings = None

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(linear_cls, **precision_kwargs)
        base_ctor = partial(linear_cls, rngs=rngs, **precision_kwargs, **kwargs)

        hidden_layers = []
        norm_layers = []
        context_layers = []
        with mesh_context(self._mesh):
            for i in range(num_layers):
                ctor = base_ctor
                if per_layer_sharding is not None:
                    sharding_kwargs = linear_sharding_kwargs(
                        ctor, per_layer_sharding[i]
                    )
                    if sharding_kwargs:
                        ctor = make_sharded_linear_ctor(ctor, sharding_kwargs)

                if i == 0:
                    self.in_layer = ctor(in_features, hidden_dim)
                elif i == num_layers - 1:
                    self.out_layer = ctor(hidden_dim, out_features)
                else:
                    hidden_layers.append(ctor(hidden_dim, hidden_dim))

                    if norm_cls is not None:
                        norm_layers.append(norm_cls(hidden_dim, rngs=rngs))
                    if context_dim is not None:
                        context_layers.append(
                            context_fuse_cls(hidden_dim, context_dim, rngs=rngs)
                        )

        self.hidden_layers = nnx.List(hidden_layers)
        self.norm_layers = nnx.List(norm_layers) if norm_cls is not None else None
        self.activation = activation
        self.activate_final = activate_final

        self.context_layers = (
            nnx.List(context_layers) if context_dim is not None else None
        )

    def __call__(self, x: ArrayLike, context: Optional[ArrayLike] = None) -> Array:
        """Forward pass through the ResNet.

        Args:
            x: Input array of shape [..., in_dim].
            context: Optional context array of shape [..., context_dim].

        Returns:
            Output array of shape [..., out_dim].

        Raises:
            ValueError: If context is expected but not provided, or vice versa.
        """
        if self.context_dim is not None and context is None:
            raise ValueError("context is required when context_dim is specified")
        if self.context_dim is None and context is not None:
            raise ValueError("context provided but context_dim is None")

        with mesh_context(self._mesh):
            h = self.in_layer(x)
            h = self.activation(h)
            if self._activation_shardings is not None:
                spec = self._activation_shardings[0]
                if spec is not None:
                    h = jax.lax.with_sharding_constraint(h, spec)
            for i in range(len(self.hidden_layers)):
                h_old = h
                if self.norm_layers is not None:
                    h = self.norm_layers[i](h)
                h = self.hidden_layers[i](h)
                h = self.activation(h)
                if self.context_layers is not None:
                    h = self.context_layers[i](h, context)
                if self._activation_shardings is not None:
                    spec = self._activation_shardings[i + 1]
                    if spec is not None:
                        h = jax.lax.with_sharding_constraint(h, spec)

                h = h + h_old

            out = self.out_layer(h) if len(self.hidden_layers) > 0 else h

            if self.activate_final:
                out = self.activation(out)
                if self._activation_shardings is not None:
                    spec = self._activation_shardings[-1]
                    if spec is not None:
                        out = jax.lax.with_sharding_constraint(out, spec)
            return out


class DeepSet(nnx.Module):
    """Deep Sets module for permutation-invariant functions.

    Implements the Deep Sets architecture that processes sets of elements
    in a permutation-invariant manner using the formula:
    f(X) = ρ(Σ φ(x_i)) where X = {x_1, ..., x_n}
    """

    def __init__(
        self,
        phi: ModuleLike,
        rho: ModuleLike,
        *,
        reduction: Callable = jnp.sum,
        axis: tuple[int] | int = -2,
        dropout_rate: float = 0.0,
        sharding: jax.sharding.Mesh | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize the DeepSets module.

        The only requirement is that both `phi` and `rho` accept the input
        arrays as their first argument.

        Args:
            phi: Neural network module or callable function that processes
                individual elements of the input set.
            rho: Neural network module or callable function that processes
                the aggregated output of phi.
            reduction: Reduction function to aggregate the outputs of phi.
                Defaults to jnp.sum.
            axis: Axis along which to apply the reduction. Defaults to -2.
            dropout_rate: Dropout rate applied after phi and before aggregation.
                Must be between 0.0 and 1.0. Defaults to 0.0 (no dropout).
            rngs: Random number generators.

        Raises:
            ValueError: If phi or rho are None, or if dropout_rate is invalid.
        """
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}"
            )

        self.phi = phi
        self.rho = rho
        self.reduction = reduction
        self.axis = axis
        self.dropout_rate = dropout_rate
        self._mesh = sharding

        if dropout_rate > 0.0:
            self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        else:
            self.dropout = None

    def __call__(
        self,
        x: PyTree[ArrayLike],
        *,
        deterministic: bool = True,
        phi_args: Optional[tuple] = None,
        rho_args: Optional[tuple] = None,
        phi_kwargs: Optional[dict] = None,
        rho_kwargs: Optional[dict] = None,
    ) -> Array:
        """Apply the Deep Sets computation.

        Args:
            x: Input array to be processed by phi.
            phi_kwargs: Additional keyword arguments to pass to the phi module.
                Defaults to None.
            rho_kwargs: Additional keyword arguments to pass to the rho module.
                Defaults to None.

        Returns:
            Output array after applying phi, dropout (if enabled), reduction, and rho.
        """
        phi_args = phi_args if phi_args is not None else ()
        rho_args = rho_args if rho_args is not None else ()
        phi_kwargs = phi_kwargs if phi_kwargs is not None else {}
        rho_kwargs = rho_kwargs if rho_kwargs is not None else {}
        with mesh_context(self._mesh):
            # Apply phi to each element
            phi_x = self.phi(x, *phi_args, **phi_kwargs)

            # Apply dropout if enabled
            if self.dropout is not None:
                phi_x = self.dropout(phi_x, deterministic=deterministic)

            # Aggregate
            h = self.reduction(phi_x, axis=self.axis)

            # Apply rho
            return self.rho(h, **(rho_kwargs if rho_kwargs is not None else {}))

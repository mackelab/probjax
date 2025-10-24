from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.layers.fuse import AffineFuse, ContextFuse
from probjax.nn.layers.masked import MaskedLinear
from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
)
from probjax.utils.typing import Array, ArrayLike, ModuleLikeType, PyTree


class Sequential(nnx.Module):
    def __init__(self, *layers):
        """Sequential module.

        Args:
            layers (nnx.Module): List of layers.
        """
        self.layers = nnx.List(layers)

    def __call__(self, x, *args, **kwargs) -> Array:
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
        precision: Optional[jax.lax.Precision] = None,
        dtype: Optional[jax.numpy.dtype] = None,
        param_dtype: Optional[jax.numpy.dtype] = None,
        preferred_element_dtype: Optional[jax.numpy.dtype] = None,
        norm_cls: type[nnx.Module] | None = None,
        # Allow a single linear class or a per-layer sequence
        linear_cls: ModuleLikeType | Sequence[ModuleLikeType] = nnx.Linear,
        context_fuse_cls: type[ContextFuse] = AffineFuse,
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
        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_dtype
        )
        # Build per-layer linear constructors (support sequence of linear classes)
        num_layers = len(feature_dims) - 1
        if isinstance(linear_cls, Sequence) and not isinstance(linear_cls, type):
            if len(linear_cls) != num_layers:
                raise ValueError(
                    f"linear_cls sequence must have length {num_layers}, got {len(linear_cls)}"
                )
            linears = [
                partial(
                    lcls,
                    rngs=rngs,
                    **filter_precision_kwargs(lcls, **precision_kwargs),
                    **kwargs,
                )
                for lcls in linear_cls
            ]
        else:
            # Single class applied to all layers
            filtered = filter_precision_kwargs(linear_cls, **precision_kwargs)
            ctor = partial(linear_cls, rngs=rngs, **filtered, **kwargs)
            linears = [ctor for _ in range(num_layers)]

        self.layers = nnx.List([
            linears[i](
                feature_dims[i],
                feature_dims[i + 1],
            )
            for i in range(num_layers)
        ])
        if norm_cls is not None:
            self.norm_layers = nnx.List([
                norm_cls(feature_dims[i + 1], rngs=rngs)
                for i in range(len(feature_dims) - 2)
            ])
        else:
            self.norm_layers = None
        # Build context fuses if an effective context dimension is provided
        _ctx_dim = self.context_dim
        if _ctx_dim is not None:
            self.context_fuses = nnx.List([
                context_fuse_cls(feature_dims[i + 1], _ctx_dim, rngs=rngs)
                for i in range(len(feature_dims) - 1)
            ])
        else:
            self.context_fuses = None
        self.activation = activation
        self.activate_final = activate_final

    def __call__(self, x: ArrayLike, context: ArrayLike | None = None) -> Array:
        """Forward pass through the MLP.

        Args:
            x: Input array of shape [..., input_dim].

        Returns:
            Output array of shape [..., output_dim].
        """
        h = self.layers[0](x)
        h = self.activation(h)
        for i in range(1, len(self.layers) - 1):
            h = self.layers[i](h)
            if self.norm_layers is not None:
                h = self.norm_layers[i - 1](h)
            h = self.activation(h)
            if self.context_fuses is not None:
                h = self.context_fuses[i - 1](h, context)

        out = self.layers[-1](h) if len(self.layers) > 1 else h

        if self.activate_final:
            out = self.activation(out)
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
        precision: Optional[jax.lax.Precision] = None,
        dtype: Optional[jax.numpy.dtype] = None,
        param_dtype: Optional[jax.numpy.dtype] = None,
        preferred_element_dtype: Optional[jax.numpy.dtype] = None,
        context_fuse_cls: type[ContextFuse] = AffineFuse,
        norm_cls: Optional[nnx.LayerNorm | nnx.BatchNorm | nnx.Module] = None,
        linear_cls: nnx.Linear | nnx.LoRALinear | nnx.Module = nnx.Linear,
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
        self.context_dim = context_dim

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_dtype
        )
        precision_kwargs = filter_precision_kwargs(linear_cls, **precision_kwargs)
        _linear = partial(linear_cls, rngs=rngs, **precision_kwargs, **kwargs)
        self.in_layer = _linear(
            in_features,
            hidden_dim,
        )
        self.out_layer = _linear(
            hidden_dim,
            out_features,
        )
        self.hidden_layers = nnx.List([
            _linear(
                hidden_dim,
                hidden_dim,
            )
            for _ in range(num_hidden_layers)
        ])
        if norm_cls is not None:
            self.norm_layers = nnx.List([
                norm_cls(hidden_dim, rngs=rngs) for _ in range(num_hidden_layers)
            ])
        else:
            self.norm_layers = None
        self.activation = activation
        self.activate_final = activate_final

        if context_dim is not None:
            if context_dim <= 0:
                raise ValueError(f"context_dim must be positive, got {context_dim}")
            self.context_layers = nnx.List([
                context_fuse_cls(hidden_dim, context_dim, rngs=rngs)
                for _ in range(num_hidden_layers)
            ])
        else:
            self.context_layers = None

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

        h = self.in_layer(x)
        h = self.activation(h)
        for i in range(len(self.hidden_layers)):
            h_old = h
            if self.norm_layers is not None:
                h = self.norm_layers[i](h)
            h = self.hidden_layers[i](h)
            h = self.activation(h)
            if self.context_layers is not None:
                h = self.context_layers[i](h, context)

            h = h + h_old

        out = self.out_layer(h) if len(self.hidden_layers) > 0 else h

        if self.activate_final:
            out = self.activation(out)
        return out


class DeepSet(nnx.Module):
    """Deep Sets module for permutation-invariant functions.

    Implements the Deep Sets architecture that processes sets of elements
    in a permutation-invariant manner using the formula:
    f(X) = ρ(Σ φ(x_i)) where X = {x_1, ..., x_n}
    """

    def __init__(
        self,
        phi: nnx.Module,
        rho: nnx.Module,
        *,
        reduction: Callable = jnp.sum,
        axis: tuple[int] | int = -2,
        dropout_rate: float = 0.0,
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
        # Apply phi to each element
        phi_x = self.phi(x, *phi_args, **phi_kwargs)

        # Apply dropout if enabled
        if self.dropout is not None:
            phi_x = self.dropout(phi_x, deterministic=deterministic)

        # Aggregate
        h = self.reduction(phi_x, axis=self.axis)

        # Apply rho
        return self.rho(h, **(rho_kwargs if rho_kwargs is not None else {}))

from functools import partial
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
)
from probjax.nn.layers.fuse import (
    AffineFuse,
    AdditiveFuse,
    ConcatFuse,
)
from probjax.utils.typing import (
    Array,
    ArrayLike,
    PyTree,
)


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
        precision: Optional[jax.lax.Precision] = None,
        dtype: Optional[jax.numpy.dtype] = None,
        param_dtype: Optional[jax.numpy.dtype] = None,
        preferred_element_dtype: Optional[jax.numpy.dtype] = None,
        norm_cls: type[nnx.Module] | None = None,
        linear_cls: nnx.Linear | nnx.LoRALinear | nnx.Module = nnx.Linear,
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

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_dtype
        )
        precision_kwargs = filter_precision_kwargs(linear_cls, **precision_kwargs)
        _linear = partial(linear_cls, rngs=rngs, **precision_kwargs, **kwargs)
        self.layers = nnx.List([
            _linear(
                feature_dims[i],
                feature_dims[i + 1],
            )
            for i in range(len(feature_dims) - 1)
        ])
        self.norm = norm_cls
        if norm_cls is not None:
            self.norm_layers = nnx.List([
                norm_cls(feature_dims[i + 1], rngs=rngs)
                for i in range(len(feature_dims) - 2)
            ])
        self.activation = activation
        self.activate_final = activate_final

    def __call__(self, x: ArrayLike) -> Array:
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
            if self.norm is not None:
                h = self.norm_layers[i - 1](h)
            h = self.activation(h)

        out = self.layers[-1](h) if len(self.layers) > 1 else h

        if self.activate_final:
            out = self.activation(out)
        return out


class ResNet(nnx.Module):
    """Residual neural network with optional context conditioning."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rngs: nnx.Rngs,
        *,
        hidden_dim: int = 50,
        num_hidden_layers: int = 2,
        context_dim: Optional[int] = None,
        linear: nnx.Linear | nnx.LoRALinear | nnx.Module = nnx.Linear,
        context_fuse: type[AffineFuse]
        | type[AdditiveFuse]
        | type[nnx.Module] = AffineFuse,
        norm: Optional[nnx.LayerNorm | nnx.BatchNorm | nnx.Module] = None,
        activation=jax.nn.gelu,
        activate_final: bool = False,
        precision: Optional[jax.lax.Precision] = None,
        dtype: Optional[jax.numpy.dtype] = None,
        param_dtype: Optional[jax.numpy.dtype] = None,
        preferred_element_dtype: Optional[jax.numpy.dtype] = None,
        **kwargs,
    ):
        """Initialize ResNet module.

        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            rngs: Random number generators.
            hidden_dim: Hidden layer dimension. Defaults to 50.
            num_hidden_layers: Number of hidden layers. Defaults to 2.
            context_dim: Optional context dimension for conditioning.
            linear: Linear layer module to use. Defaults to nnx.Linear.
            context_fuse: Context fusion module. Defaults to AffineFuse.
            norm: Optional normalization layer.
            activation: Activation function. Defaults to GELU.
            activate_final: Whether to apply activation to final layer output.
                Defaults to False.
            precision: Computation precision.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            preferred_element_dtype: Preferred element dtype.
            **kwargs: Additional arguments passed to linear layers.

        Raises:
            ValueError: If input/output dimensions or hidden dimensions
                are not positive.
        """
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_hidden_layers < 0:
            raise ValueError(
                f"num_hidden_layers must be non-negative, got {num_hidden_layers}"
            )

        self.context_dim = context_dim

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_dtype
        )
        precision_kwargs = filter_precision_kwargs(linear, **precision_kwargs)
        _linear = partial(linear, rngs=rngs, **precision_kwargs, **kwargs)
        self.in_layer = _linear(
            in_dim,
            hidden_dim,
        )
        self.out_layer = _linear(
            hidden_dim,
            out_dim,
        )
        self.hidden_layers = nnx.List([
            _linear(
                hidden_dim,
                hidden_dim,
            )
            for _ in range(num_hidden_layers)
        ])
        self.norm = norm
        if norm is not None:
            self.norm_layers = nnx.List([
                norm(hidden_dim, rngs=rngs) for _ in range(num_hidden_layers)
            ])
        self.activation = activation
        self.activate_final = activate_final

        if context_dim is not None:
            if context_dim <= 0:
                raise ValueError(f"context_dim must be positive, got {context_dim}")
            self.context_init = context_fuse(hidden_dim, context_dim, rngs=rngs)
            self.context_layers = nnx.List([
                context_fuse(hidden_dim, context_dim, rngs=rngs)
                for _ in range(num_hidden_layers)
            ])

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
        if context is not None:
            h = self.context_init(h, context)
        h = self.activation(h)
        for i in range(len(self.hidden_layers)):
            h_old = h
            h = self.hidden_layers[i](h)
            if self.norm is not None:
                h = self.norm_layers[i](h)
            h = self.activation(h)
            if context is not None:
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
        axis: int = -2,
        phi_kwargs: Optional[dict] = None,
        rho_kwargs: Optional[dict] = None,
        dropout_rate: float = 0.0,
        rngs: nnx.Rngs,
    ):
        """Initialize the DeepSets module.

        Args:
            phi: Neural network module or callable function that processes
                individual elements of the input set.
            rho: Neural network module or callable function that processes
                the aggregated output of phi.
            reduction: Reduction function to aggregate the outputs of phi.
                Defaults to jnp.sum.
            axis: Axis along which to apply the reduction. Defaults to -2.
            phi_kwargs: Additional keyword arguments to pass to the phi module
                or callable. Defaults to None.
            rho_kwargs: Additional keyword arguments to pass to the rho module
                or callable. Defaults to None.
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
        # Apply phi to each element
        phi_x = self.phi(x, **(phi_kwargs if phi_kwargs is not None else {}))

        # Apply dropout if enabled
        if self.dropout is not None:
            phi_x = self.dropout(phi_x, deterministic=deterministic)

        # Aggregate
        h = self.reduction(phi_x, axis=self.axis)

        # Apply rho
        return self.rho(h, **(rho_kwargs if rho_kwargs is not None else {}))

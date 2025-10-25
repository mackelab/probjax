from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
)
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, ModuleLikeType


class ContextFuse(nnx.Module):
    """Base class for fusion modules."""

    def __call__(self, x: Array, context: Array, *args, **kwargs) -> ArrayLike:
        del x, context, args, kwargs
        raise NotImplementedError("Fuse is an abstract base class.")


class AdditiveFuse(ContextFuse):
    """Additive fusion module for combining input and context."""

    def __init__(
        self,
        in_features: int,
        context_features: int,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        layer_cls: ModuleLikeType = nnx.Linear,
        rngs: nnx.Rngs,
    ):
        """Additive fusion module that applies linear transformation to context
        and adds it to the input.

        Args:
            in_features (int): Dimension of the input features.
            context_features (int): Dimension of the context features.
            dtype: Computation dtype (optional).
            param_dtype: Parameter dtype (optional).
            precision: Computation precision (optional).
            preferred_element_type: Preferred element type (optional).
            rngs (nnx.Rngs): Random number generators.

        Raises:
            ValueError: If in_features or context_features are not positive.
        """
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if context_features <= 0:
            raise ValueError("context_features must be positive")

        super().__init__()
        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(layer_cls, **precision_kwargs)

        self.linear = layer_cls(
            context_features,
            in_features,
            rngs=rngs,
            **precision_kwargs,
        )

    def __call__(self, x: Array, context: Array) -> Array:
        """Apply additive fusion to input and context.

        Args:
            x: Input array of shape [..., input_dim]
            context: Context array of shape [..., context_dim]

        Returns:
            Array with same shape as x, with transformed context added.
        """
        return x + self.linear(context)


def default_scale_activation(x: ArrayLike) -> ArrayLike:
    return x + 1.0  # For identity initialization


class AffineFuse(ContextFuse):
    """Affine fusion module that applies scale and bias transformations."""

    def __init__(
        self,
        in_features: int,
        context_features: int,
        *,
        scale_activation: Callable = default_scale_activation,
        use_bias: bool = False,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        layer_cls: ModuleLikeType = nnx.Linear,
        rngs: nnx.Rngs,
    ):
        """Affine fusion module that applies scale and bias to the input
        based on context.

        Args:
            in_features (int): Dimension of the input features.
            context_features (int): Dimension of the context features.
            scale_activation (Callable, optional): Activation function to
                apply to scale.
            use_bias (bool): Whether to use bias in linear layers.
                Defaults to False.
            dtype: Computation dtype (optional).
            param_dtype: Parameter dtype (optional).
            precision: Computation precision (optional).
            preferred_element_type: Preferred element type (optional).
            rngs (nnx.Rngs): Random number generators.

        Raises:
            ValueError: If in_features or context_features are not positive.
        """
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if context_features <= 0:
            raise ValueError("context_features must be positive")

        super().__init__()
        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(layer_cls, **precision_kwargs)

        self.linear_scale = layer_cls(
            context_features,
            in_features,
            use_bias=use_bias,
            kernel_init=nnx.initializers.zeros,
            rngs=rngs,
            **precision_kwargs,
        )
        self.linear_bias = layer_cls(
            context_features,
            in_features,
            use_bias=use_bias,
            kernel_init=nnx.initializers.zeros,
            rngs=rngs,
            **precision_kwargs,
        )
        self.scale_activation = scale_activation

    def __call__(self, x: Array, context: Array) -> Array:
        """Apply affine fusion to input and context.

        Args:
            x: Input array of shape [..., input_dim]
            context: Context array of shape [..., context_dim]

        Returns:
            Array with same shape as x, with affine transformation applied.
        """
        scale = self.linear_scale(context)
        scale = self.scale_activation(scale)
        bias = self.linear_bias(context)
        return x * scale + bias


class ConcatFuse(ContextFuse):
    """Concatenation fusion module that combines input and context features."""

    def __init__(
        self,
        in_features: int,
        context_features: int,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        layer_cls: ModuleLikeType = nnx.Linear,
        rngs: nnx.Rngs,
    ):
        """Concatenation fusion module that linearly transforms context
        and concatenates it with the input.

        Args:
            in_features (int): Dimension of the input features.
            context_features (int): Dimension of the context features.
            dtype: Computation dtype (optional).
            param_dtype: Parameter dtype (optional).
            precision: Computation precision (optional).
            preferred_element_type: Preferred element type (optional).
            rngs (nnx.Rngs): Random number generators.

        Raises:
            ValueError: If in_features or context_features are not positive.
        """
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if context_features <= 0:
            raise ValueError("context_features must be positive")

        super().__init__()
        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(layer_cls, **precision_kwargs)

        self.ctx_layer = layer_cls(
            context_features,
            in_features,
            rngs=rngs,
            **precision_kwargs,
        )
        self.merge_layer = layer_cls(
            in_features + in_features,
            in_features,
            rngs=rngs,
            **precision_kwargs,
        )

    def __call__(self, x: Array, context: Array) -> Array:
        """Apply concatenation fusion to input and context.

        Args:
            x: Input array of shape [..., input_dim]
            context: Context array of shape [..., context_dim]

        Returns:
            Array of shape [..., input_dim + input_dim] with transformed context
            concatenated to the input.
        """
        x = jnp.asarray(x)
        context = jnp.asarray(context)
        context = self.ctx_layer(context)
        # Ensure same leading dimensions as x
        context = jnp.broadcast_to(context, x.shape[:-1] + (context.shape[-1],))  # type: ignore

        x_ctx = jnp.concatenate([x, context], axis=-1)
        x = self.merge_layer(x_ctx)
        return x


class GatedFuse(ContextFuse):
    """Gated fusion module that combines input and context features."""

    def __init__(
        self,
        in_features: int,
        context_features: int,
        *,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        mode: Literal["convex", "left", "right"] = "convex",
        layer_cls: ModuleLikeType = nnx.Linear,
        rngs: nnx.Rngs,
    ):
        """Gated fusion module that linearly transforms context
        and combines it with the input using a gating mechanism.

        Args:
            in_features (int): Dimension of the input features.
            context_features (int): Dimension of the context features.
            dtype: Computation dtype (optional).
            param_dtype: Parameter dtype (optional).
            precision: Computation precision (optional).
            preferred_element_type: Preferred element type (optional).
            rngs (nnx.Rngs): Random number generators.

        Raises:
            ValueError: If in_features or context_features are not positive.
        """
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if context_features <= 0:
            raise ValueError("context_features must be positive")

        super().__init__()

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        precision_kwargs = filter_precision_kwargs(layer_cls, **precision_kwargs)

        self.mode = mode
        self.gate_layer = layer_cls(
            context_features,
            in_features,
            rngs=rngs,
            **precision_kwargs,
        )

    def __call__(self, x: Array, y: Array, context: Array) -> Array:
        """Apply gated fusion to input and context.

        Args:
            x: Input array of shape [..., input_dim]
            context: Context array of shape [..., context_dim]

        Returns:
            Array of shape [..., input_dim] with gated combination of input
            and transformed context.
        """
        x = jnp.asarray(x)
        context = jnp.asarray(context)
        # Ensure same leading dimensions as x
        context = jnp.broadcast_to(context, x.shape[:-1] + (context.shape[-1],))
        gate = jax.nn.sigmoid(self.gate_layer(context))
        if self.mode == "left":
            return x * gate + y
        elif self.mode == "right":
            return x + y * gate
        else:  # convex
            return x * gate + y * (1 - gate)

from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.layers.reg import DropPath
from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
)
from probjax.utils.typing import (
    Array,
    ArrayLike,
    DTypeLike,
    ModuleLikeType,
    PrecisionLike,
)


def default_scale_activation(x: ArrayLike) -> ArrayLike:
    return x + 1.0  # For identity initialization


def identity(x: ArrayLike) -> ArrayLike:
    return x


class ContextFuse(nnx.Module):
    """Base class for fusion modules."""

    def __call__(self, x: Array, context: Array) -> Array: ...


class BinaryFuse(nnx.Module):
    """Base class for binary fusion modules."""

    def __call__(
        self,
        x: Array,
        y: Array,
        context: Array | None,
        *,
        deterministic: bool = True,
    ) -> Array: ...


class MLPConditioner(nnx.Module):
    """Two-layer MLP used as the default fusion projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int | None = None,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        rngs: nnx.Rngs,
    ):
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features <= 0:
            raise ValueError("out_features must be positive")

        super().__init__()
        hidden_features = hidden_features or max(in_features, out_features)
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive")

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )
        linear_kwargs = filter_precision_kwargs(nnx.Linear, **precision_kwargs)

        self.activation = activation
        self.hidden = nnx.Linear(
            in_features,
            hidden_features,
            rngs=rngs,
            **linear_kwargs,
        )
        self.proj = nnx.Linear(
            hidden_features,
            out_features,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
            **linear_kwargs,
        )

    def __call__(self, x: Array) -> Array:
        x = self.hidden(x)
        x = self.activation(x)
        return self.proj(x)


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
        layer_cls: ModuleLikeType = MLPConditioner,
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


class AffineFuse(ContextFuse):
    """Affine fusion module that applies scale and bias transformations."""

    def __init__(
        self,
        in_features: int,
        context_features: int,
        *,
        scale_activation: Callable = default_scale_activation,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        layer_cls: ModuleLikeType = MLPConditioner,
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

        self.linear_scale_bias = layer_cls(
            context_features,
            2 * in_features,
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
        scale_bias = self.linear_scale_bias(context)
        scale, bias = jnp.split(scale_bias, 2, axis=-1)
        scale = self.scale_activation(scale)
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
        layer_cls: ModuleLikeType = MLPConditioner,
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
        context = self.ctx_layer(context)
        # Ensure same leading dimensions as x
        context = jnp.broadcast_to(context, x.shape[:-1] + (context.shape[-1],))  # type: ignore

        x_ctx = jnp.concatenate([x, context], axis=-1)
        x = self.merge_layer(x_ctx)
        return x


class AdditiveBinaryFuse(BinaryFuse):
    def __init__(self, *, drop_path_rate: float = 0.0, rngs: nnx.Rngs):
        """Additive binary fusion module that adds two inputs."""
        super().__init__()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_rate=drop_path_rate, rngs=rngs)
        else:
            self.drop_path = None

    def __call__(
        self,
        x: Array,
        y: Array,
        context: Array | None,
        *,
        deterministic: bool = True,
    ) -> Array:
        del context
        if self.drop_path is not None:
            y = self.drop_path(y, deterministic=deterministic)
        return x + y


class GatedFuse(BinaryFuse):
    """Gated fusion module that combines input and context features."""

    def __init__(
        self,
        in_features: int,
        context_features: int,
        *,
        gate_activation: Callable = jax.nn.sigmoid,
        drop_path_rate: float = 0.0,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        mode: Literal["convex", "left", "right"] = "right",
        layer_cls: ModuleLikeType = MLPConditioner,
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
        self.gate_activation = gate_activation

        if self.mode not in {"convex", "left", "right"}:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be 'convex', 'left', or 'right'."
            )

        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_rate=drop_path_rate, rngs=rngs)
        else:
            self.drop_path = None

    def __call__(
        self,
        x: Array,
        y: Array,
        context: Array | None,
        *,
        deterministic: bool = True,
    ) -> Array:
        """Apply gated fusion to input and context.

        Args:
            x: Input array of shape [..., input_dim]
            context: Context array of shape [..., context_dim]

        Returns:
            Array of shape [..., input_dim] with gated combination of input
            and transformed context.
        """
        if context is None:
            raise ValueError("Context must be provided for GatedFuse.")
        if self.drop_path is not None:
            y = self.drop_path(y, deterministic=deterministic)
        # Ensure same leading dimensions as x
        context = jnp.broadcast_to(context, x.shape[:-1] + (context.shape[-1],))
        gate = self.gate_activation(self.gate_layer(context))
        if self.mode == "left":
            return (x * gate + y) / jnp.sqrt(1 + gate * gate)
        elif self.mode == "right":
            return (x + y * gate) / jnp.sqrt(1 + gate * gate)
        else:  # convex
            return x * gate + y * (1 - gate)

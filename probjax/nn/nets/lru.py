from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.layers.lru import LRUBlock
from probjax.nn.nets.simple import MLP
from probjax.nn.utils import filter_precision_kwargs, get_active_precision_kwargs
from probjax.utils.typing import Array, ArrayLike, DTypeLike, PrecisionLike, ModuleLikeType


class LRUModel(nnx.Module):
    """Linear Recurrent Unit (LRU) model with optional bidirectional processing.

    This model stacks LRU blocks with MLP layers and residual connections,
    optionally alternating between forward and backward processing for
    bidirectional modeling. Each LRU and MLP block is preceded by layer
    normalization following the pre-norm architecture pattern.
    """

    input_dim: int  # Input dimension
    model_dim: int  # Model hidden dimension
    output_dim: int  # Output dimension
    num_layers: int  # Number of LRU layers
    bidirectional: bool  # Whether to use bidirectional processing
    dropout_rate: float | None  # Dropout rate

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        output_dim: int,
        num_layers: int,
        *,
        bidirectional: bool = True,
        dropout_rate: Optional[float] = None,
        mlp_widening_factor: int = 2,
        activation: Callable = jax.nn.gelu,
        norm_cls: type[nnx.Module] = nnx.LayerNorm,
        mlp_cls: ModuleLikeType = MLP,
        initializer: Optional[nnx.initializers.Initializer] = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize an LRU model.

        Args:
            input_dim: Input dimension.
            model_dim: Model hidden dimension.
            output_dim: Output dimension.
            num_layers: Number of LRU layers to stack.
            bidirectional: Whether to use bidirectional processing by alternating
                forward and backward passes. Defaults to True.
            dropout_rate: Dropout rate. If None, no dropout is applied.
                Defaults to None.
            mlp_widening_factor: Factor by which to widen the MLP hidden dimension.
                Defaults to 2.
            activation: Activation function. Defaults to jax.nn.gelu.
            norm_cls: Normalization layer class. Defaults to nnx.LayerNorm.
            mlp_cls: MLP class to use. Defaults to MLP.
            initializer: Weight initializer. If None, uses default initialization.
                Defaults to None.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            precision: Computation precision.
            preferred_element_type: Preferred element type.
            rngs: Random number generators.

            Raises:
                ValueError: If any dimension is not positive or if num_layers is
                    negative.
        """
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {model_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        if dropout_rate is not None and not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}"
            )

        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        # Precision and dtype settings
        precision_kwargs = get_active_precision_kwargs(
            dtype,
            precision,
            param_dtype,
            preferred_element_type,
        )

        # Initialize linear layers with precision kwargs
        linear_kwargs = filter_precision_kwargs(nnx.Linear, **precision_kwargs)
        if initializer is not None:
            linear_kwargs['kernel_init'] = initializer

        self.in_layer = nnx.Linear(input_dim, model_dim, rngs=rngs, **linear_kwargs)
        self.out_layer = nnx.Linear(model_dim, output_dim, rngs=rngs, **linear_kwargs)

        # Layer norms for LRU and MLP blocks
        self.layer_norms_lru = nnx.List([
            norm_cls(model_dim, rngs=rngs) for _ in range(num_layers)
        ])
        self.layer_norms_mlp = nnx.List([
            norm_cls(model_dim, rngs=rngs) for _ in range(num_layers)
        ])

        # Final output layer norm
        self.out_layer_norm = norm_cls(model_dim, rngs=rngs)

        # LRU layers
        self.lru_layers = nnx.List([
            LRUBlock(
                model_dim,
                rngs=rngs,
                dropout=dropout_rate,
                norm=norm_cls,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

        # MLP layers for processing between LRU blocks
        mlp_dims = [model_dim, mlp_widening_factor * model_dim, model_dim]
        mlp_kwargs = filter_precision_kwargs(mlp_cls, **precision_kwargs)
        self.mlp_layers = nnx.List([
            mlp_cls(
                mlp_dims,
                rngs=rngs,
                activation=activation,
                **mlp_kwargs,
            )
            for _ in range(num_layers)
        ])

    def __call__(
        self,
        inputs: ArrayLike,
        deterministic: bool | None = None,
    ) -> Array:
        """Forward pass through the LRU model.

        Args:
            inputs: Input array of shape [..., seq_len, input_dim].
            deterministic: Whether to run in deterministic mode (for dropout).
                If None, uses training mode.

        Returns:
            Output array of shape [..., seq_len, output_dim].
        """
        inputs = jnp.asarray(inputs)
        h = self.in_layer(inputs)

        for i, (lru_layer, mlp_layer) in enumerate(
            zip(self.lru_layers, self.mlp_layers)
        ):
            # Apply layer norm before LRU layer
            h_normed = self.layer_norms_lru[i](h)

            # Apply LRU layer, optionally with bidirectional processing
            if self.bidirectional:
                # Alternate between forward and backward processing
                if i % 2 == 0:
                    h_lru = lru_layer(h_normed, deterministic=deterministic)
                else:
                    # Reverse sequence, apply LRU, then reverse back
                    h_reversed = h_normed[..., ::-1, :]
                    h_lru = lru_layer(h_reversed, deterministic=deterministic)
                    h_lru = h_lru[..., ::-1, :]
            else:
                h_lru = lru_layer(h_normed, deterministic=deterministic)

            # Residual connection for LRU
            h = h + h_lru

            # Apply layer norm before MLP layer
            h_normed = self.layer_norms_mlp[i](h)

            # Apply MLP layer
            h_mlp = mlp_layer(h_normed)

            # Residual connection for MLP
            h = h + h_mlp

        # Apply final layer norm and output projection
        h = self.out_layer_norm(h)
        return self.out_layer(h)

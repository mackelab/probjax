from typing import Callable, Mapping, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.layers.lru import LRUCell
from probjax.nn.nets.simple import MLP

from probjax.nn.utils import filter_precision_kwargs, get_active_precision_kwargs
from probjax.utils.typing import (
    Array,
    ArrayLike,
    DTypeLike,
    ModuleLikeType,
    PrecisionLike,
)


class LRUModel(nnx.Module):
    """Stacked LRU-style sequence model with optional bidirectionality.

    - Stacks recurrent cells (default: LRUCell) and MLP residual blocks.
    - Pre-norm architecture: each block is preceded by LayerNorm.
    - Optional alternating forward/backward passes for bidirectional context.

    References:
    - Orvieto et al., 2023: Linear Recurrent Units (LRU).
    - Gu & Dao, 2023: Mamba — Selective State Space Models.
    - Dao et al., 2024: Mamba-2 / SSD (Selective SSMs with diffusion).

    Notes:
    - Normalization, residual connections, and GLU heads live in this module.
      The recurrent cells (LRUCell, MambaCell, SSDCell) implement only the
      [B, L, D] -> [B, L, D] recurrent transformation.
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
        mlp_widening_factor: int = 4,
        mlp_num_hidden_layers: int = 1,
        skip_connection_lru: bool = True,
        skip_connection_mlp: bool = True,
        activation: Callable = jax.nn.gelu,
        norm_cls: ModuleLikeType = nnx.LayerNorm,
        mlp_cls: ModuleLikeType = MLP,
        initializer: Optional[nnx.Initializer] = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        # Recurrent cell choice and kwargs
        recurrent_cls: ModuleLikeType = LRUCell,
        recurrent_kwargs: Optional[Mapping] = None,
        sharding: jax.sharding.Mesh | None = None,
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
        self.recurrent_cls = recurrent_cls
        self.recurrent_kwargs = dict(recurrent_kwargs or {})
        self.skip_connection_lru = skip_connection_lru
        self.skip_connection_mlp = skip_connection_mlp
        self._mesh = sharding

        # Precision and dtype settings
        precision_kwargs = get_active_precision_kwargs(
            dtype,
            precision,
            param_dtype,
            preferred_element_type,
        )

        # Initialize linear layers with precision kwargs
        init_default = (
            nnx.initializers.variance_scaling(
                2 / max(num_layers, 1), 'fan_in', 'truncated_normal'
            )
            if initializer is None
            else initializer
        )
        linear_kwargs = filter_precision_kwargs(nnx.Linear, **precision_kwargs)
        linear_kwargs['kernel_init'] = init_default

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

        # Recurrent cell stack (each cell maps [B, T, D] -> [B, T, D])
        self.recurrent_layers = nnx.List([
            self.recurrent_cls(
                model_dim, rngs=rngs, sharding=sharding, **self.recurrent_kwargs
            )
            for _ in range(num_layers)
        ])
        # Heads for block post-processing (norm, activation, GLU, dropout)
        self.block_norms = nnx.List([
            norm_cls(model_dim, rngs=rngs) for _ in range(num_layers)
        ])
        if dropout_rate is not None:
            self.block_dropout1 = nnx.List([
                nnx.Dropout(dropout_rate, rngs=rngs) for _ in range(num_layers)
            ])
            self.block_dropout2 = nnx.List([
                nnx.Dropout(dropout_rate, rngs=rngs) for _ in range(num_layers)
            ])
        else:
            self.block_dropout1 = None
            self.block_dropout2 = None
        self.block_out1 = nnx.List([
            nnx.Linear(model_dim, model_dim, rngs=rngs, **linear_kwargs)
            for _ in range(num_layers)
        ])
        self.block_out2 = nnx.List([
            nnx.Linear(model_dim, model_dim, rngs=rngs, **linear_kwargs)
            for _ in range(num_layers)
        ])
        self.block_activation = activation

        # MLP layers for processing between LRU blocks
        mlp_dims = (
            [model_dim]
            + [mlp_widening_factor * model_dim] * mlp_num_hidden_layers
            + [model_dim]
        )
        mlp_kwargs = filter_precision_kwargs(mlp_cls, **precision_kwargs)
        self.mlp_layers = nnx.List([
            mlp_cls(
                mlp_dims,
                rngs=rngs,
                activation=activation,
                activate_final=True,
                sharding=sharding,
                **mlp_kwargs,
            )
            for _ in range(num_layers)
        ])

    def __call__(
        self,
        inputs: ArrayLike,
        deterministic: bool | None = None,
        rng: jax.Array | None = None,
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
        shape = inputs.shape
        # Flatten leading batch dims to [-1, T, D]
        x = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        h = self.in_layer(x)

        for i, mlp_layer in enumerate(self.mlp_layers):
            # Apply layer norm before LRU layer
            h_normed = self.layer_norms_lru[i](h)

            # Apply LRU layer, optionally with bidirectional processing
            if self.bidirectional:
                # Alternate between forward and backward processing
                if i % 2 == 0:
                    h_cell_in = self.block_norms[i](h_normed)
                    h_cell = self.recurrent_layers[i](h_cell_in, rng=rng)
                else:
                    # Reverse sequence, apply LRU, then reverse back
                    h_reversed = h_normed[..., ::-1, :]
                    h_cell_in = self.block_norms[i](h_reversed)
                    h_cell = self.recurrent_layers[i](h_cell_in, rng=rng)
                    h_cell = h_cell[..., ::-1, :]
            else:
                h_cell_in = self.block_norms[i](h_normed)
                h_cell = self.recurrent_layers[i](h_cell_in, rng=rng)

            # Residual connection for LRU
            # GLU head: activation + optional dropout + gated linear
            x = self.block_activation(h_cell)
            if self.block_dropout1 is not None:
                x = self.block_dropout1[i](x, deterministic=deterministic, rngs=rng)
            x = self.block_out1[i](x) * jax.nn.sigmoid(self.block_out2[i](x))
            if self.block_dropout2 is not None:
                x = self.block_dropout2[i](x, deterministic=deterministic, rngs=rng)
            h = h + x if self.skip_connection_lru else x

            # Apply layer norm before MLP layer
            h_normed = self.layer_norms_mlp[i](h)

            # Apply MLP layer
            h_mlp = mlp_layer(h_normed, rng=rng)

            # Residual connection for MLP
            h = h + h_mlp if self.skip_connection_mlp else h_mlp

        # Apply final layer norm and output projection
        h = self.out_layer_norm(h)
        h = self.out_layer(h)
        return h.reshape(shape[:-2] + (shape[-2], self.output_dim))

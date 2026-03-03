from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.nets.simple import MLP
from probjax.nn.sharding import ShardingCfg, resolve_sharding_mesh

from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
    module_accepts_rng,
)
from probjax.utils.typing import (
    Array,
    ArrayLike,
    DTypeLike,
    ModuleLikeType,
    PrecisionLike,
)


class CouplingMLP(nnx.Module):
    """Coupling layer using MLP for bijective transformations.

    This module implements a coupling layer that splits the input into two parts,
    uses an MLP to compute transformation parameters for one part based on the other,
    and applies a bijective transformation.
    """

    def __init__(
        self,
        split_index: int,
        bij_params_dim: int,
        bijector: Callable,
        rngs: nnx.Rngs,
        *,
        context_dim: Optional[int] = None,
        context_features: Optional[int] = None,
        hidden_dims: Sequence[int] = (50, 50),
        activation: Callable = jax.nn.gelu,
        activate_final: bool = False,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        mlp_cls: ModuleLikeType = MLP,
        sharding_cfg: ShardingCfg | None = None,
        **kwargs,
    ):
        """Initialize the CouplingMLP module.

        Args:
            split_index (int): Index at which to split the input along the last axis.
                Must be positive and less than the input dimension.
            bij_params_dim (int): Dimension of the bijector parameters output
                by the MLP. Must be positive.
            bijector (Callable): Bijective transformation function that takes
                parameters and input, and returns transformed output.
            rngs (nnx.Rngs): Random number generator state.
            context_dim (Optional[int], optional): Dimension of additional context
                to be concatenated with the first part of the split input. If None,
                no context is used. Defaults to None.
            hidden_dims (Sequence[int], optional): Sequence of hidden layer
                dimensions for the MLP. Defaults to (50, 50).
            activation (Callable, optional): Activation function for the MLP.
                Defaults to jax.nn.gelu.
            activate_final (bool, optional): Whether to apply activation to the
                final layer of the MLP. Defaults to False.
            dtype (DTypeLike | None, optional): Computation dtype. Defaults to None.
            param_dtype (DTypeLike | None, optional): Parameter dtype.
                Defaults to None.
            precision (PrecisionLike | None, optional): Computation precision.
                Defaults to None.
            preferred_element_type (DTypeLike | None, optional): Preferred element
                type. Defaults to None.
            mlp_cls (type[nnx.Module], optional): MLP class to use for the
                conditioner. Defaults to MLP.
            **kwargs: Additional keyword arguments passed to the MLP constructor.

        Raises:
            ValueError: If split_index is not positive, bij_params_dim is not
                positive, hidden_dims is empty, or context_dim is not positive
                when provided.
        """
        super().__init__()

        # Input validation
        if split_index <= 0:
            raise ValueError(f"split_index must be positive, got {split_index}")
        if bij_params_dim <= 0:
            raise ValueError(f"bij_params_dim must be positive, got {bij_params_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        if any(dim <= 0 for dim in hidden_dims):
            raise ValueError(
                f"All hidden dimensions must be positive, got {hidden_dims}"
            )
        if context_dim is not None and context_dim <= 0:
            raise ValueError(
                f"context_dim must be positive when provided, got {context_dim}"
            )

        self.split_index = split_index
        self.bij_params_dim = bij_params_dim
        # Prefer explicit context_dim; fallback to alias for consistency
        self.context_dim = context_dim if context_dim is not None else context_features
        self.bijector = bijector
        self._sharding_cfg = sharding_cfg
        self._mesh = resolve_sharding_mesh(sharding_cfg)

        # Precision and dtype settings
        precision_kwargs = get_active_precision_kwargs(
            dtype,
            precision,
            param_dtype,
            preferred_element_type,
        )

        # Build MLP dimensions
        in_dim = split_index + (self.context_dim if self.context_dim is not None else 0)
        feature_dims = [in_dim] + list(hidden_dims) + [bij_params_dim]

        # Create MLP conditioner
        self.conditioner = mlp_cls(
            feature_dims,
            rngs=rngs,
            activation=activation,
            activate_final=activate_final,
            context_dim=None,  # Context is handled manually in this layer
            sharding_cfg=self._sharding_cfg,
            **filter_precision_kwargs(mlp_cls, **precision_kwargs),
            **kwargs,
        )
        self._conditioner_accepts_rng = module_accepts_rng(self.conditioner)

    def __call__(
        self,
        x: ArrayLike,
        context: Optional[ArrayLike] = None,
        rng: jax.Array | None = None,
        **bijector_kwargs,
    ) -> Array:
        """Apply the coupling transformation.

        Args:
            x (ArrayLike): Input array of shape [..., input_dim] where
                input_dim > split_index.
            context (Optional[ArrayLike], optional): Context array of shape
                [..., context_dim]. Required if context_dim was specified
                during initialization. Defaults to None.
            **bijector_kwargs: Additional keyword arguments passed to the
                bijector function.

        Returns:
            Array: Transformed output of the same shape as input.

        Raises:
            ValueError: If context is required but not provided, if context is provided
                but context_dim is None, or if input dimensions are invalid.
        """
        x = jnp.asarray(x)

        # Validate input dimensions
        if x.shape[-1] <= self.split_index:
            raise ValueError(
                f"Input last dimension ({x.shape[-1]}) must be greater than "
                f"split_index ({self.split_index})"
            )

        # Validate context requirements
        if self.context_dim is not None and context is None:
            raise ValueError("context is required when context_dim is specified")
        if self.context_dim is None and context is not None:
            raise ValueError("context provided but context_dim is None")

        # Split the input
        x1, x2 = jnp.split(x, [self.split_index], axis=-1)

        # Prepare input for conditioner
        conditioner_input = x1
        if context is not None:
            context = jnp.asarray(context)
            # Ensure context has compatible shape for broadcasting
            if context.ndim < x1.ndim:
                # Add dimensions to match x1's batch dimensions
                for _ in range(x1.ndim - context.ndim):
                    context = context[None, ...]
            conditioner_input = jnp.concatenate([x1, context], axis=-1)

        # Compute bijector parameters
        if self._conditioner_accepts_rng:
            bijector_params = self.conditioner(conditioner_input, rng=rng)
        else:
            bijector_params = self.conditioner(conditioner_input)

        # Apply bijective transformation to x2, keeping x1 unchanged
        y1 = x1
        y2 = self.bijector(bijector_params, x2, **bijector_kwargs)

        # Concatenate results
        return jnp.concatenate([y1, y2], axis=-1)

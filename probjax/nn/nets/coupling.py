from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.nets.simple import MLP
from probjax.nn.nets.transformer import Transformer
from probjax.nn.sharding import ShardingCfg

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


def _default_split_fn(x, split_index: int):
    parts = jnp.split(x, [split_index], axis=-1)
    x1, x2 = parts[0], parts[1]
    return x1, x2


def _default_merge_fn(y1, y2):
    return jnp.concatenate([y1, y2], axis=-1)


def _validate_context(context_dim: Optional[int], context: Optional[ArrayLike]) -> None:
    if context_dim is not None and context is None:
        raise ValueError("context is required when context_dim is specified")
    if context_dim is None and context is not None:
        raise ValueError("context provided but context_dim is None")


def _broadcast_context_to_x1(context: ArrayLike, x1):
    context = jnp.asarray(context)
    if context.ndim < x1.ndim:
        for _ in range(x1.ndim - context.ndim):
            context = context[None, ...]
    return context


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
        split_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None,
        merge_fn: Optional[Callable[[Array, Array], Array]] = None,
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
        self.split_fn = split_fn
        self.merge_fn = merge_fn
        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)

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
            sharding_cfg=self.sharding_cfg,
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

        _validate_context(self.context_dim, context)

        # Split the input
        if self.split_fn is None:
            x1, x2 = _default_split_fn(x, self.split_index)
        else:
            x1, x2 = self.split_fn(x)
            if x1.shape[-1] != self.split_index:
                raise ValueError(
                    "split_fn must produce x1 with last dimension equal to "
                    f"split_index ({self.split_index}), got {x1.shape[-1]}"
                )

        # Prepare input for conditioner
        conditioner_input = x1
        if context is not None:
            ctx = _broadcast_context_to_x1(context, x1)
            conditioner_input = jnp.concatenate([x1, ctx], axis=-1)

        # Compute bijector parameters
        if self._conditioner_accepts_rng:
            bijector_params = self.conditioner(conditioner_input, rng=rng)
        else:
            bijector_params = self.conditioner(conditioner_input)

        # Apply bijective transformation to x2, keeping x1 unchanged
        y1 = x1
        y2 = self.bijector(bijector_params, x2, **bijector_kwargs)

        # Concatenate results
        if self.merge_fn is None:
            return _default_merge_fn(y1, y2)
        return self.merge_fn(y1, y2)


class CouplingTransformer(nnx.Module):
    """Coupling layer with a Transformer conditioner."""

    def __init__(
        self,
        split_index: int,
        bij_params_dim: int,
        bijector: Callable,
        rngs: nnx.Rngs,
        *,
        context_dim: Optional[int] = None,
        context_features: Optional[int] = None,
        split_fn: Optional[Callable[[Array], tuple[Array, Array]]] = None,
        merge_fn: Optional[Callable[[Array, Array], Array]] = None,
        model_dim: int = 32,
        num_heads: int = 2,
        num_layers: int = 2,
        attn_size: int = 8,
        widening_factor: int = 2,
        transformer: Optional[Transformer] = None,
        sharding_cfg: ShardingCfg | None = None,
        **kwargs,
    ):
        super().__init__()
        if split_index <= 0:
            raise ValueError(f"split_index must be positive, got {split_index}")
        if bij_params_dim <= 0:
            raise ValueError(f"bij_params_dim must be positive, got {bij_params_dim}")

        self.split_index = split_index
        self.bij_params_dim = bij_params_dim
        self.context_dim = context_dim if context_dim is not None else context_features
        if self.context_dim is not None and self.context_dim <= 0:
            raise ValueError(
                f"context_dim must be positive when provided, got {self.context_dim}"
            )
        self.bijector = bijector
        self.split_fn = split_fn
        self.merge_fn = merge_fn
        self.sharding_cfg = ShardingCfg.resolve_or_noop(sharding_cfg)

        if transformer is None:
            transformer = Transformer(
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                attn_size=attn_size,
                widening_factor=widening_factor,
                context_dim=self.context_dim,
                sharding_cfg=self.sharding_cfg,
                rngs=rngs,
                **kwargs,
            )
        self.transformer = transformer
        self.encoder = nnx.Linear(1, model_dim, rngs=rngs)
        self.decoder = nnx.Linear(split_index * model_dim, bij_params_dim, rngs=rngs)

    def __call__(
        self,
        x: ArrayLike,
        context: Optional[ArrayLike] = None,
        rng: jax.Array | None = None,
        **bijector_kwargs,
    ) -> Array:
        x = jnp.asarray(x)
        if x.shape[-1] <= self.split_index:
            raise ValueError(
                f"Input last dimension ({x.shape[-1]}) must be greater than "
                f"split_index ({self.split_index})"
            )

        _validate_context(self.context_dim, context)

        if self.split_fn is None:
            x1, x2 = _default_split_fn(x, self.split_index)
        else:
            x1, x2 = self.split_fn(x)
            if x1.shape[-1] != self.split_index:
                raise ValueError(
                    "split_fn must produce x1 with last dimension equal to "
                    f"split_index ({self.split_index}), got {x1.shape[-1]}"
                )

        tokens = x1[..., :, None]
        tokens = self.encoder(tokens)
        ctx = None
        if context is not None:
            ctx = _broadcast_context_to_x1(context, x1)
        tokens = self.transformer(tokens, context=ctx, rng=rng)

        flat_tokens = tokens.reshape(
            tokens.shape[:-2] + (self.split_index * tokens.shape[-1],)
        )
        bijector_params = self.decoder(flat_tokens)

        y1 = x1
        y2 = self.bijector(bijector_params, x2, **bijector_kwargs)
        if self.merge_fn is None:
            return _default_merge_fn(y1, y2)
        return self.merge_fn(y1, y2)

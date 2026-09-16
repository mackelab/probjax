"""Time-conditioned transformer for continuous token-valued events."""

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.layers.encoding import GaussianFourierEmbedding, PosEncode
from probjax.nn.nets.simple import MLP
from probjax.nn.nets.transformer import Transformer
from probjax.nn.utils import DEFAULT_MODULE, filter_precision_kwargs


class DiffusionTransformer(Transformer):
    """Transformer with time and optional global conditioning at every block.

    Inputs have shape ``[..., tokens, features]``. Token counts can change;
    feature widths are fixed. Time is scalar, ``[...]``, or ``[..., 1]``.
    Global context has shape ``[..., context_dim]`` and is required when that
    constructor dimension is set. Optional ``r`` adds a second time embedding
    for mean-flow models. Masks, cross-attention and dropout options are
    forwarded to :class:`Transformer`.

    All Transformer class factories remain overridable. Additional factories
    control input/output projections, time encoding and position encoding.
    Set ``position_cls=None`` for permutation-equivariant token processing.
    """

    projection_cls = nnx.Linear
    fourier_cls = GaussianFourierEmbedding
    time_mlp_cls = MLP
    position_cls = PosEncode

    def __init__(
        self,
        features: int,
        *,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        attn_size: int = 32,
        time_embed_dim: int = 128,
        fourier_dim: int = 64,
        context_dim: int | None = None,
        projection_cls=DEFAULT_MODULE,
        fourier_cls=DEFAULT_MODULE,
        time_mlp_cls=DEFAULT_MODULE,
        position_cls=DEFAULT_MODULE,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        for name, value in dict(
            features=features,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=attn_size,
            time_embed_dim=time_embed_dim,
            fourier_dim=fourier_dim,
        ).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if context_dim is not None and context_dim <= 0:
            raise ValueError("context_dim must be positive")
        factories = dict(
            projection_cls=projection_cls,
            fourier_cls=fourier_cls,
            time_mlp_cls=time_mlp_cls,
            position_cls=position_cls,
        )
        factories = {
            name: getattr(type(self), name) if cls is DEFAULT_MODULE else cls
            for name, cls in factories.items()
        }
        projection_cls = factories['projection_cls']
        fourier_cls = factories['fourier_cls']
        time_mlp_cls = factories['time_mlp_cls']
        position_cls = factories['position_cls']
        super().__init__(
            model_dim,
            num_heads,
            num_layers,
            attn_size,
            context_dim=time_embed_dim + (context_dim or 0),
            rngs=rngs,
            **kwargs,
        )
        self.features = features
        self.global_context_dim = context_dim
        precision = {
            k: kwargs[k]
            for k in ('dtype', 'param_dtype', 'precision', 'preferred_element_type')
            if k in kwargs
        }
        self.input_projection = projection_cls(
            features,
            model_dim,
            rngs=rngs,
            **filter_precision_kwargs(projection_cls, **precision),
        )
        self.output_projection = projection_cls(
            model_dim,
            features,
            rngs=rngs,
            **filter_precision_kwargs(projection_cls, **precision),
        )
        self.time_fourier = fourier_cls(
            1,
            fourier_dim,
            rngs=rngs,
            **filter_precision_kwargs(fourier_cls, **precision),
        )
        self.time_mlp = time_mlp_cls(
            [fourier_dim, time_embed_dim, time_embed_dim],
            activation=jax.nn.silu,
            rngs=rngs,
            **filter_precision_kwargs(time_mlp_cls, **precision),
        )
        self.position = position_cls(rngs=rngs) if position_cls is not None else None

    def _time_embedding(self, t, batch_shape):
        t = jnp.asarray(t)
        while t.ndim > len(batch_shape) + 1 and t.shape[-1] == 1:
            t = jnp.squeeze(t, axis=-1)
        if t.ndim <= len(batch_shape):
            t = t[..., None]
        t = jnp.broadcast_to(t, batch_shape + (1,))
        return self.time_mlp(self.time_fourier(t))

    def __call__(self, t, x, r=None, context=None, *, rng=None, **kwargs):
        if x.ndim < 2 or x.shape[-1] != self.features:
            raise ValueError("x must have shape [..., tokens, features]")
        batch_shape = x.shape[:-2]
        condition = self._time_embedding(t, batch_shape)
        if r is not None:
            # Normalize separately so (batch,) and (batch, 1) do not outer-broadcast.
            condition = condition + self._time_embedding(r, batch_shape)
        if self.global_context_dim is not None:
            if context is None or context.shape[-1] != self.global_context_dim:
                raise ValueError("context must have the configured context_dim")
            context = jnp.broadcast_to(
                context, batch_shape + (self.global_context_dim,)
            )
            condition = jnp.concatenate([condition, context], axis=-1)
        elif context is not None:
            raise ValueError("context provided but context_dim is None")
        hidden = self.input_projection(x)
        if self.position is not None:
            hidden = self.position(hidden)
        hidden = super().__call__(hidden, context=condition, rng=rng, **kwargs)
        return self.output_projection(hidden)

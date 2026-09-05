from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.nn.layers.encoding import GaussianFourierEmbedding
from probjax.nn.layers.fuse import AffineFuse
from probjax.nn.nets.simple import MLP, ResNet
from probjax.nn.utils import filter_precision_kwargs, get_active_precision_kwargs
from probjax.utils.typing import Array, ArrayLike, DTypeLike, ModuleLikeType, PrecisionLike


class TimeMLP(nnx.Module):
    """Time-conditioned residual MLP for flow-matching / diffusion nets.

    Time (and, for mean-flow-style ``(t, r)`` pairs, the second time ``r``) is
    encoded with a :class:`GaussianFourierEmbedding` followed by a small MLP,
    matching the sinusoidal-embedding recipe used in DDPM/EDM. The resulting
    embedding conditions a residual MLP body (:class:`ResNet`) via AdaLN-style
    per-block affine modulation (:class:`AffineFuse` after a
    :class:`~flax.nnx.LayerNorm`): each block predicts a learned scale and
    shift from the time embedding and applies it to its (normalized)
    activations. The modulation starts at the identity transform (the
    underlying fusion projection is zero-initialized), so the network starts
    out as a plain residual MLP and only learns time-dependent behaviour as
    needed.

    An optional extra ``context`` (e.g. a class label embedding) is
    concatenated with the time embedding before conditioning, so the same
    mechanism also carries auxiliary conditioning.

    Call signature matches the ``net(t, x, r=None, context=None, rng=None)``
    convention used by :mod:`probjax.nn.generative` (flow matching, mean
    flow, diffusion).
    """

    def __init__(
        self,
        features: int,
        *,
        hidden_dim: int = 128,
        depth: int = 3,
        time_embed_dim: int = 128,
        fourier_dim: int = 64,
        context_dim: Optional[int] = None,
        activation=jax.nn.silu,
        norm_cls: ModuleLikeType = nnx.LayerNorm,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike | None = None,
        precision: PrecisionLike | None = None,
        preferred_element_type: DTypeLike | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize the time-conditioned MLP.

        Args:
            features: Dimensionality of ``x`` (both input and output).
            hidden_dim: Hidden width of the residual MLP body.
            depth: Number of residual blocks in the body.
            time_embed_dim: Dimensionality of the (post-MLP) time embedding
                used for AdaLN conditioning.
            fourier_dim: Dimensionality of the raw Gaussian Fourier features
                before the time embedding MLP projects them down/up to
                ``time_embed_dim``.
            context_dim: Optional dimensionality of an auxiliary conditioning
                vector, concatenated with the time embedding.
            activation: Activation function used throughout.
            norm_cls: Normalization layer applied before each block's AdaLN
                modulation. Defaults to ``nnx.LayerNorm``.
            dtype: Computation dtype (optional).
            param_dtype: Parameter dtype (optional).
            precision: Computation precision (optional).
            preferred_element_type: Preferred element type (optional).
            rngs: Random number generators.

        Raises:
            ValueError: If any dimension argument is not positive.
        """
        if features <= 0:
            raise ValueError(f"features must be positive, got {features}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if time_embed_dim <= 0:
            raise ValueError(f"time_embed_dim must be positive, got {time_embed_dim}")
        if fourier_dim <= 0:
            raise ValueError(f"fourier_dim must be positive, got {fourier_dim}")
        if context_dim is not None and context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")

        self.features = features
        self.context_dim = context_dim

        precision_kwargs = get_active_precision_kwargs(
            dtype, precision, param_dtype, preferred_element_type
        )

        self.time_fourier = GaussianFourierEmbedding(
            1,
            fourier_dim,
            rngs=rngs,
            **filter_precision_kwargs(GaussianFourierEmbedding, **precision_kwargs),
        )
        self.time_mlp = MLP(
            [fourier_dim, time_embed_dim, time_embed_dim],
            activation=activation,
            rngs=rngs,
            **filter_precision_kwargs(MLP, **precision_kwargs),
        )

        cond_dim = time_embed_dim + (context_dim or 0)
        self.body = ResNet(
            features,
            features,
            hidden_dim=hidden_dim,
            num_hidden_layers=depth,
            context_dim=cond_dim,
            activation=activation,
            norm_cls=norm_cls,
            context_fuse_cls=AffineFuse,
            rngs=rngs,
            **filter_precision_kwargs(ResNet, **precision_kwargs),
        )

    def _time_embedding(self, t: ArrayLike, batch_shape: tuple[int, ...]) -> Array:
        t = jnp.broadcast_to(jnp.asarray(t), batch_shape + (1,))
        return self.time_mlp(self.time_fourier(t))

    def __call__(
        self,
        t: ArrayLike,
        x: Array,
        r: Optional[ArrayLike] = None,
        context: Optional[Array] = None,
        *,
        rng: jax.Array | None = None,
        **kwargs,
    ) -> Array:
        """Forward pass.

        Args:
            t: Time, broadcastable to ``x.shape[:-1]``.
            x: Input array of shape ``[..., features]``.
            r: Optional second time (mean-flow convention). When given, its
                embedding is added to ``t``'s.
            context: Optional auxiliary conditioning of shape
                ``[..., context_dim]``. Required iff ``context_dim`` was set.
            rng: Optional RNG, forwarded to the residual body.

        Returns:
            Array of shape ``[..., features]``.
        """
        del kwargs
        batch_shape = x.shape[:-1]

        temb = self._time_embedding(t, batch_shape)
        if r is not None:
            temb = temb + self._time_embedding(r, batch_shape)

        if self.context_dim is not None:
            if context is None:
                raise ValueError("context is required when context_dim is specified")
            context = jnp.broadcast_to(context, batch_shape + (context.shape[-1],))
            temb = jnp.concatenate([temb, context], axis=-1)
        elif context is not None:
            raise ValueError("context provided but context_dim is None")

        return self.body(x, temb, rng=rng)

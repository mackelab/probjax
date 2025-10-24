from functools import partial
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Array, PrecisionLike

from probjax.nn.layers.conv import (
    ResnetBlock,
    SpatialSelfAttention,
)
from probjax.nn.utils import (
    filter_precision_kwargs,
    get_active_precision_kwargs,
)


class UNet(nnx.Module):
    """
    Flexible U-Net with pluggable builders for:
      - resnet blocks (default: ResnetBlock)
      - downsampling convs (default: nnx.Conv)
      - upsampling convs (default: nnx.ConvTranspose)
      - attention blocks (default: SpatialSelfAttention, or None to disable)

    You can pass either:
      - *_cls
      - *_factory callables that build a module for a given (in_ch, out_ch).

    Factories take precedence if provided.
    """

    def __init__(
        self,
        in_features: int,
        out_features: Sequence[int],
        *,
        # --- shape/behavior ---
        kernel_size: int | Sequence[int] = 4,
        strides: int | Sequence[int] = 2,
        kernel_size_resnet: int | Sequence[int] = 3,
        strides_resnet: int | Sequence[int] = 1,
        use_attention: bool | Sequence[bool] = False,
        context_features: int | None = None,
        resize_method: str = "bilinear",
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        # --- initialization/precision ---
        precision: PrecisionLike | None = None,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype | None = None,
        preferred_element_type: jnp.dtype | None = None,
        # --- pluggable builders: classes ---
        resnet_block_cls: type[nnx.Module] = ResnetBlock,
        conv_down_cls: type[nnx.Module] | Sequence[type[nnx.Module]] = nnx.Conv,
        conv_up_cls: type[nnx.Module] | Sequence[type[nnx.Module]] = nnx.ConvTranspose,
        attn_cls: type[nnx.Module] = SpatialSelfAttention,
        conv_cls: type[nnx.Module] = nnx.Conv,
        rngs: nnx.Rngs,
    ):
        assert len(out_features) >= 2, "Must have at least 2 output channels"

        self.in_features = in_features
        self.out_features = list(out_features)
        self.num_stages = len(out_features)
        self.resize_method = resize_method  # Triggered if user shapes do not mat
        self.preferred_element_type = preferred_element_type

        precision_kwargs = get_active_precision_kwargs(
            dtype,
            precision,
            param_dtype,
            preferred_element_type,
        )

        # Normalize/validate attention mask
        if isinstance(use_attention, Sequence):
            assert len(use_attention) == self.num_stages, (
                "`use_attention` list must match number of stages"
            )
            self.attn_mask = list(bool(x) for x in use_attention)
            self.use_attention = any(self.attn_mask)
        else:
            self.attn_mask = [bool(use_attention)] * self.num_stages
            self.use_attention = bool(use_attention)

        # ---------------------------------------------------------------------
        # Builder helpers
        # ---------------------------------------------------------------------
        # ResNet block builder
        _resnet_block = partial(
            resnet_block_cls,
            kernel_size=kernel_size_resnet,
            strides=strides_resnet,
            context_features=context_features,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            rngs=rngs,
            **filter_precision_kwargs(resnet_block_cls, **precision_kwargs),
        )

        _down_blocks = nnx.List()
        for i in range(self.num_stages - 1):
            if isinstance(conv_down_cls, Sequence):
                conv_down_cls_i = conv_down_cls[i]
            else:
                conv_down_cls_i = conv_down_cls

            _down_blocks.append(
                partial(
                    conv_down_cls_i,
                    kernel_size=kernel_size,
                    strides=strides,
                    rngs=rngs,
                    **filter_precision_kwargs(conv_down_cls_i, **precision_kwargs),
                )
            )

        _up_blocks = nnx.List()
        for i in range(self.num_stages - 1):
            if isinstance(conv_up_cls, Sequence):
                conv_up_cls_i = conv_up_cls[i]
            else:
                conv_up_cls_i = conv_up_cls
            _up_blocks.append(
                partial(
                    conv_up_cls_i,
                    kernel_size=kernel_size,
                    strides=strides,
                    rngs=rngs,
                    **filter_precision_kwargs(conv_up_cls_i, **precision_kwargs),
                )
            )

        # Initial/final projection builders
        _init_final = partial(
            conv_cls,
            kernel_size=1,
            use_bias=False,
            rngs=rngs,
            **filter_precision_kwargs(conv_cls, **precision_kwargs),
        )

        _attn_block = partial(
            attn_cls,
            dropout_rate=dropout_rate,
            rngs=rngs,
            **filter_precision_kwargs(attn_cls, **precision_kwargs),
        )

        # ---------------------------------------------------------------------
        # Down path
        # ---------------------------------------------------------------------
        self.resnet_blocks_down = nnx.List()
        self.downsampling_layers = nnx.List()
        self.att_layers_down = nnx.List()

        for i in range(self.num_stages):
            # ResNet in each stage works on out_features[i]
            self.resnet_blocks_down.append(
                _resnet_block(self.out_features[i], self.out_features[i])
            )
            # Optional attention for this stage
            if self.attn_mask[i]:
                self.att_layers_down.append(_attn_block(self.out_features[i]))
            else:
                self.att_layers_down.append(None)

            # Insert a downsample conv between stages (0->1, 1->2, ...)
            if i > 0:
                self.downsampling_layers.append(
                    _down_blocks[i - 1](self.out_features[i - 1], self.out_features[i])
                )

        # ---------------------------------------------------------------------
        # Middle block
        # ---------------------------------------------------------------------
        top_ch = self.out_features[-1]
        self.middle_block1 = _resnet_block(top_ch, top_ch)
        self.middle_block2 = _resnet_block(top_ch, top_ch)
        self.att_middle = _attn_block(top_ch) if self.attn_mask[-1] else None

        # ---------------------------------------------------------------------
        # Up path (mirror of down)
        # We build resnet_blocks_up to take (ch*2)->ch due to skip concat.
        # Attention order mirrors the down path order.
        # ---------------------------------------------------------------------
        self.resnet_blocks_up = nnx.List()
        self.att_layers_up = nnx.List()
        self.upsampling_layers = nnx.List()

        for i in reversed(range(self.num_stages)):
            ch = self.out_features[i]

            # ResNet block after concatenating skip connection
            self.resnet_blocks_up.append(_resnet_block(ch * 2, ch))

            # Attention layer (mirroring down path)
            if self.attn_mask[self.num_stages - i - 1]:
                self.att_layers_up.append(_attn_block(ch))
            else:
                self.att_layers_up.append(None)

            # Upsample conv (skip for bottom-most stage)
            if i > 0:
                self.upsampling_layers.append(
                    _up_blocks[i - 1](ch, self.out_features[i - 1])
                )

        # Initial and final 1x1 projections
        self.conv_initial = _init_final(in_features, self.out_features[0])
        self.conv_final = _init_final(
            self.out_features[0] * 2, in_features, kernel_init=nnx.initializers.zeros
        )

    def __call__(
        self,
        inputs: Array,
        context: Optional[Array] = None,
        verbose: bool = False,
        deterministic: bool = True,
    ) -> Array:
        # 1) Initial projection
        x = self.conv_initial(inputs)

        # Stash features before each downsample for skips
        pre_downsampling = [x]

        # 2) Down path
        for i in range(self.num_stages):
            x = self.resnet_blocks_down[i](x, context, deterministic=deterministic)
            if self.att_layers_down[i] is not None:
                x = self.att_layers_down[i](x, context, deterministic=deterministic)
            pre_downsampling.append(x)
            if i < self.num_stages - 1:
                if verbose:
                    print("Down:", x.shape)
                x = self.downsampling_layers[i](x).astype(self.preferred_element_type)

        # 3) Middle
        x = self.middle_block1(x, context, deterministic=deterministic)
        if self.att_middle is not None:
            x = self.att_middle(x, context, deterministic=deterministic)
        x = self.middle_block2(x, context, deterministic=deterministic)
        if verbose:
            print("Mid:", x.shape)

        # 4) Up path
        for idx in range(self.num_stages):
            down = pre_downsampling.pop()
            # Ensure spatial match (covers odd sizes / stride combos)
            if x.shape != down.shape:
                x = jax.image.resize(x, down.shape, method=self.resize_method)
            x = jnp.concatenate([down, x], axis=-1)

            x = self.resnet_blocks_up[idx](x, context, deterministic=deterministic)
            if self.att_layers_up[idx] is not None:
                x = self.att_layers_up[idx](x, context, deterministic=deterministic)

            if idx < self.num_stages - 1:
                x = self.upsampling_layers[idx](x).astype(self.preferred_element_type)
                if verbose:
                    print("Up:", x.shape)

        # 5) Final projection (+ last skip from very beginning)
        pre_in = pre_downsampling.pop()
        x = jnp.concatenate([pre_in, x], axis=-1)
        x = self.conv_final(x)

        return x

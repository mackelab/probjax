"""Shim: old_pallas shares the mask/bias pytree classes with the live package
so A/B comparisons operate on identical objects."""

from probjax.nn.pallas_kernels.attention_mask_bias import *  # noqa: F401,F403
from probjax.nn.pallas_kernels.attention_mask_bias import (  # noqa: F401
    AttentionBias,
    AttentionMask,
)

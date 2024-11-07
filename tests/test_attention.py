import jax.numpy as jnp
import numpy as np
import pytest

import jax
from probjax.nn.attention import (
    memory_efficient_dot_product_attention,
)


# @pytest.mark.parametrize(
#     "batch_shape, seq_len, dim, num_heads",
#     [
#         ((), 10, 32, 4),
#         ((1,), 10000, 32, 1),
#         ((50,), 100, 5, 2),
#     ],
# )
# def test_attention(batch_shape, seq_len, dim, num_heads):
#     x = np.random.rand(*batch_shape, seq_len, num_heads, dim)

#     # Test dense dot product attention
#     x_attn = jax.nn.dot_product_attention(x, x, x).reshape(x.shape)
#     x_attn_chunked = memory_efficient_dot_product_attention(x, x, x)
#     assert jnp.allclose(
#         x_attn, x_attn_chunked, atol=1e-2
#     ), "Dense and chunked attention should be equal"

#     # Test masked dot product attention
#     mask = np.random.binomial(1, 0.5, (seq_len, seq_len))
#     x_attn_masked = jax.nn.dot_product_attention(x, x, x, mask=mask)
#     x_attn_masked_chunked = memory_efficient_dot_product_attention(x, x, x, mask=mask)
#     assert jnp.allclose(
#         x_attn_masked, x_attn_masked_chunked, atol=1e-2
#     ), "Dense and chunked attention should be equal with mask"

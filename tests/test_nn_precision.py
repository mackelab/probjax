"""Tests for the precision kwargs plumbing in probjax.nn (P0-2 / P1-7).

Covers:
- cast_output semantics (None is a strict no-op; explicit dtype casts).
- filter_precision_kwargs drops None and unsupported kwargs.
- The four precision kwargs are forwarded.
- The Conv/ConvTranspose UserWarning for silently discarded
  preferred_element_type (added in PR #32) is preserved.
- Regression: constructing/running layers with preferred_element_type=None
  (the default) never passes None to ``x.astype`` -- on strict JAX this used
  to raise TypeError, on lenient JAX it silently corrupted dtypes.
"""

import warnings

import jax.numpy as jnp
import pytest
from flax import nnx

from probjax.nn.layers.attention import InducedSelfAttention
from probjax.nn.layers.conv import (
    ConvBlock,
    RescaleConv,
    ResizeConv,
    ResnetBlock,
    SpatialSelfAttention,
)
from probjax.nn.layers.encoding import GaussianFourierEmbedding
from probjax.nn.nets.unets import UNet
from probjax.nn.utils import cast_output, filter_precision_kwargs

# ---------------------------------------------------------------------------
# Helper behavior
# ---------------------------------------------------------------------------


def test_cast_output_none_is_noop():
    x = jnp.ones(3)
    # Identity: astype must never be invoked with None.
    assert cast_output(x, None) is x


def test_cast_output_casts_when_set():
    out = cast_output(jnp.ones(3, dtype=jnp.float32), jnp.float16)
    assert out.dtype == jnp.float16


def test_filter_precision_kwargs_drops_nones_and_unsupported():
    kwargs = dict(dtype=jnp.float16, precision=None, preferred_element_type=None)
    assert filter_precision_kwargs(nnx.Linear, **kwargs) == {"dtype": jnp.float16}
    # LayerNorm takes no `precision`.
    assert filter_precision_kwargs(
        nnx.LayerNorm, dtype=jnp.float16, precision="highest"
    ) == {"dtype": jnp.float16}


def test_filter_precision_kwargs_forwards_core_kwargs_to_var_kwargs_class():
    """A ``**kwargs`` constructor must still receive the core precision kwargs.

    ``MultiHeadAttention`` takes ``*args, **kwargs``, so a name check sees
    nothing and used to drop ``dtype``/``param_dtype``/``precision``, leaving the
    sublayer in fp32 regardless of the requested dtype.
    """
    from probjax.nn.layers.attention import MultiHeadAttention

    got = filter_precision_kwargs(
        MultiHeadAttention,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        precision=None,
        preferred_element_type=jnp.float32,
    )
    assert got == {"dtype": jnp.bfloat16, "param_dtype": jnp.float32}
    # `preferred_element_type` is not a standard Flax kwarg and must stay out.
    assert "preferred_element_type" not in got


def test_induced_self_attention_runs_in_requested_dtype():
    """ISAB must not silently promote bf16 activations to fp32.

    The six LayerNorms were built without ``dtype``, so every pre-norm output --
    and therefore every Q/K/V and feedforward input -- was fp32 even for a bf16
    block.  Pin the attention operands and the block output dtype.
    """
    from probjax.nn.layers.attention import MultiHeadAttention

    seen = []
    original = MultiHeadAttention.__call__

    def spy(self, inputs_q, *args, **kwargs):
        seen.append(jnp.asarray(inputs_q).dtype)
        return original(self, inputs_q, *args, **kwargs)

    block = InducedSelfAttention(
        32,
        num_inducing_points=4,
        num_heads=4,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 6, 32), dtype=jnp.bfloat16)

    MultiHeadAttention.__call__ = spy
    try:
        y = block(x)
    finally:
        MultiHeadAttention.__call__ = original

    assert seen, "the induced and output MHAs must both be called"
    assert all(d == jnp.bfloat16 for d in seen), seen
    assert y.dtype == jnp.bfloat16
    # Norms carry the compute dtype but keep fp32 parameters.
    assert block.inducing_norm.dtype == jnp.bfloat16
    assert block.inducing_norm.param_dtype == jnp.float32


def test_subclass_forwards_precision_kwargs():
    blk = ConvBlock(
        8,
        8,
        norm_cls=None,
        dtype=jnp.float16,
        param_dtype=jnp.float32,
        preferred_element_type=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    assert blk.conv.dtype == jnp.float16
    assert blk.conv.param_dtype == jnp.float32
    assert blk.preferred_element_type == jnp.float32


# ---------------------------------------------------------------------------
# Conv/ConvTranspose UserWarning preservation (PR #32)
# ---------------------------------------------------------------------------


def test_conv_block_preferred_element_type_warns():
    # nnx.Conv cannot use preferred_element_type (JAX#31592): it is dropped
    # with a warning, while the layer-level cast_output still applies.
    with pytest.warns(UserWarning, match="Ignoring preferred_element_type"):
        ConvBlock(
            8,
            8,
            norm_cls=None,
            preferred_element_type=jnp.float32,
            rngs=nnx.Rngs(0),
        )


def test_unet_convtranspose_preferred_element_type_warns():
    with pytest.warns(
        UserWarning, match=r"Ignoring preferred_element_type.*ConvTranspose"
    ):
        UNet(
            32,
            [32, 64],
            preferred_element_type=jnp.float32,
            rngs=nnx.Rngs(0),
        )


def test_no_preferred_element_type_warning_when_none():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        ConvBlock(8, 8, norm_cls=None, rngs=nnx.Rngs(0))
    assert not [w for w in rec if "preferred_element_type" in str(w.message)], [
        str(w.message) for w in rec
    ]


# ---------------------------------------------------------------------------
# Regression: preferred_element_type=None must never reach x.astype(None)
# ---------------------------------------------------------------------------

_NONE_CASES = [
    (
        "ConvBlock",
        lambda: ConvBlock(8, 8, norm_cls=None, rngs=nnx.Rngs(0))(
            jnp.ones((2, 8, 8, 8))
        ),
    ),
    (
        "ResnetBlock",
        lambda: ResnetBlock(8, 8, norm_cls=None, rngs=nnx.Rngs(1))(
            jnp.ones((2, 8, 8, 8))
        ),
    ),
    (
        "ResizeConv",
        lambda: ResizeConv(8, 8, out_shape=(8, 8), rngs=nnx.Rngs(2))(
            jnp.ones((2, 8, 8, 8))
        ),
    ),
    (
        "RescaleConv",
        lambda: RescaleConv(8, 8, resize_factor=1.0, rngs=nnx.Rngs(3))(
            jnp.ones((2, 8, 8, 8))
        ),
    ),
    (
        "SpatialSelfAttention",
        lambda: SpatialSelfAttention(32, num_heads=4, rngs=nnx.Rngs(4))(
            jnp.ones((2, 8, 8, 32))
        ),
    ),
    (
        "UNet",
        lambda: UNet(32, [32, 64], rngs=nnx.Rngs(5))(jnp.ones((1, 4, 4, 32))),
    ),
    (
        "InducedSelfAttention",
        lambda: InducedSelfAttention(
            8, num_inducing_points=2, num_heads=2, rngs=nnx.Rngs(6)
        )(jnp.ones((2, 6, 8))),
    ),
    (
        "GaussianFourierEmbedding",
        lambda: GaussianFourierEmbedding(4, 8, rngs=nnx.Rngs(7))(jnp.ones((2, 4))),
    ),
]


@pytest.mark.parametrize("name,run", _NONE_CASES, ids=[c[0] for c in _NONE_CASES])
def test_none_preferred_element_type_runs(name, run):
    # Would raise TypeError on strict JAX (None passed to astype) before P1-7.
    run()


def test_explicit_preferred_element_type_still_casts():
    blk = ConvBlock(
        8,
        8,
        norm_cls=None,
        preferred_element_type=jnp.float16,
        rngs=nnx.Rngs(0),
    )
    out = blk(jnp.ones((2, 8, 8, 8), dtype=jnp.float32))
    assert out.dtype == jnp.float16

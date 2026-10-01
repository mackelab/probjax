"""Tests for probjax.nn.utils.PrecisionMixin (P0-2 / P1-7).

Covers:
- cast_output semantics (None is a strict no-op; explicit dtype casts).
- The four precision kwargs are stored and forwarded.
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
    ResnetBlock,
    ResizeConv,
    SpatialSelfAttention,
)
from probjax.nn.layers.encoding import GaussianFourierEmbedding
from probjax.nn.nets.unets import UNet
from probjax.nn.utils import PrecisionMixin


# ---------------------------------------------------------------------------
# PrecisionMixin unit behavior
# ---------------------------------------------------------------------------


def test_cast_output_none_is_noop():
    m = PrecisionMixin()
    x = jnp.ones(3)
    # Identity: astype must never be invoked with None.
    assert m.cast_output(x) is x
    assert (m.dtype, m.precision, m.param_dtype, m.preferred_element_type) == (
        None,
        None,
        None,
        None,
    )
    assert m.active_precision_kwargs() == {}


def test_cast_output_casts_when_set():
    m = PrecisionMixin(preferred_element_type=jnp.float16)
    x = jnp.ones(3, dtype=jnp.float32)
    out = m.cast_output(x)
    assert out.dtype == jnp.float16


def test_active_precision_kwargs_omits_nones():
    m = PrecisionMixin(dtype=jnp.float16, preferred_element_type=jnp.float32)
    assert m.active_precision_kwargs() == {
        "dtype": jnp.float16,
        "preferred_element_type": jnp.float32,
    }


def test_subclass_stores_precision_kwargs():
    blk = ConvBlock(
        8,
        8,
        norm_cls=None,
        dtype=jnp.float16,
        param_dtype=jnp.float32,
        preferred_element_type=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    assert isinstance(blk, PrecisionMixin)
    assert blk.dtype == jnp.float16
    assert blk.param_dtype == jnp.float32
    assert blk.preferred_element_type == jnp.float32
    assert blk.active_precision_kwargs()["dtype"] == jnp.float16


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
    with pytest.warns(UserWarning, match=r"Ignoring preferred_element_type.*ConvTranspose"):
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
    assert not [
        w for w in rec if "preferred_element_type" in str(w.message)
    ], [str(w.message) for w in rec]


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
        lambda: GaussianFourierEmbedding(4, 8, rngs=nnx.Rngs(7))(
            jnp.ones((2, 4))
        ),
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

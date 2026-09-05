"""Tests that custom_partitioning primitives survive inside custom_inverse.

The core issue: ``custom_inverse_call_p`` stores its forward jaxpr in a
``Lazy`` wrapper, which ``jax._src.core.subjaxprs`` / ``jaxprs_in_params``
cannot traverse.  JAX's ``jaxpr_has_prim_requiring_devices`` therefore misses
any ``custom_partitioning`` inside the captured jaxpr and constructs a
``ShardingContext`` with ``device_assignment=None``.  CP's lowering rule then
asserts.

The fix is to register ``custom_inverse_call_p`` in JAX's
``prim_requires_devices_during_lowering`` set so that ``device_assignment``
is always populated when this primitive appears.

Run with the ``mesh`` marker to enable multi-device CPU testing:

    pytest -m mesh tests/test_cp_in_custom_inverse.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from probjax.core import custom_inverse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_DEVICES_FOR_MESH = 2


def _have_mesh_devices() -> bool:
    return len(jax.devices()) >= _MIN_DEVICES_FOR_MESH


def _make_mesh():
    devices = np.array(jax.devices())
    return Mesh(devices, ("x",))


def _make_cp_double():
    """A simple custom_partitioning function that doubles its input."""

    @custom_partitioning
    def cp_double(x):
        return x * 2

    def infer(mesh, arg_shapes, result_shape):
        return arg_shapes[0].sharding

    def partition(mesh, arg_shapes, result_shape):
        x_sharding = arg_shapes[0].sharding
        out_sharding = result_shape.sharding

        def lower_fn(x):
            return x * 2

        return mesh, lower_fn, out_sharding, (x_sharding,)

    cp_double.def_partition(
        partition=partition,
        infer_sharding_from_operands=infer,
        sharding_rule="i -> i",
    )
    return cp_double


def _make_cp_half():
    """A simple custom_partitioning function that halves its input."""

    @custom_partitioning
    def cp_half(x):
        return x / 2

    def infer(mesh, arg_shapes, result_shape):
        return arg_shapes[0].sharding

    def partition(mesh, arg_shapes, result_shape):
        x_sharding = arg_shapes[0].sharding
        out_sharding = result_shape.sharding

        def lower_fn(x):
            return x / 2

        return mesh, lower_fn, out_sharding, (x_sharding,)

    cp_half.def_partition(
        partition=partition,
        infer_sharding_from_operands=infer,
        sharding_rule="i -> i",
    )
    return cp_half


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.mesh
class TestCPInCustomInverse:
    """custom_partitioning primitives captured inside custom_inverse jaxprs."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_devices(self):
        if not _have_mesh_devices():
            pytest.skip(
                f"Need >= {_MIN_DEVICES_FOR_MESH} devices; "
                f"run with -m mesh to enable multi-device CPU"
            )

    def test_cp_in_custom_inverse_executes(self):
        """CP function inside custom_inverse produces correct output."""
        cp_double = _make_cp_double()
        mesh = _make_mesh()
        n = len(jax.devices())

        @custom_inverse
        def f(x):
            return cp_double(x)

        f.definv(lambda y: y / 2)

        sharding = NamedSharding(mesh, P("x"))
        x = jax.device_put(jnp.arange(float(n)), sharding)

        with mesh:
            result = jax.jit(f, in_shardings=sharding, out_shardings=sharding)(x)

        expected = x * 2
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cp_in_custom_inverse_mlir_has_cp(self):
        """Lowered MLIR from custom_inverse(cp_fn) contains CustomSPMDPartitioning."""
        cp_double = _make_cp_double()
        mesh = _make_mesh()
        n = len(jax.devices())

        @custom_inverse
        def f(x):
            return cp_double(x)

        f.definv(lambda y: y / 2)

        sharding = NamedSharding(mesh, P("x"))
        x = jax.device_put(jnp.arange(float(n)), sharding)

        with mesh:
            lowered = jax.jit(f, in_shardings=sharding, out_shardings=sharding).lower(x)
            mlir_text = lowered.as_text()

        assert "CustomSPMDPartitioning" in mlir_text, (
            "Expected custom_partitioning to survive inside custom_inverse. "
            "MLIR text did not contain CustomSPMDPartitioning."
        )

    def test_cp_in_custom_inverse_inverse_still_works(self):
        """The inverse path still works when forward uses CP."""
        from probjax.core import inverse

        cp_double = _make_cp_double()

        @custom_inverse
        def f(x):
            return cp_double(x)

        f.definv(lambda y: y / 2)

        inv_f = inverse(f)
        y = jnp.array([2.0, 4.0, 6.0, 8.0])
        result = inv_f(y)
        expected = y / 2
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cp_in_custom_inverse_plain_jit(self):
        """CP inside custom_inverse works with plain jit (no explicit shardings)."""
        cp_double = _make_cp_double()

        @custom_inverse
        def f(x):
            return cp_double(x)

        f.definv(lambda y: y / 2)

        x = jnp.arange(8.0)
        result = jax.jit(f)(x)
        expected = x * 2
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cp_in_custom_inverse_logdet_mesh(self):
        """CP primitive inside definv_and_logdet, not just forward."""
        from probjax.core import inverse_and_logabsdet

        cp_double = _make_cp_double()
        cp_half = _make_cp_half()

        @custom_inverse
        def f(x):
            return cp_double(x)

        f.definv_and_logdet(
            lambda y: (cp_half(y), jnp.full_like(y, -jnp.log(jnp.array(2.0))))
        )

        mesh = _make_mesh()
        n = len(jax.devices())
        sharding = NamedSharding(mesh, P("x"))

        x = jax.device_put(jnp.arange(float(n)), sharding)

        with mesh:
            y = jax.jit(
                f, in_shardings=sharding, out_shardings=sharding
            )(x)

            inv_det = inverse_and_logabsdet(f)
            jit_inv_det = jax.jit(
                inv_det,
                in_shardings=sharding,
                out_shardings=(sharding, None),
            )
            x_rec, logdet = jit_inv_det(y)

            np.testing.assert_allclose(x_rec, x, rtol=1e-5)
            np.testing.assert_allclose(
                logdet,
                -jnp.log(jnp.array(2.0)) * x.shape[0],
                rtol=1e-5,
            )

            lowered = jit_inv_det.lower(y)
            mlir_text = lowered.as_text()
            assert "CustomSPMDPartitioning" in mlir_text, (
                "Expected custom_partitioning in inverse lowered MLIR. "
                "MLIR text did not contain CustomSPMDPartitioning."
            )

    def test_cp_in_custom_inverse_sharded_vmap(self):
        """jax.jit(jax.vmap(f)) over a sharded leading dimension."""
        from probjax.core import inverse

        cp_double = _make_cp_double()

        @custom_inverse
        def f(x):
            return cp_double(x)

        f.definv(lambda y: y / 2)

        mesh = _make_mesh()
        n = len(jax.devices())
        sharding = NamedSharding(mesh, P("x"))

        x = jax.device_put(
            jnp.arange(2.0 * n).reshape(n, 2), sharding
        )

        with mesh:
            vmapped_f = jax.jit(
                jax.vmap(f, in_axes=0, out_axes=0),
                in_shardings=sharding,
                out_shardings=sharding,
            )
            y = vmapped_f(x)

            inv_vmap = inverse(jax.vmap(f, in_axes=0, out_axes=0))
            jit_inv = jax.jit(
                inv_vmap,
                in_shardings=sharding,
                out_shardings=sharding,
            )
            x_rec = jit_inv(y)

            np.testing.assert_allclose(x_rec, x, rtol=1e-5)
            assert x_rec.sharding.is_equivalent_to(sharding, ndim=2), (
                "Inverse output sharding should match input sharding."
            )


class TestDeviceRegistration:
    """Verify custom_inverse_call_p is registered for device-requiring lowering."""

    def test_primitive_in_requires_devices_set(self):
        from jax._src.dispatch import prim_requires_devices_during_lowering
        from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p

        assert custom_inverse_call_p in prim_requires_devices_during_lowering

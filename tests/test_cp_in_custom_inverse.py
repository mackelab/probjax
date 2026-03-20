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


class TestDeviceRegistration:
    """Verify custom_inverse_call_p is registered for device-requiring lowering."""

    def test_primitive_in_requires_devices_set(self):
        from jax._src.dispatch import prim_requires_devices_during_lowering
        from probjax.core.custom_primitives.custom_inverse import custom_inverse_call_p

        assert custom_inverse_call_p in prim_requires_devices_during_lowering

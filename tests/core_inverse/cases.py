"""Declarative test-case registry for inverse interpreter primitives.

This module defines compact, immutable case dataclasses that describe:
- Round-trip expectations for bijective / principal-branch primitives
- Runtime-invalid output handling (NaN for inexact, checkify for exact)
- Structural non-injectivity (e.g. broadcast, gather, slice)
- Log-determinant formulas and expected values

The registries are consumed by parameterized tests in the sibling test_*.py
modules, plus a meta-test that ensures every registered inverse rule has
at least one declarative case or an explicit specialization entry.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from probjax.core.registry import Context, REGISTRY

# ---------------------------------------------------------------------------
# Case dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AtomicCase:
    """A primitive that is either bijective or has a defined principal inverse."""

    id: str
    primitive: jax.extend.core.Primitive
    forward: Callable[[Any], Any]
    args_factory: Callable[[], Any]
    inverse_kind: Literal["bijective", "principal"] = "bijective"


@dataclasses.dataclass(frozen=True)
class InvalidOutputCase:
    """A primitive whose inverse exists but may receive runtime-invalid output."""

    id: str
    primitive: jax.extend.core.Primitive
    forward: Callable[[Any], Any]
    output_factory: Callable[[], Any]
    expected_factory: Callable[[], Any]


@dataclasses.dataclass(frozen=True)
class StructuralInvalidCase:
    """A structurally non-injective primitive with no unique inverse."""

    id: str
    primitive: jax.extend.core.Primitive
    forward: Callable[[Any], Any]
    args_factory: Callable[[], Any]


@dataclasses.dataclass(frozen=True)
class ManualCoverage:
    primitive: jax.extend.core.Primitive
    context: str
    module: str
    test_name: str


# ---------------------------------------------------------------------------
# Atomic round-trip cases
# ---------------------------------------------------------------------------

ATOMIC_ROUNDTRIP_CASES: list[AtomicCase] = [
    # Trigonometric / inverse trig (principal branches)
    AtomicCase("sin", jax.lax.sin_p, jnp.sin, lambda: jnp.linspace(-1.0, 1.0, 7)),
    AtomicCase("asin", jax.lax.asin_p, jnp.arcsin, lambda: jnp.linspace(-0.9, 0.9, 7)),
    AtomicCase("cos", jax.lax.cos_p, jnp.cos, lambda: jnp.linspace(0.2, 2.9, 7)),
    AtomicCase("acos", jax.lax.acos_p, jnp.arccos, lambda: jnp.linspace(-0.9, 0.9, 7)),
    AtomicCase("tan", jax.lax.tan_p, jnp.tan, lambda: jnp.linspace(-1.0, 1.0, 7)),
    AtomicCase("atan", jax.lax.atan_p, jnp.arctan, lambda: jnp.linspace(-3.0, 3.0, 7)),
    # Hyperbolic
    AtomicCase("tanh", jax.lax.tanh_p, jnp.tanh, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("sinh", jax.lax.sinh_p, jnp.sinh, lambda: jnp.linspace(-0.4, 0.4, 7)),
    AtomicCase("asinh", jax.lax.asinh_p, jnp.arcsinh, lambda: jnp.linspace(-0.5, 0.5, 7)),
    # cosh is even: moved to principal branch cases
    AtomicCase("acosh", jax.lax.acosh_p, lambda x: jax.lax.acosh_p.bind(x), lambda: jnp.linspace(1.1, 2.0, 7)),
    AtomicCase("atanh", jax.lax.atanh_p, jnp.arctanh, lambda: jnp.linspace(-0.5, 0.5, 7)),
    # Exponential / logarithmic
    AtomicCase("exp", jax.lax.exp_p, jnp.exp, lambda: jnp.linspace(-1.0, 1.0, 7)),
    AtomicCase("exp2", jax.lax.exp2_p, jnp.exp2, lambda: jnp.linspace(-1.0, 1.0, 7)),
    AtomicCase("log", jax.lax.log_p, jnp.log, lambda: jnp.linspace(0.1, 2.0, 7)),
    AtomicCase("log1p", jax.lax.log1p_p, jnp.log1p, lambda: jnp.linspace(0.05, 0.95, 7)),
    AtomicCase("expm1", jax.lax.expm1_p, jnp.expm1, lambda: jnp.linspace(-0.9, 0.9, 7)),
    # Roots / powers
    AtomicCase("sqrt", jax.lax.sqrt_p, jnp.sqrt, lambda: jnp.linspace(0.25, 2.0, 7)),
    AtomicCase("rsqrt", jax.lax.rsqrt_p, lambda x: jax.lax.rsqrt_p.bind(x), lambda: jnp.linspace(0.25, 2.0, 7)),
    AtomicCase("cbrt", jax.lax.cbrt_p, jnp.cbrt, lambda: jnp.linspace(-8.0, 8.0, 7)),
    # Linear
    AtomicCase("neg", jax.lax.neg_p, jnp.negative, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("copy", jax.lax.copy_p, jnp.copy, lambda: jnp.linspace(-0.5, 0.5, 7)),
    # Special
    AtomicCase("erf", jax.lax.erf_p, jax.lax.erf, lambda: jnp.linspace(-1.0, 1.0, 7)),
    AtomicCase("erf_inv", jax.lax.erf_inv_p, jax.lax.erf_inv, lambda: jnp.linspace(-0.9, 0.9, 7)),
    # Complex
    AtomicCase("conj", jax.lax.conj_p, jnp.conj, lambda: jnp.array([1.0 + 2.0j, 3.0 - 1.0j])),
    AtomicCase("logistic", jax.lax.logistic_p, jax.lax.logistic, lambda: jnp.linspace(-2.0, 2.0, 7)),
    # Bivariate (tested via binary tests)
    AtomicCase("add", jax.lax.add_p, lambda x: x + 1.0, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("sub", jax.lax.sub_p, lambda x: x - 1.0, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("mul", jax.lax.mul_p, lambda x: x * 2.0, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("div", jax.lax.div_p, lambda x: x / 2.0, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("pow", jax.lax.pow_p, lambda x: jnp.power(x, 3.0), lambda: jnp.linspace(0.1, 1.0, 7)),
]

# ---------------------------------------------------------------------------
# Principal-branch-only cases
# ---------------------------------------------------------------------------

PRINCIPAL_BRANCH_CASES: list[AtomicCase] = [
    # Even powers: principal sqrt branch, so negative inputs map to positive.
    AtomicCase(
        "integer_pow_even",
        jax.lax.integer_pow_p,
        lambda x: x ** 2,
        lambda: jnp.array([-2.0, 3.0]),
        inverse_kind="principal",
    ),
    # cosh is even: acosh returns the non-negative branch.
    AtomicCase(
        "cosh",
        jax.lax.cosh_p,
        jnp.cosh,
        lambda: jnp.linspace(-0.4, 0.4, 7),
        inverse_kind="principal",
    ),
]

# ---------------------------------------------------------------------------
# Invalid-output cases (runtime values outside the image)
# ---------------------------------------------------------------------------

INVALID_OUTPUT_CASES: list[InvalidOutputCase] = [
    # sqrt: negative output has no real preimage
    InvalidOutputCase(
        "sqrt_negative",
        jax.lax.sqrt_p,
        jnp.sqrt,
        lambda: jnp.array([-1.0, 2.0]),
        lambda: jnp.array([jnp.nan, 4.0]),
    ),
    # tanh: output outside (-1, 1) has no real preimage
    InvalidOutputCase(
        "tanh_out_of_range",
        jax.lax.tanh_p,
        jnp.tanh,
        lambda: jnp.array([1.1, -1.5]),
        lambda: jnp.array([jnp.nan, jnp.nan]),
    ),
    # logistic (sigmoid): output outside (0, 1)
    InvalidOutputCase(
        "logistic_out_of_range",
        jax.lax.logistic_p,
        jax.lax.logistic,
        lambda: jnp.array([-0.1, 1.2]),
        lambda: jnp.array([jnp.nan, jnp.nan]),
    ),
    # asin has image [-pi/2, pi/2].
    InvalidOutputCase(
        "asin_out_of_range",
        jax.lax.asin_p,
        jnp.arcsin,
        lambda: jnp.array([1.5, -2.0]),
        lambda: jnp.array([jnp.sin(1.5), jnp.nan]),
    ),
    # acos has image [0, pi].
    InvalidOutputCase(
        "acos_out_of_range",
        jax.lax.acos_p,
        jnp.arccos,
        lambda: jnp.array([1.5, -2.0]),
        lambda: jnp.array([jnp.cos(1.5), jnp.nan]),
    ),
]

# ---------------------------------------------------------------------------
# Structurally non-injective reject cases
# ---------------------------------------------------------------------------

STRUCTURALLY_INVALID_CASES: list[StructuralInvalidCase] = [
    # Integer pow 0: every x maps to 1 -> completely non-injective.
    StructuralInvalidCase(
        "integer_pow_zero",
        jax.lax.integer_pow_p,
        lambda x: x**0,
        lambda: jnp.array([2.0, -3.0, 4.0]),
    ),
    # real() discards imaginary component.
    StructuralInvalidCase(
        "real_discards_imag",
        jax.lax.real_p,
        jnp.real,
        lambda: jnp.array([1.0 + 2.0j, 3.0 - 1.0j]),
    ),
    # imag() discards real component.
    StructuralInvalidCase(
        "imag_discards_real",
        jax.lax.imag_p,
        jnp.imag,
        lambda: jnp.array([1.0 + 2.0j, 3.0 - 1.0j]),
    ),
]

# ---------------------------------------------------------------------------
# Logdet explicit cases (formula-checked against autodiff)
# ---------------------------------------------------------------------------

LOGDET_EXPLICIT_CASES: list[AtomicCase] = [
    AtomicCase("exp", jax.lax.exp_p, jnp.exp, lambda: jnp.linspace(-1.0, 1.0, 7)),
    AtomicCase("log", jax.lax.log_p, jnp.log, lambda: jnp.linspace(0.1, 2.0, 7)),
    AtomicCase("neg", jax.lax.neg_p, jnp.negative, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("copy", jax.lax.copy_p, jnp.copy, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("sqrt", jax.lax.sqrt_p, jnp.sqrt, lambda: jnp.linspace(0.25, 2.0, 7)),
    AtomicCase("cbrt", jax.lax.cbrt_p, jnp.cbrt, lambda: jnp.linspace(-8.0, 8.0, 7)),
    AtomicCase("tanh", jax.lax.tanh_p, jnp.tanh, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("log1p", jax.lax.log1p_p, jnp.log1p, lambda: jnp.linspace(0.05, 0.95, 7)),
    AtomicCase("expm1", jax.lax.expm1_p, jnp.expm1, lambda: jnp.linspace(-0.9, 0.9, 7)),
    AtomicCase("logistic", jax.lax.logistic_p, jax.lax.logistic, lambda: jnp.linspace(-2.0, 2.0, 7)),
    # Bivariate
    AtomicCase("add", jax.lax.add_p, lambda x: x + 1.0, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("sub", jax.lax.sub_p, lambda x: x - 1.0, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("mul", jax.lax.mul_p, lambda x: x * 2.0, lambda: jnp.linspace(-0.5, 0.5, 7)),
    AtomicCase("div", jax.lax.div_p, lambda x: x / 2.0, lambda: jnp.linspace(-0.5, 0.5, 7)),
]


MANUAL_COVERAGE: list[ManualCoverage] = [
    ManualCoverage(jax.lax.dot_general_p, Context.INVERSE, "dot", "test_inverse_dot_general_lhs"),
    ManualCoverage(jax.lax.concatenate_p, Context.INVERSE, "array", "test_inverse_split"),
    ManualCoverage(jax.lax.squeeze_p, Context.INVERSE, "array", "test_inverse_squeeze_broadcast"),
    ManualCoverage(jax.lax.broadcast_in_dim_p, Context.INVERSE, "array", "test_inverse_broadcast_scalar_to_vector_consistent"),
    ManualCoverage(jax.lax.rev_p, Context.INVERSE, "array", "test_inverse_rev_reshape"),
    ManualCoverage(jax.lax.gather_p, Context.INVERSE, "array", "test_inverse_gather_permutation"),
    ManualCoverage(jax.lax.scatter_p, Context.INVERSE, "array", "test_inverse_scatter_overwrite_nonconstant"),
    ManualCoverage(jax.lax.select_n_p, Context.INVERSE, "partial", "test_inverse_uniform_select_n"),
    ManualCoverage(jax.lax.reshape_p, Context.INVERSE, "array", "test_inverse_rev_reshape"),
    ManualCoverage(jax.lax.convert_element_type_p, Context.INVERSE, "array", "test_inverse_convert_element_type_widen"),
    ManualCoverage(jax.lax.bitcast_convert_type_p, Context.INVERSE, "array", "test_inverse_bitcast_convert_type"),
    ManualCoverage(jax.lax.transpose_p, Context.INVERSE, "array", "test_inverse_transpose"),
    ManualCoverage(jax.lax.slice_p, Context.INVERSE, "array", "test_inverse_strided_slice"),
    ManualCoverage(jax.lax.dynamic_slice_p, Context.INVERSE, "partial", "test_inverse_disjoint_slices_scheduling"),
    ManualCoverage(jax.lax.split_p, Context.INVERSE, "array", "test_inverse_split"),
    ManualCoverage(jax.lax.cond_p, Context.INVERSE, "control", "test_inverse_cond_with_known_branch"),
    ManualCoverage(jax.lax.scan_p, Context.INVERSE, "control", "test_inverse_scan_carry_only"),
    ManualCoverage(jax.lax.while_p, Context.INVERSE, "control", "test_inverse_while_rule_level"),
    ManualCoverage(jax.lax.split_p, Context.INVERSE_LOGDET, "array", "test_inverse_and_logabsdet_split_rule"),
    ManualCoverage(jax.lax.rev_p, Context.INVERSE_LOGDET, "array", "test_logabsdet_nested_jit_flip"),
    ManualCoverage(jax.lax.squeeze_p, Context.INVERSE_LOGDET, "array", "test_inverse_and_logabsdet_squeeze_rule"),
    ManualCoverage(jax.lax.transpose_p, Context.INVERSE_LOGDET, "array", "test_logabsdet_transpose_chain"),
    ManualCoverage(jax.lax.gather_p, Context.INVERSE_LOGDET, "array", "test_logabsdet_gather_permutation"),
    ManualCoverage(jax.lax.broadcast_in_dim_p, Context.INVERSE_LOGDET, "array", "test_inverse_and_logabsdet_broadcast_is_undefined"),
    ManualCoverage(jax.lax.dot_general_p, Context.INVERSE_LOGDET, "dot", "test_inverse_and_logabsdet_dot_general_lhs"),
    ManualCoverage(jax.lax.cond_p, Context.INVERSE_LOGDET, "control", "test_inverse_and_logabsdet_cond_with_known_branch"),
    ManualCoverage(jax.lax.scan_p, Context.INVERSE_LOGDET, "control", "test_inverse_and_logabsdet_scan_carry_only"),
    ManualCoverage(jax.lax.while_p, Context.INVERSE_LOGDET, "control", "test_inverse_and_logabsdet_while_rule_level"),
]


# ---------------------------------------------------------------------------
# Coverage helpers
# ---------------------------------------------------------------------------


def registered_inverse_primitives() -> set[jax.extend.core.Primitive]:
    """Return the set of primitives with INVERSE rules in the global registry."""
    return set(REGISTRY.list_primitives(Context.INVERSE))


def registered_logdet_primitives() -> set[jax.extend.core.Primitive]:
    """Return the set of primitives with INVERSE_LOGDET rules."""
    return set(REGISTRY.list_primitives(Context.INVERSE_LOGDET))

"""Inverting the affine programs that equation-by-equation propagation cannot.

The inverse interpreter resolves one equation at a time, so it inverts a *tree*
of operations. A variable used twice breaks that: in ``3 * x - x`` the ``sub``
has two unknown operands, because both depend on the ``x`` being solved for, and
the bivariate rules need exactly one unknown. Propagation stalls and the target
comes back NaN.

That is a limitation of local propagation, not of the problem. When a stalled
program or section is **affine** in its input -- ``y = A x + b`` -- its inverse
is a linear solve, and both ``A`` and ``b`` can be extracted exactly:

    b = f(0)                      the constant term
    A = jacfwd(f)(0)              constant, because f is affine
    x = solve(A, y - b)
    log|d(inv)/dy| = -log|det A|

Affinity is decided **structurally**, from the jaxpr, so it is a proof rather
than a numerical guess: a program that samples as affine at a few points is not
necessarily affine. The recovery interpreter uses these solvers on a stall,
then resumes propagation through registered nonlinear inverses. Whole-program
recovery remains available for coupled multiple-leaf affine inputs.

Nonlinear fan-out (``x + tanh(x)``, a residual block) stays unsupported. It is
invertible by fixed-point iteration when the residual branch is a contraction,
but nothing in a jaxpr states a Lipschitz bound, so the guarantee has to come
from the author. Register such a map with
:class:`~probjax.core.custom_inverse` instead.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax.extend.core import Jaxpr, Var

from probjax.core.interpreters.inverse.diagonal import DiagonalAffineSystem

__all__ = ["is_affine_in", "solve_affine_inverse"]


#: Primitives that are affine in every operand, so they can never introduce a
#: nonlinearity in the target no matter which operand carries it.
_AFFINE_PRIMITIVES = frozenset({
    "add",
    "add_any",
    "sub",
    "neg",
    "reduce_sum",
    "cumsum",
    "reshape",
    "transpose",
    "squeeze",
    "expand_dims",
    "broadcast_in_dim",
    "concatenate",
    "slice",
    "dynamic_slice",
    "rev",
    "pad",
    "gather",
    "convert_element_type",
    "copy",
    "select_n",
})

#: Primitives that are affine in *one* operand while the others are constant.
#: ``x * x`` is not affine, so a tainted operand count above one disqualifies.
_BILINEAR_PRIMITIVES = frozenset({
    "mul",
    "div",
    "dot_general",
    "conv_general_dilated",
})


def _tainted_operands(eqn, tainted: set) -> list:
    return [v for v in eqn.invars if isinstance(v, Var) and v in tainted]


def affine_equation(eqn, dependent: Sequence[bool]) -> bool:
    """Prove affinity in data operands, never in indices or predicates."""
    if eqn.effects:
        return False
    if not any(dependent):
        return True
    name = eqn.primitive.name
    if name == "div":
        return not dependent[1]
    if name in {"dynamic_slice", "gather"}:
        return not any(dependent[1:])
    if name == "select_n":
        return not dependent[0]
    if name == "convert_element_type":
        # Narrowing, integer conversion, and complex-to-real casts are lossy.
        source = jnp.dtype(eqn.invars[0].aval.dtype)
        target = jnp.dtype(eqn.outvars[0].aval.dtype)
        return (
            jnp.issubdtype(source, jnp.inexact)
            and jnp.issubdtype(target, jnp.inexact)
            and jnp.can_cast(source, target, casting="safe")
            and source.kind == target.kind
            and jnp.finfo(target).bits >= jnp.finfo(source).bits
        )
    if name in _BILINEAR_PRIMITIVES:
        return sum(dependent) == 1
    return name in _AFFINE_PRIMITIVES


def is_affine_in(jaxpr: Jaxpr, target_vars: Sequence[Var]) -> bool:
    """Whether ``jaxpr`` is affine in ``target_vars``, proven from its structure.

    Walks forward, tracking which variables depend on the target. An equation
    whose inputs are all independent of the target is a constant subgraph and is
    ignored however nonlinear it is -- ``exp(c) * x`` is affine in ``x``.
    """
    tainted = set(target_vars)
    for eqn in jaxpr.eqns:
        touching = _tainted_operands(eqn, tainted)
        if not touching:
            continue
        if not affine_equation(
            eqn, [isinstance(v, Var) and v in tainted for v in eqn.invars]
        ):
            return False
        tainted.update(v for v in eqn.outvars if isinstance(v, Var))
    return True


def _flat_size(var: Var) -> int:
    # math.prod, not jnp: a shape is a compile-time quantity, and inside an
    # active trace even `jnp.prod` of a literal tuple is staged out, which then
    # fails the `int()` with a ConcretizationTypeError under jit.
    return math.prod(var.aval.shape)


def solve_affine_inverse(
    jaxpr: Jaxpr,
    consts: Sequence[Any],
    target_vars: Sequence[Var],
    known: dict,
    outputs: Sequence[Any],
    *,
    compute_logdet: bool = True,
) -> tuple[list, Any] | None:
    """Invert an affine ``jaxpr`` for ``target_vars`` by a linear solve.

    Args:
        jaxpr: the traced forward program.
        consts: its constants.
        target_vars: the input variables being solved for.
        known: values for every other input variable.
        outputs: the values of ``jaxpr.outvars`` to invert.
        compute_logdet: whether to emit a determinant calculation. False for
            value-only inversion.

    Returns:
        ``(values, log_abs_det)`` where ``values`` aligns with ``target_vars``
        and ``log_abs_det`` is that of the *inverse* map, or None if the program
        is not affine in the target or the system is not square.

    The Jacobian is materialised, which costs O(n^2) in the target's size. That
    is acceptable here because this path runs only where the alternative is
    NaN, but it does mean a very large affine fan-out is better served by a
    hand-written :class:`~probjax.core.custom_inverse`.
    """
    if not target_vars or not is_affine_in(jaxpr, target_vars):
        return None

    if sum(_flat_size(v) for v in target_vars) != sum(jnp.size(o) for o in outputs):
        return None
    return AffineSystem(jaxpr, target_vars).solve(
        consts, known, outputs, compute_logdet=compute_logdet
    )


class AffineSystem:
    """Reusable evaluator for a structurally proven, square affine section.

    Only graph/shape metadata is retained. Constants and coefficients are
    explicit arguments so a cached evaluator never captures runtime tracers.
    """

    def __init__(self, jaxpr: Jaxpr, target_vars: Sequence[Var]):
        self.targets = tuple(target_vars)
        self.shapes = tuple(v.aval.shape for v in self.targets)
        self.sizes = tuple(_flat_size(v) for v in self.targets)
        self.size = sum(self.sizes)
        self.dtype = jnp.result_type(*[v.aval.dtype for v in self.targets])
        targets = set(self.targets)
        self.known_vars = tuple(v for v in jaxpr.invars if v not in targets)
        self.diagonal = DiagonalAffineSystem.try_build(jaxpr, self.targets)
        if self.diagonal is not None:
            return

        def evaluate(flat_target, consts, known_values):
            environment = dict(zip(self.known_vars, known_values, strict=True))
            offset = 0
            for var, shape, size in zip(
                self.targets, self.shapes, self.sizes, strict=True
            ):
                environment[var] = flat_target[offset : offset + size].reshape(shape)
                offset += size
            result = jax_core.eval_jaxpr(
                jaxpr, consts, *[environment[v] for v in jaxpr.invars]
            )
            return jnp.concatenate([jnp.asarray(r).reshape(-1) for r in result])

        self.evaluate = evaluate
        self.jacobian = jax.jacfwd(evaluate, argnums=0)

    def solve(self, consts, known, outputs, *, compute_logdet):
        if self.diagonal is not None:
            return self.diagonal.solve(
                consts, known, outputs, compute_logdet=compute_logdet
            )
        known_values = tuple(known[v] for v in self.known_vars)
        zero = jnp.zeros((self.size,), self.dtype)
        constant = self.evaluate(zero, consts, known_values)
        matrix = self.jacobian(zero, consts, known_values)
        rhs = jnp.concatenate([jnp.asarray(o).reshape(-1) for o in outputs]) - constant
        solution = jnp.linalg.solve(matrix, rhs)
        logdet = -jnp.linalg.slogdet(matrix)[1] if compute_logdet else None
        values, offset = [], 0
        for shape, size in zip(self.shapes, self.sizes, strict=True):
            values.append(solution[offset : offset + size].reshape(shape))
            offset += size
        return values, logdet

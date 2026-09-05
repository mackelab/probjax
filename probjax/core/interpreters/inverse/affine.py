"""Inverting the affine programs that equation-by-equation propagation cannot.

The inverse interpreter resolves one equation at a time, so it inverts a *tree*
of operations. A variable used twice breaks that: in ``3 * x - x`` the ``sub``
has two unknown operands, because both depend on the ``x`` being solved for, and
the bivariate rules need exactly one unknown. Propagation stalls and the target
comes back NaN.

That is a limitation of local propagation, not of the problem. When the stalled
program is **affine** in the target -- ``y = A x + b`` -- the inverse is a linear
solve. ``b`` is the program evaluated at zero and ``A`` is its linear part,
``A v = f(v) - b``; both are staged as ordinary JAX code, so the whole fallback
traces under ``jit``/``vmap`` with no Python control flow on traced values.

Three tiers keep the common cases cheap:

* elementwise maps (``x + x``, ``3 * x - x``) -- one extra forward evaluation
  with all ones gives the diagonal, so the inverse is ``(y - b) / d`` in O(n);
* small coupled maps (``sum(x) - x``) -- the matrix is built with a single
  ``vmap`` over the basis and solved directly (exact, needs O(n^2) memory);
* large coupled maps -- a matrix-free ``gmres`` solve using only matvecs, so
  memory stays O(n). The log-determinant needs the dense matrix, so that tier
  returns a NaN log-det alongside a working inverse.

Affinity is decided **structurally**, from the jaxpr, so it is a proof rather
than a numerical guess: a program that samples as affine at a few points is not
necessarily affine. This runs only as a fallback after propagation has failed,
so it can never change an answer the interpreter already produced -- it only
replaces NaN with a value.

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
import numpy as np
from jax._src import core as jax_core
from jax.extend.core import Jaxpr, Literal, Var

from probjax.core.registry import apply_inverse_guard, inverse_roundtrip_valid

__all__ = ["is_affine_in", "is_volume_preserving", "solve_affine_inverse"]


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
    "rev",
    "pad",
    "tile",
    "convert_element_type",
    "copy",
})

#: Primitives that are affine in *one* operand while the others are constant.
#: ``x * x`` is not affine, so a tainted operand count above one disqualifies.
_BILINEAR_PRIMITIVES = frozenset({
    "mul",
    "dot_general",
    "conv_general_dilated",
})

# These are affine only when target dependence is confined to the listed
# operands. A target-dependent divisor, index, or selector is nonlinear.
_DATA_OPERAND_ONLY_PRIMITIVES = frozenset({"div", "dynamic_slice", "gather"})

#: Pointwise primitives: with a single target variable these map each output
#: element from the same-index input element, so the linear part is diagonal.
#: ``mul``/``div`` qualify only single-tainted (constant other side), checked
#: in the walk below rather than by membership alone.
_ELEMENTWISE_PRIMITIVES = frozenset({
    "add",
    "add_any",
    "sub",
    "neg",
    "convert_element_type",
    "copy",
})

#: Above this many target elements the dense matrix no longer fits the
#: "cheap fallback" brief, so coupled maps switch to a matrix-free solve.
_DENSE_THRESHOLD = 512


def _tainted_operands(eqn, tainted: set) -> list:
    return [i for i, v in enumerate(eqn.invars) if isinstance(v, Var) and v in tainted]


def _closed_sub_jaxpr(eqn):
    """A nested program carried by ``eqn``, mirroring the engine's descent.

    Same policy as the propagation engine: ``scan`` is excluded (carry
    semantics need more than inlining), anything without a ``jaxpr`` /
    ``call_jaxpr`` parameter (``cond`` branches, ``while_loop`` bodies) is
    left to the default-deny below.
    """
    if eqn.primitive.name == "scan":
        return None
    for key in ("jaxpr", "call_jaxpr"):
        candidate = eqn.params.get(key)
        if (
            candidate is not None
            and hasattr(candidate, "jaxpr")
            and hasattr(candidate, "consts")
        ):
            return candidate
    return None


def _mapped_sub_targets(eqn, sub, tainted: set) -> list | None:
    """Map tainted outer operands to the nested jaxpr's input variables.

    Call-like primitives bind subjaxpr invars positionally to *all* outer
    invars -- a closed-over constant arrives as an outer ``Literal`` yet
    still consumes a subjaxpr invar (e.g. ``pad``'s edge value), so literals
    participate in the alignment but never in the taint. Returns None when
    the correspondence cannot be established; callers must decline.
    """
    if len(eqn.invars) != len(sub.jaxpr.invars):
        return None
    return [
        sub.jaxpr.invars[k]
        for k, v in enumerate(eqn.invars)
        if isinstance(v, Var) and v in tainted
    ]


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
        sub = _closed_sub_jaxpr(eqn)
        if sub is not None:
            # A jitted (or otherwise nested) subprogram is affine exactly when
            # the subprogram is affine in the operands carrying the target.
            mapped = _mapped_sub_targets(eqn, sub, tainted)
            if mapped is None or not is_affine_in(sub.jaxpr, mapped):
                return False
            tainted.update(v for v in eqn.outvars if isinstance(v, Var))
            continue
        name = eqn.primitive.name
        if name in _AFFINE_PRIMITIVES:
            # When inverse() only receives a scalar reduction output, tracing
            # loses the original input shape and produces reduce_sum[axes=()].
            # Treating that apparent identity as invertible would invent a
            # scalar inverse for an underdetermined vector reduction.
            if name == "reduce_sum" and not eqn.params["axes"]:
                return False
        elif name in _BILINEAR_PRIMITIVES:
            if len(touching) > 1:
                return False  # e.g. x * x
        elif name in _DATA_OPERAND_ONLY_PRIMITIVES:
            if any(index != 0 for index in touching):
                return False
        elif name == "select_n":
            if 0 in touching:
                return False
        else:
            return False
        tainted.update(v for v in eqn.outvars if isinstance(v, Var))
    return True


def _is_elementwise_affine_in(jaxpr: Jaxpr, target_vars: Sequence[Var]) -> bool:
    """Whether the map is pointwise in a single target variable.

    Pointwise ops on one variable keep each output element a function of the
    same-index input element, so ``A`` is diagonal and one forward evaluation
    with all ones reads it off. Anything that moves data across indices
    (reshape, transpose, broadcast, concatenate, slice, gather, reductions,
    selection) disqualifies, as does any second target leaf, whose cross-talk
    a bare op-name check cannot see.
    """
    if len(target_vars) != 1:
        return False
    tainted = set(target_vars)
    for eqn in jaxpr.eqns:
        touching = _tainted_operands(eqn, tainted)
        if not touching:
            continue
        sub = _closed_sub_jaxpr(eqn)
        if sub is not None:
            mapped = _mapped_sub_targets(eqn, sub, tainted)
            if mapped is None or not _is_elementwise_affine_in(sub.jaxpr, mapped):
                return False
            tainted.update(v for v in eqn.outvars if isinstance(v, Var))
            continue
        name = eqn.primitive.name
        if name in _ELEMENTWISE_PRIMITIVES:
            pass
        elif name == "mul":
            if len(touching) > 1:
                return False
        elif name == "div":
            if touching != [0]:
                return False
        else:
            return False
        tainted.update(v for v in eqn.outvars if isinstance(v, Var))
    return True


#: Rearrangements that only move elements around: volume-preserving whenever
#: the element count is unchanged (checked statically per equation below).
_VOLUME_REARRANGEMENTS = frozenset({
    "reshape",
    "transpose",
    "squeeze",
    "expand_dims",
    "rev",
})

#: Pointwise unit-Jacobian ops: each output element is ± the same-index input
#: element, so |det| = 1. `convert_element_type` only changes dtype, and the
#: slow path already reports it as a zero-contribution rearrangement.
_VOLUME_POINTWISE = frozenset({
    "neg",
    "conj",
    "copy",
    "convert_element_type",
})


def _is_unit_literal(var) -> bool:
    """Whether ``var`` is a compile-time literal with |value| == 1 everywhere."""
    if not isinstance(var, Literal):
        return False
    try:
        magnitude = np.abs(np.asarray(var.val))
    except Exception:
        return False
    return bool(np.all(magnitude == 1))


def is_volume_preserving(jaxpr: Jaxpr, target_vars: Sequence[Var]) -> bool:
    """Whether ``jaxpr`` has |det J| = 1 in ``target_vars``, proven structurally.

    Same taint-tracking pattern as :func:`is_affine_in`: equations whose inputs
    are all independent of the target are constant subgraphs and are ignored.
    Every tainted equation must be a size-preserving rearrangement or a
    pointwise unit-Jacobian op:

    * ``add``/``sub`` (and ``add_any``) with exactly one tainted operand -- a
      translation. Two tainted operands (``x + x``) scale and disqualify.
    * ``mul`` single-tainted with a ``±1`` literal other side; ``div`` with a
      tainted numerator and a ``±1`` literal divisor. Anything else scales.
    * ``neg``, ``conj``, ``copy``, ``convert_element_type`` unconditionally.

    Deliberately excluded (they keep the current, already-correct slow path):
    size-changing ops (``slice``, ``concatenate``, ``broadcast_in_dim``,
    ``gather``, ``dynamic_slice``, ``reduce_sum``), data-dependent index or
    divisor positions, ``select_n`` (mixing two rearrangements can duplicate
    coordinates), ``±1`` held in tracers rather than literals, and control flow.
    The check is conservative: it may say False for a map that happens to
    preserve volume, but it never says True for one that does not.
    """
    # Tracking only whether a value is tainted cannot prove that multiple
    # targets are represented exactly once in the outputs. Keep this shortcut
    # conservative and let the general log-determinant path handle joint maps.
    if len(target_vars) != 1:
        return False
    tainted = set(target_vars)
    for eqn in jaxpr.eqns:
        touching = _tainted_operands(eqn, tainted)
        if not touching:
            continue
        sub = _closed_sub_jaxpr(eqn)
        if sub is not None:
            # A nested subprogram preserves volume exactly when it does so
            # in the operands carrying the target; every tainted variable
            # stays of the form ±target + const across the boundary.
            mapped = _mapped_sub_targets(eqn, sub, tainted)
            if mapped is None or not is_volume_preserving(sub.jaxpr, mapped):
                return False
            tainted.update(v for v in eqn.outvars if isinstance(v, Var))
            continue
        if len(eqn.outvars) != 1:
            return False
        out_size = math.prod(eqn.outvars[0].aval.shape)
        name = eqn.primitive.name
        if name in _VOLUME_REARRANGEMENTS or name in _VOLUME_POINTWISE:
            if any(math.prod(eqn.invars[i].aval.shape) != out_size for i in touching):
                return False
        elif name in ("add", "add_any", "sub"):
            if len(touching) > 1:
                return False  # e.g. x + x scales by 2
            if math.prod(eqn.invars[touching[0]].aval.shape) != out_size:
                return False  # broadcasting a scalar target is not square
        elif name == "mul":
            if len(touching) > 1 or len(eqn.invars) != 2:
                return False
            if not _is_unit_literal(eqn.invars[1 - touching[0]]):
                return False
            if math.prod(eqn.invars[touching[0]].aval.shape) != out_size:
                return False
        elif name == "div":
            if touching != [0]:
                return False
            if not _is_unit_literal(eqn.invars[1]):
                return False
            if math.prod(eqn.invars[0].aval.shape) != out_size:
                return False
        else:
            return False
        tainted.update(v for v in eqn.outvars if isinstance(v, Var))
    output_vars = jaxpr.outvars
    return all(isinstance(var, Var) and var in tainted for var in output_vars) and sum(
        math.prod(var.aval.shape) for var in output_vars
    ) == math.prod(target_vars[0].aval.shape)


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
    need_logdet: bool = True,
) -> tuple[list, Any] | None:
    """Invert an affine ``jaxpr`` for ``target_vars`` by a linear solve.

    Args:
        jaxpr: the traced forward program.
        consts: its constants.
        target_vars: the input variables being solved for.
        known: values for every other input variable.
        outputs: the values of ``jaxpr.outvars`` to invert.

    Returns:
        ``(values, log_abs_det)`` where ``values`` aligns with ``target_vars``
        and ``log_abs_det`` is that of the *inverse* map, or None if the program
        is not affine in the target, is underdetermined, or is too large.

    Overdetermined but consistent systems (``tile``, ``concat([x, x])``,
    padding -- more outputs than inputs) are solved by least squares with a
    replay guard: an inconsistent output reports NaN rather than a value.
    There is no square Jacobian there, so the log-det comes back NaN by
    design, exactly like the large coupled tier below.

    Everything below is staged JAX -- no Python branching on traced values --
    so the result traces under ``jit``/``vmap``. The per-call Python work is a
    structural walk plus a fixed handful of forward evaluations; nothing scales
    with the target size in Python. A very large *coupled* fan-out still costs
    O(n^2) work in XLA on the dense tier and an iterative solve beyond it, so
    that shape is better served by a hand-written
    :class:`~probjax.core.custom_inverse`.
    """
    if not target_vars or not is_affine_in(jaxpr, target_vars):
        return None

    shapes = [v.aval.shape for v in target_vars]
    sizes = [_flat_size(v) for v in target_vars]
    dtype = jnp.result_type(*[v.aval.dtype for v in target_vars])
    total_in = sum(sizes)

    flat_outputs = [jnp.asarray(o).reshape(-1) for o in outputs]
    total_out = sum(o.size for o in flat_outputs)
    traced_out = sum(_flat_size(v) for v in jaxpr.outvars)
    if total_out != traced_out:
        # The supplied outputs do not belong to the traced program -- e.g. a
        # shape-changing map traced at the wrong shape. There is nothing sound
        # to solve; decline (NaN) rather than broadcasting garbage.
        return None
    if total_out < total_in:
        # Wide system: information was lost, so no unique inverse exists.
        return None
    tall = total_out > total_in
    if tall and (
        total_in > _DENSE_THRESHOLD
        or total_in * total_out > 4 * _DENSE_THRESHOLD * _DENSE_THRESHOLD
    ):
        # The dense matrix would be absurd; a custom_inverse serves this.
        return None

    def evaluate(flat_target):
        pieces, offset = [], 0
        for shape, size in zip(shapes, sizes, strict=False):
            pieces.append(flat_target[offset : offset + size].reshape(shape))
            offset += size
        environment = dict(known)
        environment.update(dict(zip(target_vars, pieces, strict=False)))
        args = [
            v.val if isinstance(v, Literal) else environment[v] for v in jaxpr.invars
        ]
        result = jax_core.eval_jaxpr(jaxpr, consts, *args)
        return jnp.concatenate([jnp.asarray(r).reshape(-1) for r in result])

    zero = jnp.zeros((total_in,), dtype)
    constant = evaluate(zero)

    def matvec(v):
        return evaluate(v) - constant

    rhs = jnp.concatenate(flat_outputs) - constant

    if tall:
        # Overdetermined consistent system (`tile`, `concat([x, x])`, padding):
        # least squares recovers the input exactly when the outputs agree,
        # and the replay guard reports NaN where they do not.
        basis = jnp.eye(total_in, dtype=dtype)
        matrix = jax.vmap(matvec)(basis).T
        solution, _, _, _ = jnp.linalg.lstsq(matrix, rhs)
        valid = inverse_roundtrip_valid(
            evaluate(solution), jnp.concatenate(flat_outputs)
        )
        values, offset = [], 0
        for var, shape, size in zip(target_vars, shapes, sizes, strict=False):
            piece = solution[offset : offset + size].reshape(shape)
            values.append(
                apply_inverse_guard(
                    piece,
                    var.aval,
                    valid,
                    message=("affine system is inconsistent: no inverse at this value"),
                )
            )
            offset += size
        return values, jnp.asarray(jnp.nan)

    if _is_elementwise_affine_in(jaxpr, target_vars):
        # A is diagonal: one forward evaluation with all ones reads it off.
        # x + x -> d = 2, and the inverse is (y - b) / d in O(n).
        diagonal = matvec(jnp.ones((total_in,), dtype))
        invertible = diagonal != 0
        flat_solution = jnp.where(
            invertible, rhs / diagonal, jnp.full_like(rhs, jnp.nan)
        )
        if need_logdet:
            safe_diagonal = jnp.where(invertible, diagonal, jnp.ones_like(diagonal))
            safe_logdet = -jnp.sum(jnp.log(jnp.abs(safe_diagonal)))
            log_abs_det = jnp.where(
                jnp.all(invertible), safe_logdet, jnp.full((), jnp.inf)
            )
        else:
            log_abs_det = jnp.asarray(jnp.nan)
        return [flat_solution.reshape(shapes[0])], log_abs_det

    if total_in <= _DENSE_THRESHOLD:
        # Small coupled map: build A with one vmapped sweep over the basis
        # (a single staged trace, not a Python loop) and solve directly.
        basis = jnp.eye(total_in, dtype=dtype)
        matrix = jax.vmap(matvec)(basis).T
        solution = jnp.linalg.solve(matrix, rhs)
        if need_logdet:
            _, log_abs_det = jnp.linalg.slogdet(matrix)
        else:
            log_abs_det = jnp.asarray(jnp.nan)
    else:
        # Large coupled map: matrix-free gmres, O(n) memory. The log-det
        # needs the dense matrix, so it comes back NaN by design.
        from jax.scipy.sparse.linalg import gmres as _gmres

        solution, info = _gmres(matvec, rhs, tol=1e-8, atol=1e-8)
        residual = matvec(solution) - rhs
        scale = 1 + jnp.linalg.norm(rhs)
        converged = (info == 0) & (jnp.linalg.norm(residual) <= 1e-6 * scale)
        solution = jnp.where(converged, solution, jnp.full_like(solution, jnp.nan))
        log_abs_det = jnp.asarray(jnp.nan)

    values, offset = [], 0
    for shape, size in zip(shapes, sizes, strict=False):
        values.append(solution[offset : offset + size].reshape(shape))
        offset += size
    # log|d(inv)/dy| = -log|det A|; a singular A gives -inf here and NaN above,
    # which is the correct report for a map that is not invertible.
    return values, -log_abs_det

"""Inverting the affine programs that equation-by-equation propagation cannot.

The inverse interpreter resolves one equation at a time, so it inverts a *tree*
of operations. A variable used twice breaks that: in ``3 * x - x`` the ``sub``
has two unknown operands, because both depend on the ``x`` being solved for, and
the bivariate rules need exactly one unknown. Propagation stalls and the target
comes back NaN.

That is a limitation of local propagation, not of the problem. When the stalled
program is **affine** in the target -- ``y = A x + b`` -- the inverse is a linear
solve, and both ``A`` and ``b`` can be extracted exactly:

    b = f(0)                      the constant term
    A = jacfwd(f)(0)              constant, because f is affine
    x = solve(A, y - b)
    log|d(inv)/dy| = -log|det A|

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
from jax._src import core as jax_core
from jax.extend.core import Jaxpr, Literal, Var

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
        name = eqn.primitive.name
        if name in _AFFINE_PRIMITIVES:
            pass
        elif name in _BILINEAR_PRIMITIVES:
            if len(touching) > 1:
                return False  # e.g. x * x
        else:
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
        is not affine in the target or the system is not square.

    The Jacobian is materialised, which costs O(n^2) in the target's size. That
    is acceptable here because this path runs only where the alternative is
    NaN, but it does mean a very large affine fan-out is better served by a
    hand-written :class:`~probjax.core.custom_inverse`.
    """
    if not target_vars or not is_affine_in(jaxpr, target_vars):
        return None

    shapes = [v.aval.shape for v in target_vars]
    sizes = [_flat_size(v) for v in target_vars]
    dtype = jnp.result_type(*[v.aval.dtype for v in target_vars])
    total_in = sum(sizes)

    flat_outputs = [jnp.asarray(o).reshape(-1) for o in outputs]
    total_out = sum(o.size for o in flat_outputs)
    if total_out != total_in:
        # Not a square system, so not a bijection on these variables.
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
    matrix = jax.jacfwd(evaluate)(zero)

    rhs = jnp.concatenate(flat_outputs) - constant
    solution = jnp.linalg.solve(matrix, rhs)
    _, log_abs_det = jnp.linalg.slogdet(matrix)

    values, offset = [], 0
    for shape, size in zip(shapes, sizes, strict=False):
        values.append(solution[offset : offset + size].reshape(shape))
        offset += size
    # log|d(inv)/dy| = -log|det A|; a singular A gives -inf here and NaN above,
    # which is the correct report for a map that is not invertible.
    return values, -log_abs_det

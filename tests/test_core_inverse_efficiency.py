"""The generated inverse must cost what a hand-written one costs.

The interpreter used to validate every inverse by re-binding the forward
primitive and comparing its output. XLA does not erase that -- it was six extra
equations per equation, and on an 8-deep chain it made ``inverse`` 3.2x slower
and ``inverse_and_logabsdet`` 1.7x slower than equivalent hand-written code.
These tests hold the replacement to parity.

The check is now per-rule and declared only where an inverse is genuinely
value-dependent (dividing by something that may be zero, a branch chosen at
runtime). Where a guard can be settled at trace time -- a literal multiplier --
nothing is emitted at all.
"""

import time

import jax
import jax.numpy as jnp
import pytest

from probjax.core import inverse, inverse_and_logabsdet

DEPTH = 8


def forward(x):
    for _ in range(DEPTH):
        x = 2.0 * jnp.exp(x) + 1.0
    return x


def hand_written_inverse(y):
    for _ in range(DEPTH):
        y = jnp.log((y - 1.0) / 2.0)
    return y


def hand_written_inverse_and_logdet(y):
    logdet = jnp.zeros(())
    for _ in range(DEPTH):
        # d/dy log((y - 1) / 2) = 1 / (y - 1)
        logdet = logdet - jnp.sum(jnp.log(jnp.abs(y - 1.0)))
        y = jnp.log((y - 1.0) / 2.0)
    return y, logdet


def _walk(jaxpr):
    """Every equation in a jaxpr, descending into nested ones."""
    stack, seen = [jaxpr], set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for eqn in current.eqns:
            yield eqn
            for param in eqn.params.values():
                for candidate in param if isinstance(param, (list, tuple)) else [param]:
                    inner = getattr(candidate, "jaxpr", None)
                    if inner is not None:
                        stack.append(getattr(inner, "jaxpr", inner))


def count_equations(fn, *args):
    return sum(1 for _ in _walk(jax.make_jaxpr(fn)(*args).jaxpr))


def all_primitives(fn, *args):
    return {str(eqn.primitive) for eqn in _walk(jax.make_jaxpr(fn)(*args).jaxpr)}


def best_of(fn, arg, repeats=40):
    compiled = jax.jit(fn)
    jax.block_until_ready(compiled(arg))
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(compiled(arg))
        timings.append(time.perf_counter() - start)
    return min(timings)


def test_inverse_emits_exactly_the_hand_written_arithmetic():
    """The strong claim: the same op count, not merely a similar one.

    ``2*exp(x)+1`` inverts to ``log((y-1)/2)`` -- one sub, one div and one log
    per layer, nothing else. Validation-by-replay used to add a forward bind, an
    isclose, a reduce_and and a select on top of every one of them.
    """
    x = jnp.asarray([0.5])
    generated = count_equations(inverse(forward), x)
    hand = count_equations(hand_written_inverse, x)
    assert generated == hand, f"{generated} equations against {hand} hand-written"


def test_inverse_and_logabsdet_stays_within_a_small_factor():
    """The log-det path emits more equations, but they fuse.

    Accumulating the running log-determinant costs adds and reductions that a
    hand-written loop folds together, so the equation count overstates it: the
    compiled HLO is about 1.2x and the measured runtime is within noise (below).
    This bound is a regression guard -- with per-primitive validation it was
    past 4x.
    """
    x = jnp.asarray([0.5])
    generated = count_equations(lambda y: inverse_and_logabsdet(forward)(y), x)
    hand = count_equations(hand_written_inverse_and_logdet, x)
    assert generated < hand * 3, f"{generated} equations against {hand} hand-written"


@pytest.mark.parametrize(
    "generated,handwritten",
    [
        (inverse(forward), hand_written_inverse),
        (lambda y: inverse_and_logabsdet(forward)(y), hand_written_inverse_and_logdet),
    ],
    ids=["inverse", "inverse_and_logabsdet"],
)
def test_generated_inverse_runs_at_hand_written_speed(generated, handwritten):
    """The criterion that matters: cost after jit, not jaxpr size.

    Measured across 40 runs the ratio sits at 0.8-1.1x for both variants, i.e.
    parity within noise. The bound is set well above that because a loaded
    machine skews the minimum -- ten repeats under a concurrent test run
    produced 1.57x for the same code. It still catches the regression it exists
    for: validation-by-replay was 3.2x.
    """
    y = jax.jit(forward)(jnp.linspace(0.1, 0.5, 100_000))
    ratio = best_of(generated, y) / best_of(handwritten, y)
    assert ratio < 2.0, f"generated inverse is {ratio:.2f}x hand-written"


def _vp_forward(x):
    for _ in range(DEPTH):
        x = jnp.flip(x + 1.0) - 2.0
    return x


def _vp_hand_written(y):
    for _ in range(DEPTH):
        y = jnp.flip(y + 2.0) - 1.0
    return y, jnp.zeros(())


def test_volume_preserving_inverse_runs_at_hand_written_speed():
    """The fast path must leave no runtime residue: no `+ 0.0` accumulation,
    no staged log-det arithmetic. Measured at ~1.0x; the bound is the file's
    usual allowance for loaded machines."""
    y = jax.jit(_vp_forward)(jnp.linspace(0.1, 0.5, 100_000))
    generated = lambda v: inverse_and_logabsdet(_vp_forward)(v)  # noqa: E731
    ratio = best_of(generated, y) / best_of(_vp_hand_written, y)
    assert ratio < 2.0, f"generated VP inverse is {ratio:.2f}x hand-written"


def test_no_guard_means_no_emitted_check():
    """A chain of exactly-invertible primitives emits only the inverse.

    ``exp``, ``add`` and ``mul``-by-a-literal are exact wherever they are
    defined, so nothing should be spent deciding whether the answer is valid: a
    domain violation surfaces as NaN through IEEE for free.
    """
    primitives = all_primitives(inverse(forward), jnp.asarray([0.5]))
    assert primitives == {"sub", "div", "log"}, primitives


def test_guard_on_a_literal_costs_nothing():
    """A statically non-zero multiplier must not emit a runtime check."""
    assert all_primitives(inverse(lambda x: x * 2.0), jnp.asarray([4.0])) == {"div"}


def test_guarded_rule_emits_a_check_where_it_is_needed():
    """A multiplier that is zero somewhere cannot be settled at trace time.

    The guard has to become a runtime select here, in contrast to the literal
    case above -- that is the whole point of declaring guards per rule instead
    of validating every primitive.
    """
    scale = jnp.asarray([0.0, 2.0])
    primitives = all_primitives(inverse(lambda x: x * scale), jnp.asarray([5.0, 4.0]))
    assert "select_n" in primitives, primitives


def test_guard_produces_nan_rather_than_a_wrong_number():
    scale = jnp.asarray([0.0, 2.0])
    recovered = inverse(lambda x: x * scale)(jnp.asarray([5.0, 4.0]))
    assert bool(jnp.isnan(recovered[0]))
    assert float(recovered[1]) == pytest.approx(2.0)

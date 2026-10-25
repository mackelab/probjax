"""The compiled-scan progress bar: same results as lax.scan, a forward-only bar."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probjax.utils.progress import ProgressBar, progress_scan


def _body(carry, x):
    carry = carry + x
    return carry, (carry, 2.0 * x)


@pytest.mark.parametrize("length", [1, 7, 50, 103])
def test_progress_scan_matches_scan(length, capsys):
    xs = jnp.arange(length, dtype=jnp.float32)
    expected = jax.lax.scan(_body, jnp.float32(0.0), xs)
    actual = jax.jit(
        lambda xs: progress_scan(
            _body,
            jnp.float32(0.0),
            xs,
            length,
            names=("double",),
            stats_fn=lambda ys: (ys[1],),
            num_updates=10,
        )
    )(xs)
    jax.effects_barrier()
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b)
    output = capsys.readouterr().out
    assert output.endswith("\n")
    assert f"{length}/{length}" in output.split("\r")[-1]


def test_progress_scan_under_vmap_draws_one_bar(capsys):
    def one(x0):
        return progress_scan(_body, x0, jnp.ones(20), 20, num_updates=4)[0]

    finals = jax.jit(jax.vmap(one))(jnp.arange(3.0))
    jax.effects_barrier()
    np.testing.assert_allclose(finals, jnp.arange(3.0) + 20)
    assert capsys.readouterr().out.count("\n") == 1


def test_bar_only_moves_forward(capsys):
    bar = ProgressBar(10, ("loss",), first_step=1, min_interval=0.0)
    for step in [1, 5, 3, 5, 9, 9]:  # late and duplicate reports
        bar.update(step, (0.5,))
    draws = capsys.readouterr().out.split("\r")[1:]
    assert [draw.split("]")[0].split()[1] for draw in draws] == ["2/10", "6/10", "10/10"]
    assert draws[-1].endswith("loss: 0.5\n")
    # A rerun of the compiled program reports its first step again.
    bar.update(1, (0.1,))
    assert "2/10" in capsys.readouterr().out

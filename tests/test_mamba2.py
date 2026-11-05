"""SSD mixer, SSDCell and Mamba2Cell: reference recurrence, stability, config."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from probjax.nn.layers.ssm import Mamba2Cell, SSDCell, ssd, ssd_mixer
from probjax.nn.nets.ssm import SSMModel


def _mixer_reference(x, b, c, dt, a_log, d, num_heads, num_groups):
    """Sequential float64 recurrence of the Mamba-2 SSM (see ssd_mixer)."""
    x, b, c, dt, a_log = (np.asarray(t, np.float64) for t in (x, b, c, dt, a_log))
    batch, length, width = x.shape
    p, n = width // num_heads, b.shape[-1] // num_groups
    rep = num_heads // num_groups
    x = x.reshape(batch, length, num_heads, p)
    b = b.reshape(batch, length, num_groups, n)
    c = c.reshape(batch, length, num_groups, n)
    a = -np.exp(a_log)
    y = np.zeros_like(x)
    for h in range(num_heads):
        g = h // rep
        state = np.zeros((batch, n, p))
        for t in range(length):
            decay = np.exp(a[h] * dt[:, t, h])[:, None, None]
            write = dt[:, t, h, None, None] * b[:, t, g, :, None] * x[:, t, h, None, :]
            state = decay * state + write
            y[:, t, h] = np.einsum("bn,bnp->bp", c[:, t, g], state)
            if d is not None:
                y[:, t, h] += float(d[h]) * x[:, t, h]
    return y.reshape(batch, length, width)


def _mixer_inputs(batch, length, heads, groups, p, n, seed=0):
    ks = jax.random.split(jax.random.key(seed), 6)
    x = jax.random.normal(ks[0], (batch, length, heads * p))
    b = jax.random.normal(ks[1], (batch, length, groups * n))
    c = jax.random.normal(ks[2], (batch, length, groups * n))
    dt = jax.nn.softplus(jax.random.normal(ks[3], (batch, length, heads)) - 2.0)
    a_log = jnp.log(jax.random.uniform(ks[4], (heads,), minval=1.0, maxval=4.0))
    d = jax.random.normal(ks[5], (heads,))
    return x, b, c, dt, a_log, d


@pytest.mark.parametrize(
    "heads,groups,p,n,use_d",
    [(1, 1, 8, 4, True), (4, 2, 8, 6, True), (4, 1, 5, 3, False), (6, 3, 4, 8, True)],
)
def test_ssd_mixer_matches_sequential_recurrence(heads, groups, p, n, use_d):
    x, b, c, dt, a_log, d = _mixer_inputs(2, 37, heads, groups, p, n, seed=heads)
    d = d if use_d else None
    out = jax.jit(
        lambda *a: ssd_mixer(*a, num_heads=heads, num_groups=groups, chunk_size=16)
    )(x, b, c, dt, a_log, d)
    ref = _mixer_reference(x, b, c, dt, a_log, d, heads, groups)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=2e-4, atol=2e-4)


def test_ssd_kernel_options_are_forwarded():
    x, b, c, dt, a_log, _ = _mixer_inputs(1, 50, 2, 1, 8, 4)
    q = c.reshape(1, 50, 1, 4).transpose(0, 2, 1, 3)
    k = b.reshape(1, 50, 1, 4).transpose(0, 2, 1, 3)
    v = x.reshape(1, 50, 2, 8).transpose(0, 2, 1, 3)
    la = (-jnp.exp(a_log) * dt).transpose(0, 2, 1)
    base = ssd(q, k, v, la, implementation="xla")
    for kw in (
        dict(chunk_size=8),
        dict(chunk_size=64, remat="none"),
        dict(remat="states"),
    ):
        np.testing.assert_allclose(
            ssd(q, k, v, la, implementation="xla", **kw), base, rtol=1e-5, atol=1e-5
        )
    out, final = ssd(q, k, v, la, implementation="xla", return_final_state=True)
    np.testing.assert_allclose(out, base, rtol=1e-6, atol=1e-6)
    assert final.shape == (1, 2, 4, 8)
    # Continuing from the final state equals running the concatenation.
    full = ssd(
        jnp.concatenate([q, q], 2),
        jnp.concatenate([k, k], 2),
        jnp.concatenate([v, v], 2),
        jnp.concatenate([la, la], 2),
        implementation="xla",
    )
    second = ssd(q, k, v, la, final, implementation="xla")
    np.testing.assert_allclose(second, full[:, :, 50:], rtol=1e-4, atol=1e-4)


def test_ssd_return_final_state_rejects_pallas():
    x, b, c, dt, a_log, _ = _mixer_inputs(1, 64, 2, 1, 64, 64)
    q = c.reshape(1, 64, 1, 64).transpose(0, 2, 1, 3)
    v = x.reshape(1, 64, 2, 64).transpose(0, 2, 1, 3)
    la = (-jnp.exp(a_log) * dt).transpose(0, 2, 1)
    with pytest.raises((ValueError, RuntimeError)):
        ssd(q, q, v, la, implementation="pallas", return_final_state=True)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


CELLS = [
    (SSDCell, {}),
    (SSDCell, dict(num_heads=4, num_groups=2, state_dim=16)),
    (SSDCell, dict(num_heads=3, head_dim=8, state_dim=8, use_skip=False)),
    (Mamba2Cell, {}),
    (Mamba2Cell, dict(expand=4, head_dim=16, num_groups=2, state_dim=16)),
    (Mamba2Cell, dict(head_dim=8, conv_size=2, use_skip=False)),
]


@pytest.mark.parametrize("cls,kwargs", CELLS)
def test_cells_shape_finite_and_causal(cls, kwargs):
    model_dim, length = 24, 96
    cell = cls(model_dim, nnx.Rngs(0), **kwargs)
    x = jax.random.normal(jax.random.key(1), (2, length, model_dim))
    y = jax.jit(lambda x: cell(x))(x)
    assert y.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(y)))
    # Unbatched input.
    assert _rel(cell(x[0]), y[0]) < 1e-3
    # Causal: changing the future leaves the past unchanged.
    x2 = x.at[:, length // 2 :].add(3.0)
    y2 = cell(x2)
    assert _rel(y2[:, : length // 2], y[:, : length // 2]) < 1e-3
    assert not np.allclose(y2[:, length // 2 :], y[:, length // 2 :])


@pytest.mark.parametrize("cls", [SSDCell, Mamba2Cell])
def test_cells_stay_finite_on_long_sequences(cls):
    """Regression: the old SSDCell's unconstrained log_alpha made the state
    grow exponentially (NaN by ~64 tokens at default init)."""
    cell = cls(32, nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(2), (1, 4096, 32)) * 3.0
    y = cell(x)
    assert bool(jnp.all(jnp.isfinite(y)))
    g = jax.grad(lambda c, x: jnp.mean(c(x) ** 2))(cell, x[:, :512])
    assert all(bool(jnp.all(jnp.isfinite(t))) for t in jax.tree.leaves(g))
    assert float(jnp.max(jnp.abs(g.a_log[...]))) > 0
    assert float(jnp.max(jnp.abs(g.dt_bias[...]))) > 0


def test_ssdcell_matches_reference_from_its_parameters():
    cell = SSDCell(12, nnx.Rngs(3), num_heads=2, num_groups=1, state_dim=5)
    u = jax.random.normal(jax.random.key(4), (2, 30, 12))
    dt = jax.nn.softplus(cell.to_dt(u) + cell.dt_bias[...])
    with jax.default_matmul_precision("highest"):
        mixed = _mixer_reference(
            cell.to_x(u),
            cell.to_b(u),
            cell.to_c(u),
            dt,
            cell.a_log[...],
            cell.d[...],
            2,
            1,
        )
        ref = np.asarray(cell.to_out(jnp.asarray(mixed, jnp.float32)))
        out = cell(u)
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-4, atol=1e-4)


def test_mamba2_config_and_init():
    cell = Mamba2Cell(64, nnx.Rngs(0), state_dim=32, num_groups=2)
    assert (cell.inner_dim, cell.head_dim, cell.num_heads) == (128, 64, 2)
    a = np.exp(np.asarray(cell.a_log[...]))
    assert np.all((a >= 1.0) & (a <= 16.0))
    dt = np.asarray(jax.nn.softplus(cell.dt_bias[...]))
    assert np.all((dt >= 1e-3 - 1e-7) & (dt <= 0.1 + 1e-7))
    np.testing.assert_array_equal(np.asarray(cell.d[...]), 1.0)
    # Single head fallback when expand * model_dim is not a multiple of 64.
    assert Mamba2Cell(24, nnx.Rngs(0)).num_heads == 1
    with pytest.raises(ValueError):
        Mamba2Cell(24, nnx.Rngs(0), head_dim=10)
    with pytest.raises(ValueError):
        Mamba2Cell(64, nnx.Rngs(0), head_dim=32, num_groups=3)
    with pytest.raises(ValueError):
        SSDCell(30, nnx.Rngs(0), num_heads=4)


def test_mamba2_bf16_runs():
    cell = Mamba2Cell(32, nnx.Rngs(0), dtype=jnp.bfloat16)
    x = jax.random.normal(jax.random.key(0), (2, 128, 32)).astype(jnp.bfloat16)
    y = cell(x)
    assert y.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.isfinite(y.astype(jnp.float32))))


@pytest.mark.parametrize("cls", [SSDCell, Mamba2Cell])
def test_ssm_model_with_ssd_cells_trains(cls):
    model = SSMModel(
        4, 32, 3, 2, recurrent_cls=cls, bidirectional=False, rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(jax.random.key(0), (2, 64, 4))
    target = jnp.cumsum(x[..., :3], axis=1) / 8.0
    opt = nnx.Optimizer(model, __import__("optax").adam(3e-3), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt):
        loss, g = nnx.value_and_grad(lambda m: jnp.mean((m(x) - target) ** 2))(model)
        opt.update(model, g)
        return loss

    losses = [float(step(model, opt)) for _ in range(30)]
    assert all(np.isfinite(losses))
    assert losses[-1] < 0.5 * losses[0]

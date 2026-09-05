# Plan: Refactor `odeint.py` closure-lifting machinery

## Problem

`probjax/utils/odeint.py` reimplements the job that `jax.jit` is designed to
do: thread traced values through a callable. To pin `drift` as static for
`custom_inverse`, the current implementation introspects `drift.__closure__`
and dataclass fields, rebuilds `types.FunctionType` objects with synthetic
cells, and packs the traced cell contents into `*args` so they survive the
`static_argnames` boundary. This creates:

- **Two overlapping lift pipelines** — `_lift_drift_traced_closure` runs in
  `odeint(...)` before `_odeint_custom`, and `_prepare_drift_call` runs
  inside `_odeint_custom`. They partially duplicate and partially compose.
- **CPython-specific closure rebuilding** — `_make_cell` relies on
  `inner.__closure__[0]` being a writable cell; `types.FunctionType(...)`
  must match the free-variable count exactly.
- **Partial dataclass coverage** — only `FunctionType` dataclass fields are
  detected. `eqx.Module`, `flax.linen.Module`, `functools.partial`, and bound
  methods fall through and silently bake their tracers in.
- **"Traced by identity" heuristic** — `_has_tracer` checks
  `isinstance(x, jax.core.Tracer)`. Works for the outer-jit case; fragile
  under nested transforms.
- **Positional-packing ABI** — `drift_args_tuple` + optional
  `dynamic_kwargs` + optional `function_dynamic` + optional
  `dataclass_dynamic` are concatenated into one `packed` tuple and decoded
  positionally. Any new "dynamic" category is a packing bug waiting to
  happen. The inverse path in `odeutil/inversion.py` has to mirror this
  exactly.
- **Docstring drift** — `_odeint_custom` documents `*args` but actually
  takes `drift_args: Sequence[Any]` and `drift_kwargs: Mapping`.

## Design

### Contract change: `drift` is a pytree, not a closure

Require callers to expose traced parameters explicitly. Two supported forms:

1. **Plain function + explicit args** — `drift(t, y, *args, **kwargs)` with
   *all* traced parameters passed through `args` / `kwargs`.
2. **Pytree-callable protocol** — any object that
   - is a registered JAX pytree node (leaves = traced params, aux = static
     config), and
   - is callable with signature `(t, y, *args, **kwargs)`.

   This covers `equinox.Module`, `flax.struct.PyTreeNode`, and any user
   dataclass decorated with `jax.tree_util.register_pytree_node_class`.

With that contract, `jax.jit` handles everything natively: the drift's
traced leaves ride through as regular pytree inputs, aux data is hashed for
the cache key, and no closure introspection is needed.

### Target `odeint.py` shape (~30 lines)

```python
@partial(custom_inverse, inv_argnum=1)
def _odeint_custom(drift, y0, ts, args, kwargs, *, method, dtype,
                   filter_state, collect_trace, check_points, adaptive_params):
    return _odeint(drift, y0, ts, *args, **kwargs,
                   method=method, dtype=dtype, filter_state=filter_state,
                   collect_trace=collect_trace, check_points=check_points,
                   adaptive_params=adaptive_params)


def odeint(drift, y0, ts, *args, method="rk4", dtype=jnp.float32,
           filter_state=None, collect_trace=True, check_points=None,
           adaptive_params=None, **kwargs):
    return _odeint_custom(
        drift, y0, ts, args, kwargs,
        method=method, dtype=dtype, filter_state=filter_state,
        collect_trace=collect_trace, check_points=check_points,
        adaptive_params=adaptive_params,
    )
```

Notes:

- `drift` is no longer a static arg; it rides through as a pytree. JAX
  caches on `(aux_data, treedef)` automatically.
- `args` and `kwargs` are bundled as pytrees; no manual partitioning into
  static vs. traced is needed.
- `custom_inverse` keeps `inv_argnum=1` (i.e. `y0`). Drop
  `static_argnums=(0,)`.

### `custom_inverse` migration

Today `custom_inverse(..., static_argnums=(0,))` pins `drift`. The inverse
registry (`_inv_odeint`, `_inv_logdet_odeint` in
`odeutil/inversion.py`) must also stop assuming `drift` is static. Confirm
whether `custom_inverse` dispatches on the function identity or on the
call-site decoration; if the former, the refactor is a no-op for dispatch
and the inverses just need their signatures updated to match the new
`(drift, y0, ts, args, kwargs, ...)` layout.

### Deletions

Remove the entire closure-introspection block from `odeint.py`:

- `_has_tracer`
- `_make_cell`
- `_extract_function_dynamic_cells`
- `_rebuild_function_with_dynamic_cells`
- `_extract_dataclass_function_dynamic_cells`
- `_rebuild_dataclass_with_dynamic_cells`
- `_prepare_drift_call`
- `_partition_drift_kwargs`
- `_lift_drift_traced_closure`
- `_bind_drift_kwargs` (its job — partial-apply static kwargs — is no
  longer needed; kwargs just flow through)

### `odeutil/core.py` changes

`_odeint` already accepts `*args` and doesn't need closure introspection.
The only change is to accept `**kwargs` and forward them to `drift` inside
`ravel_arg_fun`. `ravel_arg_fun` may need a minor update to forward kwargs.

### `odeutil/inversion.py` changes

Mirror the new `(drift, y0, ts, args, kwargs, ...)` signature. Today the
inverses consume the same packed `*args` convention; after the refactor
they should consume the explicit `args` / `kwargs` pytrees, same as
`_odeint_custom`.

## Migration steps

1. **Audit call sites.**
   - `grep` for `odeint(` and `_odeint_custom(` under `probjax/` and
     `examples/` — list every caller and note how each passes drift
     parameters today (closure, `drift_args`, `drift_kwargs`, `*args`,
     `**kwargs`). `sdeint.py` likely has a parallel structure; decide
     whether to refactor it in the same PR.
2. **Update `custom_inverse`.** Verify the decorator supports the new
   signature or adjust it.
3. **Rewrite `odeint.py`.** Replace the lift machinery with the ~30-line
   version above. Keep the public `odeint(...)` signature stable
   (`*args, **kwargs` already worked for most callers).
4. **Update `odeutil/core.py::_odeint`.** Accept `**kwargs`, forward them
   through `ravel_arg_fun`.
5. **Update `odeutil/inversion.py`.** Match the new argument layout in the
   `definv` / `definv_and_logdet` hooks.
6. **Run the existing test suite.** `tests/test_odeint.py` and
   `tests/test_sdeint.py` should pass unchanged — they already exercise
   the `*args`/`**kwargs` path (see `test_odeint_supports_drift_kwargs`
   lines 193–269, which tests `jax.jit`-wrapping the whole thing).
7. **Add the new tests below.**
8. **Delete now-dead helpers** from `odeint.py`.

## Test additions (`tests/test_odeint.py`)

Add a new block after `test_odeint_supports_drift_kwargs`. The neural-net
test is the headline case — today it would either trigger closure
rebuilding (fragile) or silently bake the parameters in (wrong). After the
refactor it's just a pytree-callable.

### 1. `test_odeint_with_equinox_neural_drift` (skip if `equinox` missing)

```python
def test_odeint_with_equinox_neural_drift(ode_method):
    if ode_method in KNOWN_ERROR + SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return
    eqx = pytest.importorskip("equinox")

    class NNDrift(eqx.Module):
        mlp: eqx.nn.MLP

        def __init__(self, key, in_dim=2, width=16, depth=2):
            self.mlp = eqx.nn.MLP(
                in_size=in_dim + 1, out_size=in_dim,
                width_size=width, depth=depth, key=key,
            )

        def __call__(self, t, y):
            inp = jnp.concatenate([jnp.atleast_1d(t), y])
            return self.mlp(inp)

    key = jax.random.PRNGKey(0)
    drift = NNDrift(key)
    y0 = jnp.array([0.5, -0.3])
    ts = jnp.linspace(0.0, 1.0, 20)

    # (a) plain call works
    trace = odeint(drift, y0, ts, method=ode_method, collect_trace=True)
    assert trace.shape == (ts.shape[0], y0.shape[0])
    assert jnp.all(jnp.isfinite(trace))

    # (b) traced params survive jit — the load-bearing assertion
    @jax.jit
    def run(d, y):
        return odeint(d, y, ts, method=ode_method, collect_trace=False)
    final = run(drift, y0)
    assert final.shape == y0.shape

    # (c) grad flows through drift parameters
    def loss(d):
        out = odeint(d, y0, ts, method=ode_method, collect_trace=False)
        return jnp.sum(out ** 2)
    grads = jax.grad(loss)(drift)
    # At least one MLP weight should get a nonzero gradient.
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(jnp.abs(l) > 0) for l in leaves if l.ndim > 0)

    # (d) vmap over an ensemble of drifts
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    ensemble = jax.vmap(NNDrift)(keys)
    batched = jax.vmap(
        lambda d: odeint(d, y0, ts, method=ode_method, collect_trace=False)
    )(ensemble)
    assert batched.shape == (4, y0.shape[0])
```

What each sub-assertion catches:

- **(a)** Regression of the plain path.
- **(b)** The current failure mode: under `jax.jit`, NN weights end up
  inside `drift.__closure__` / dataclass fields; closure-rebuilding either
  misses them or packs them wrong. The refactor makes this a no-op.
- **(c)** Autodiff has to see NN weights as leaves, not baked-in
  constants.
- **(d)** `vmap`-over-parameters is the exact thing that silently breaks
  when params are treated as closure-captured constants.

### 2. `test_odeint_with_plain_closure_params_is_rejected_or_lifted`

Pins the *contract* decision. Either:

- explicitly *reject* a closure that captures traced arrays with a clear
  `TypeError` telling the user to pass params through `*args` / `**kwargs`
  or to use a pytree-callable, or
- document that plain-closure capture works for *non-traced* values only.

```python
def test_odeint_plain_closure_captures_are_constant_only():
    # Non-traced closure capture is fine.
    scale = 0.5  # Python float, not a jnp array
    def drift(t, y):
        del t
        return -scale * y

    y0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 1.0, 10)
    out = odeint(drift, y0, ts, method="rk4", collect_trace=True)
    assert out.shape == (10, 1)
```

### 3. `test_odeint_with_partial_drift`

```python
from functools import partial

def test_odeint_with_partial_drift(ode_method):
    if ode_method in KNOWN_ERROR + SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return

    def base_drift(t, y, rate, bias):
        del t
        return rate * y + bias

    drift = partial(base_drift, rate=jnp.array(-0.3), bias=jnp.array(0.1))
    y0 = jnp.array([1.0, -2.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    out = odeint(drift, y0, ts, method=ode_method, collect_trace=True)
    assert out.shape == (ts.shape[0], y0.shape[0])
    assert jnp.all(jnp.isfinite(out))
```

This is currently a landmine: `functools.partial` is not a `FunctionType`,
so `_extract_function_dynamic_cells` returns `None`, and the `jnp.array`
captures are baked in as constants on the first trace.

### 4. `test_odeint_with_dataclass_of_params`

```python
from flax import struct

def test_odeint_with_pytree_dataclass_drift(ode_method):
    if ode_method in KNOWN_ERROR + SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return

    @struct.dataclass
    class LinearDrift:
        A: jax.Array
        b: jax.Array

        def __call__(self, t, y):
            del t
            return self.A @ y + self.b

    drift = LinearDrift(
        A=jnp.array([[-1.0, 0.3], [0.0, -0.5]]),
        b=jnp.array([0.1, -0.2]),
    )
    y0 = jnp.array([1.0, 1.0])
    ts = jnp.linspace(0.0, 1.0, 20)

    @jax.jit
    def run(d, y):
        return odeint(d, y, ts, method=ode_method, collect_trace=False)

    out = run(drift, y0)
    assert out.shape == y0.shape

    # grad flows into dataclass fields
    def loss(d):
        return jnp.sum(odeint(d, y0, ts, method=ode_method, collect_trace=False) ** 2)
    g = jax.grad(loss)(drift)
    assert jnp.any(jnp.abs(g.A) > 0)
    assert jnp.any(jnp.abs(g.b) > 0)
```

### 5. `test_odeint_traced_kwargs_under_outer_jit`

Regression test for the current `_has_tracer`-based kwarg partitioning.
`rate` is a concrete array at call time but a `Tracer` under `jax.jit` —
the new code path should not care which.

```python
def test_odeint_traced_kwargs_under_outer_jit(ode_method):
    if ode_method in KNOWN_ERROR + SPLIT_DRIFT_METHODS + ["linear_exact"]:
        return

    def drift(t, y, rate):
        del t
        return rate * y

    y0 = jnp.array([1.0])
    ts = jnp.linspace(0.0, 1.0, 10)

    @jax.jit
    def run(y, r):
        return odeint(drift, y, ts, rate=r, method=ode_method, collect_trace=False)

    out_traced = run(y0, jnp.array(-0.2))
    out_direct = odeint(drift, y0, ts, rate=jnp.array(-0.2),
                        method=ode_method, collect_trace=False)
    assert jnp.allclose(out_traced, out_direct, atol=1e-5)
```

### 6. `test_odeint_inverse_still_works_with_neural_drift` (optional, harder)

If `_inv_odeint` / `_inv_logdet_odeint` are exercised elsewhere with NN
drifts (e.g. normalizing flows), add one test that runs a forward pass and
an inverse pass on the eqx `NNDrift` and asserts round-trip to tolerance.
Skip if the current inverse isn't guaranteed to converge on arbitrary
nonlinear drifts — but at minimum assert it *runs* without a
closure-rebuilding crash.

## Acceptance criteria

- `odeint.py` has no `__closure__`, `types.FunctionType`, `fields(...)`,
  or `_make_cell` references.
- Existing `tests/test_odeint.py` passes unchanged.
- New tests above pass.
- `examples/` notebooks that use `odeint` still run
  (`examples/inference/*`, `examples/nn/*`).
- `sdeint.py` is either refactored in the same PR or has a follow-up
  ticket linked, since it shares the pattern.

## Risks

- `custom_inverse` may depend on `drift` being static for dispatch. If so,
  the refactor needs a small tweak to key the inverse registry on the
  callable's `__wrapped__` or treedef aux rather than identity.
- Callers that relied on closure-capture-of-traced-arrays as a feature
  (not an accident) will break. Audit `examples/` first; most will already
  be using `*args` / `**kwargs`.
- `ravel_arg_fun` currently takes positional-only; forwarding kwargs may
  need a small update. Check `probjax/utils/jaxutils.py`.

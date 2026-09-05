# Plan: Redesign filter_smooth.py API

## Problem

Current `filter_smooth.py` is a flat set of functions inconsistent with the `MCMC`/`SMC` runner pattern. The `smooth()` function is overloaded (auto-detects args), `smooth_from_states`/`smooth_particle_from_states` are scattered helpers, and there's no progress support.

## Design

### `Filter(WithProgressBarAPI)` class — 3 methods + verbose

```python
class Filter(WithProgressBarAPI):
    def __init__(self, kernel: FilterKernel, verbose: bool = False):
        ...

    def filter(self, key, ts, t_o, x_o, *args, checkpoint_lengths=None, unroll=1, **kwargs) -> FilteringTrace:
        """Run filtering over time grid. Returns FilteringTrace."""

    def smooth(self, trace: FilteringTrace, smoother=None, key=None, transition_logdensity_fn=None) -> Any:
        """Smooth a FilteringTrace. Auto-detects Gaussian vs particle from state type."""

    def log_likelihood(self, key, ts, t_o, x_o, *args, checkpoint_lengths=None, unroll=1, **kwargs) -> ArrayLike:
        """Run filtering and return summed log-likelihood."""
```

Usage:
```python
from probjax.inference.filter_smooth import Filter

kf = kalman_filter(transition_model, observation_model)
filt = Filter(kf, verbose=True)

trace = filt.filter(key, ts, t_o, x_o, mu0, cov0)
mus_s, covs_s = filt.smooth(trace, smoother=smoother_fn)
ll = filt.log_likelihood(key, ts, t_o, x_o, mu0, cov0)
```

### Backward compat — free functions become thin wrappers

```python
def filter(key, ts, t_o, x_o, kernel, *args, **kwargs):
    return Filter(kernel).filter(key, ts, t_o, x_o, *args, **kwargs)

def smooth(*args, smoother=None, key=None, transition_logdensity_fn=None, **kwargs):
    # If first arg is FilteringTrace, route through Filter
    if args and isinstance(args[0], FilteringTrace):
        return Filter(None).smooth(args[0], smoother=smoother, key=key,
                                    transition_logdensity_fn=transition_logdensity_fn)
    # Otherwise, raw Gaussian smoothing (backward compat)
    return smooth_gaussian(*args, **kwargs)

def filter_log_likelihood(key, ts, t_o, x_o, kernel, *args, **kwargs):
    return Filter(kernel).log_likelihood(key, ts, t_o, x_o, *args, **kwargs)

def filter_and_smooth(key, ts, t_o, x_o, kernel, *args, smoother=None,
                      smooth_key=None, transition_logdensity_fn=None, **kwargs):
    filt = Filter(kernel)
    trace = filt.filter(key, ts, t_o, x_o, *args, **kwargs)
    smoothed = filt.smooth(trace, smoother=smoother, key=smooth_key,
                           transition_logdensity_fn=transition_logdensity_fn)
    return trace, smoothed
```

## Implementation details

### `Filter.filter()` — verbose scan

- Extracts scan logic from current `filter()` into `Filter._run_scan()`
- Verbose path: uses `print_scan` (like `MCMC.run()`)
- Non-verbose path: uses `jax.lax.scan` or `nested_checkpoint_scan`
- Tracked stats: `("log_likelihood",)` — extracted from `info.log_likelihood` when present

### `Filter.smooth()` — inlined dispatch

Inlines the logic from current `smooth()`, `smooth_from_states()`, `smooth_particle_from_states()`, and `_trace_ts()`:
- Detects particle states (`particles` + `log_weights`) → calls `particle_smoother()`
- Detects Gaussian states → extracts `mean`/`cov` from state + predicted from info, calls `smooth_gaussian()`
- Raises if missing required info

### `Filter.log_likelihood()` — thin wrapper

Reuses the `unpack_log_likelihood` helper, same as current `filter_log_likelihood()`.

## Files changed

1. **`probjax/inference/filter_smooth.py`** — Add `Filter` class, rewrite free functions as wrappers, remove `smooth_from_states`/`smooth_particle_from_states`/`_trace_ts` (inlined)
2. **`tests/test_filtering.py`** — Add `TestFilterClass` testing `Filter.filter()`, `Filter.smooth()`, `Filter.log_likelihood()` for Gaussian and particle cases

## Verification

```bash
python -m pytest tests/test_filtering.py -v
```
All existing tests pass (backward compat). New `TestFilterClass` tests verify the new API.

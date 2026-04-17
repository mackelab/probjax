---
title: Troubleshooting
---

# Troubleshooting Guide

This guide helps you resolve common issues when using ProbJax.

## Installation Issues

### JAX Installation Problems

#### Issue: `ImportError: cannot import name 'jax'`

**Solution:**
```bash
pip install jax jaxlib>=0.4.34
```

#### Issue: CUDA version mismatch

**Solution:**
Make sure your CUDA version matches the JAX wheel. For CUDA 12:
```bash
pip install jax[cuda12]
```

For CUDA 11:
```bash
pip install jax[cuda11_pip]
```

#### Issue: Metal (Apple Silicon) not working

**Solution:**
1. Install jax-metal:
```bash
pip install jax-metal
```

2. Set the environment variable:
```bash
export JAX_PLATFORMS=metal,cpu
```

3. Verify in Python:
```python
import jax
print(jax.devices())  # Should show Metal devices
```

### Dependency Conflicts

#### Issue: `pkg_resources.VersionConflict`

**Solution:**
Create a fresh virtual environment:
```bash
python -m venv probjax_env
source probjax_env/bin/activate  # On Windows: probjax_env\Scripts\activate
pip install -e probjax
```

## Runtime Errors

### Shape Errors

#### Issue: `ValueError: Incompatible shapes`

**Common causes:**
- Input shape doesn't match expected shape
- Batch dimensions not aligned
- Tree structures don't match

**Debug steps:**
```python
from jax import eval_shape
import jax.tree_util as jtu

# Check expected shapes
print(eval_shape(fn, *args))

# Check tree structure
print(jtu.tree_structure(args))
print(jtu.tree_structure(expected_structure))
```

### JAX Tracer Errors

#### Issue: `TracerArrayConversionError`

**Cause:** Trying to convert JAX tracers to NumPy arrays inside a JIT-compiled function.

**Solution:**
Move the conversion outside the JIT:
```python
# Bad
@jax.jit
def bad_fn(x):
    return np.array(x)  # This fails

# Good
def good_fn(x):
    return jnp.array(x)  # Use JAX arrays

# Or convert outside JIT
result = jax.jit(good_fn)(x)
np_result = np.array(result)  # Convert after
```

#### Issue: `UnexpectedTracerError`

**Cause:** Using a traced value outside its tracing context.

**Solution:**
Make sure all operations happen within the same JIT context or use `jax.lax` control flow:
```python
# Bad
def bad_fn(x):
    y = jax.jit(inner)(x)
    return y + x  # x is a tracer, y is concrete

# Good
@jax.jit
def good_fn(x):
    y = inner(x)
    return y + x
```

### Numerical Issues

#### Issue: `NaN` gradients or values

**Common causes:**
- Log(0) or division by zero
- Overflow/underflow
- Bad initialization

**Solutions:**
1. Add small epsilon to denominators:
```python
result = x / (y + 1e-8)
```

2. Use stable log-space operations:
```python
from jax.scipy.special import logsumexp
```

3. Check for NaN with `jax.debug.print`:
```python
jax.debug.print("x = {x}", x=x)
```

#### Issue: Divergent transitions in MCMC

**Solutions:**
1. Decrease step size:
```python
kernel = mcmc.hmc(..., step_size=0.01)  # Smaller step size
```

2. Increase number of integration steps
3. Use adaptive HMC
4. Check your model for numerical issues

### Memory Issues

#### Issue: Out of memory (OOM)

**Solutions:**
1. Reduce batch size
2. Use gradient checkpointing
3. Use `jax.clear_caches()` to free memory
4. Use XLA memory management:
```python
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```

#### Issue: Slow compilation

**Solutions:**
1. Use `jax.jit` with static arguments properly marked
2. Cache compiled functions
3. Simplify control flow

## Distribution Issues

### Issue: `NotImplementedError` for distribution methods

**Cause:** Not all distributions implement all methods.

**Solution:**
Check available methods:
```python
from probjax import stats as dist

# Check what's available
dist = dist.Normal(0.0, 1.0)
print(dist.sample)  # Should work
print(dist.mean)    # May not be implemented for all distributions
```

### Issue: Wrong results from transformed distributions

**Cause:** Incorrect transformation function.

**Solution:**
Make sure your transformation is bijective (invertible):
```python
from probjax import transformed

# Good: exp is bijective on positive reals
log_normal = transformed(
    dist.Normal(0.0, 1.0),
    transform=jnp.exp,
    inverse_transform=jnp.log  # Provide inverse
)
```

## Inference Issues

### MCMC Issues

#### Issue: Low acceptance rate

**Solutions:**
1. Decrease step size
2. Tune mass matrix
3. Use adaptive methods

#### Issue: High autocorrelation

**Solutions:**
1. Increase thinning
2. Run chain longer
3. Try different sampler (e.g., NUTS instead of HMC)

### SMC Issues

#### Issue: ESS (Effective Sample Size) drops to 1

**Cause:** Tempering schedule too aggressive.

**Solution:**
Use adaptive tempering:
```python
from probjax.inference.smc import adaptive_smc

kernel = adaptive_smc(target_ess=0.5)  # Tune target ESS
```

## Neural Network Issues

### Issue: Training loss not decreasing

**Solutions:**
1. Check learning rate (try 1e-4 to 1e-3)
2. Verify data preprocessing
3. Check for NaN/Inf in gradients
4. Use gradient clipping:
```python
from optax import clip_by_global_norm

optimizer = optax.chain(
    clip_by_global_norm(1.0),
    optax.adam(learning_rate)
)
```

### Issue: Flow training unstable

**Solutions:**
1. Use bounded flows (e.g., Rational Quadratic Spline)
2. Initialize with identity transformation
3. Use smaller learning rate
4. Add regularization

## Debugging Tips

### Enable Debug Mode

```python
import jax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", True)  # Disable JIT for debugging
```

### Print Intermediate Values

```python
from jax import debug

def f(x):
    debug.print("x = {x}", x=x)
    return x * 2
```

### Profile Your Code

```python
import jax.profiler

with jax.profiler.trace("/tmp/prof"):
    result = fn(x)
```

### Check Gradients

```python
from jax import grad, value_and_grad

# Check if gradients exist
grad_fn = grad(fn)
print(grad_fn(x))

# Check gradient values
value, grads = value_and_grad(fn)(x)
print("Value:", value)
print("Grads:", grads)
```

## Getting Help

If you can't resolve your issue:

1. **Check the FAQ**: See [FAQ](faq.md) for common questions
2. **Check examples**: Look at the `examples/` directory
3. **Open an issue**: On GitHub with:
   - Minimal reproducible example
   - Full error traceback
   - Environment details (Python, JAX, OS versions)
   - What you've tried

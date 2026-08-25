---
title: Installation
---

# Installation

ProbJax requires Python 3.11 or newer. It is currently documented as a source
installation rather than as a published-package workflow.

## Source Installation

Clone the repository and install it from the checkout root:

```bash
git clone https://github.com/mackelab/probjax.git
cd probjax
python -m pip install -e .
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

## Accelerator Backends

JAX backend support depends on your operating system, accelerator, drivers,
and the compatibility requirements published by JAX.

### NVIDIA CUDA 12

```bash
python -m pip install -e ".[cuda12]"
```

### Apple Silicon Metal

```bash
python -m pip install -e ".[metal]"
```

Select Metal before importing JAX:

```bash
export JAX_PLATFORMS=metal,cpu
```

or:

```python
import os

os.environ["JAX_PLATFORMS"] = "metal,cpu"

import jax

print(jax.devices())
```

The Metal extra exposes `jax-metal`, but Metal is not covered by the
repository's Linux test environment. Consult the current `jax-metal`
compatibility documentation for supported macOS, Python, and JAX versions.

## Development Installation

```bash
python -m pip install -e ".[dev]"
```

The development extra includes pytest, pytest-benchmark, pytest-xdist, Ruff,
and SciPy. Common checks are:

```bash
pytest
ruff check .
ruff format --check .
```

## Declared Requirements

- Python >= 3.11
- JAX and jaxlib >= 0.4.34
- NumPy >= 2.0.0
- Flax >= 0.12.0

See [`pyproject.toml`](../pyproject.toml) for the complete dependency set.

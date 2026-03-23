---
title: Installation
---

# Installation

## Basic Installation

Install ProbJax directly from the repository:

```bash
pip install -e probjax
```

### Using uv (Recommended)

For faster and more reliable Python package management, use [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e probjax
```

## GPU Support

### CUDA 12 (NVIDIA GPUs)

For GPU acceleration with NVIDIA CUDA 12:

```bash
pip install -e "probjax[cuda12]"
```

### Metal (Apple Silicon)

For Apple Silicon (M1/M2/M3) GPU acceleration:

```bash
pip install -e "probjax[metal]"
```

Then configure JAX to use the Metal backend:

```bash
# Bash/Zsh
export JAX_PLATFORMS=metal,cpu
```

Or in Python before importing JAX modules:

```python
import os
os.environ["JAX_PLATFORMS"] = "metal,cpu"

import jax
print(jax.devices())  # Should list Metal devices
```

**Requirements:**
- macOS 12+ on Apple Silicon (M1/M2/M3)
- Recent Xcode Command Line Tools
- Python 3.9–3.12

## Development Installation

For development and testing:

```bash
pip install -e "probjax[dev]"
```

This includes additional dependencies:
- pytest
- pytest-benchmark
- ruff
- pytest-xdist
- scipy

## Requirements

- Python >= 3.11
- JAX >= 0.4.34
- NumPy >= 2.0.0

See `pyproject.toml` for the complete list of dependencies.

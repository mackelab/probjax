# Probjax

Probabilistic computation in JAX. This library is under active development and is not yet ready for use. It aims to provide a simple and flexible way to build probabilistic models and perform inference in then. It provides the following set of tools:
- **Core**: A set of core function transformations and primitives useful for building probabilistic models.
    - **Traceing**: Tracing and manipulation of function traces. (Very incomplete)
    - **Automatic inversion**: Automatic inversion of functions. (Rather complete, with some limitations)
    - **Automatic log_prob**: Automatic computation of log-probabilities (Rather incomplete). Automatic computation of log-probabilities of transformed distributions (Rather complete, through automatic inversion and logdet).
- **Distributions**: A set of distributions with support for sampling, log-probability and more.
- **Inference**: Some inference algorithms. (incomplete)
- **Neural networks**: Some neural network layers and models. Based on [Haiku](www.github.com/deepmind/dm-haiku). Here a classical layers as Transformers, Resnets or U-Nets. But also specialised layers for normalising flows, such as coupling layers, autoregressive layers, etc. (complete)
- **Utilities**: Some utilities for numerical computation i.e. odeint, sdeint, etc. (complete)

## Installation

Probjax can be installed using pip:

```bash
pip install -e probjax
```

Additionally, you can install benchmark scripts using:

```bash
pip install -e probjax/scoresbibm
```

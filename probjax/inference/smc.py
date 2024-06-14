import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax._src.util import safe_map as map
from .marcov_kernels import MCMCKernel, MCMCState, GaussianKernel

from typing import Any, Callable, Tuple, Union, Sequence
from jaxtyping import PyTree, Array

from functools import partial
from itertools import accumulate


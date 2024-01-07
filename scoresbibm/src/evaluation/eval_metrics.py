
from sbibm import get_task as _get_torch_task

import jax 
import jax.numpy as jnp
import numpy as np
import torch


from sbi.analysis.sbc import c2st as _c2st

from probjax.distributions.divergences import wasserstein_distance


def c2st(x, y, rng=None,**kwargs):
    x = torch.as_tensor(np.array(x))
    y = torch.as_tensor(np.array(y))
    return float(_c2st(x, y, **kwargs))


def wasserstein2(x, y, rng=None, mc_samples=None, **kwargs):
    x = jnp.array(np.array(x))
    y = jnp.array(np.array(y))
    
    return float(wasserstein_distance(x, y, mc_samples=mc_samples, key=rng, **kwargs))



def get_metric(name:str):
    if "c2st" in name:
        return c2st
    elif name == "wasserstein2":
        return wasserstein2
    else:
        raise NotImplementedError()
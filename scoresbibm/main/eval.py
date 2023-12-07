
from sbibm import get_task as _get_torch_task

import jax 
import jax.numpy as jnp
import numpy as np
import torch


from sbi.analysis.sbc import c2st as _c2st


def c2st(x, y, **kwargs):
    x = torch.as_tensor(np.array(x))
    y = torch.as_tensor(np.array(y))
    return float(_c2st(x, y, **kwargs))




def get_metric(name:str):
    if name == "c2st":
        return c2st
    else:
        raise NotImplementedError()
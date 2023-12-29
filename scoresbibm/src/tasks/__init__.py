

from src.tasks.sbibm_tasks import LinearGaussian, MixtureGaussian, TwoMoons, SLCP
from src.tasks.multi_task import TwoMoonsAllConditionalTask, SLCPAllConditionalTask, NonlinearGaussianTreeAllConditionalTask, NonlinearMarcovChainAllConditionalTask


def get_task(name: str, backend: str = "torch"):
    if name == "gaussian_linear":
        return LinearGaussian(backend=backend)
    elif name == "gaussian_mixture":
        return MixtureGaussian(backend=backend)
    elif name == "two_moons":
        return TwoMoons(backend=backend)
    elif name == "slcp":
        return SLCP(backend=backend)
    elif name == "two_moons_all_cond":
        return TwoMoonsAllConditionalTask(backend=backend)
    elif name == "slcp_all_cond":
        return SLCPAllConditionalTask(backend=backend)
    elif name == "tree_all_cond":
        return NonlinearGaussianTreeAllConditionalTask(backend=backend)
    elif name == "marcov_chain_all_cond":
        return NonlinearMarcovChainAllConditionalTask(backend=backend)
    else:
        raise NotImplementedError()
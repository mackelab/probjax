"""Twisted SMC steering a frozen diffusion model toward a potential.

Prior ``p_phi`` is a (here randomly initialized) VP ``DiffusionDenoiser``;
the target is ``p_phi(x) * exp(-0.5 * ||x - y||^2)``. Compares the SMC
particle mean against plain ancestral sampling from ``as_dist``.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from probjax.inference import diffusion_smc, run
from probjax.inference.neural import diffusion_guide
from probjax.nn.generative.diffusion import VP


class Net(nnx.Module):
    def __call__(self, t, x):
        return jax.tree.map(lambda leaf: leaf * 0.1, x)


def main() -> None:
    key = jax.random.key(0)
    dim, target = 2, 1.0
    model = VP(Net(), event_spec=dim, num_steps=16)

    def log_potential(x):
        return -0.5 * jnp.sum((x - target) ** 2)

    prior = model.as_dist(mode="sde").sample(key, (2000,))
    print(f"prior mean:            {prior.mean(axis=0)}")

    guide, params = diffusion_guide(model, log_potential)
    times = model.solver_cfg.solve_schedule(
        model.train_cfg.t_min, model.train_cfg.t_max, 16
    )
    kernel = diffusion_smc(guide, times)
    state = kernel.init(jax.random.key(1), 2000, params)
    result = run(
        jax.random.key(2), kernel, state, params, len(times) - 1, collect=("resampled",)
    )
    weights = jnp.exp(result.state.log_weights)
    mean = jnp.sum(weights[:, None] * result.state.particles, axis=0)
    print(f"twisted SMC mean:      {mean}")
    print(f"log evidence:          {float(result.state.log_evidence):.4f}")
    print(f"resample steps:        {int(jnp.sum(result.trace['resampled']))}")


if __name__ == "__main__":
    main()

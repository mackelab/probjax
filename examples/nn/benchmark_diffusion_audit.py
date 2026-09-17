"""CPU/GPU-selectable audit: Gaussian accuracy and synchronized sampler runtime.

Run with PYTHONPATH=. JAX_PLATFORMS=cpu python examples/nn/benchmark_diffusion_audit.py.
Compilation/first execution is measured separately from five warm calls.
The exact Gaussian denoiser isolates solver error from model training error.
"""

import argparse
import json
import platform
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from probjax.nn.generative.diffusion import EDM, VE, VP, CosineDM


class Zero(nnx.Module):
    def __call__(self, t, x, **kwargs):
        return jnp.zeros_like(x)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--profile', choices=['gaussian', 'mixture'], default='gaussian'
    )
    parser.add_argument('--steps', nargs='+', type=int, default=[16, 64, 256])
    args = parser.parse_args()
    rows = []
    for cls in (EDM, VE, VP, CosineDM):
        for mode in ('ode', 'sde'):
            for steps in args.steps:
                if args.profile == 'mixture':

                    class Oracle(cls):
                        def denoise(self, t, x, *args, **kwargs):
                            a, s = self.scale_fn(t), self.std_fn(t)
                            variance = a * a * 0.25 + s * s
                            gain = a * 0.25 / variance
                            mean = 3 * jnp.tanh(3 * a * x / variance)
                            return gain * x + (1 - gain * a) * mean

                    model = Oracle(
                        Zero(), event_spec=1, num_steps=steps, std0=float(np.sqrt(9.25))
                    )
                else:
                    model = cls(Zero(), event_spec=1, num_steps=steps)
                dist = model.as_dist(mode=mode)
                call = jax.jit(partial(dist.sample, shape=(4096,)))
                key = jax.random.key(42)
                begin = time.perf_counter()
                dist.compile()
                values = call(key).block_until_ready()
                cold = time.perf_counter() - begin
                durations = []
                for _ in range(5):
                    begin = time.perf_counter()
                    call(key).block_until_ready()
                    durations.append(time.perf_counter() - begin)
                target = float(jnp.squeeze(model.marginal_std(model.train_cfg.t_min)))
                rows.append(
                    dict(
                        model=cls.__name__,
                        mode=mode,
                        steps=steps,
                        cold_s=cold,
                        warm_ms=1000 * float(np.median(durations)),
                        mean=float(values.mean()),
                        std=float(values.std()),
                        mean_abs=float(jnp.abs(values).mean()),
                        central_mass=float((jnp.abs(values) < 1).mean()),
                        relative_std_error=float(values.std()) / target - 1,
                    )
                )
                print(rows[-1], flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                jax=jax.__version__,
                python=platform.python_version(),
                devices=[str(d) for d in jax.devices()],
                profile=args.profile,
                batch=4096,
                repeats=5,
                results=rows,
            ),
            indent=2,
        )
        + '\n'
    )


if __name__ == '__main__':
    run()

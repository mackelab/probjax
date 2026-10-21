# Configuration benchmarks

Small, reproducible checks accompany the neural-model configuration docs. The
tables below are measured results; commands run from the repository root. All
runners live in `docs/scripts/`, and raw JSON measurements live alongside these
pages. Historical JSON retains the settings used when it was measured.

| Configuration family | Results and defaults | Runner |
| --- | --- | --- |
| Transformer, AR transformer, diffusion transformer | [Standard/compact profiles](nn_defaults.md) | `benchmark_nn_defaults.py` |
| NSF feature mixing and spline tails | [Mixing and tail controls](flow_mixing.md) | `benchmark_flow_mixing.py`, `diagnose_flow_mixing.py` |
| Gaussian, mixture, heavy-tailed, skewed, bounded data | [Representative flow workloads](flow_mixing_realistic.md) | `benchmark_flow_mixing.py --suite realistic` |
| Continuous diffusion schedules, training targets and solvers | [Diffusion config matrix](diffusion_configs.md) | `benchmark_diffusion_configs.py` |
| Categorical schedules and training configs | [Categorical controls and updated defaults](diffusion_configs.md#updated-categorical-defaults) | `benchmark_diffusion_configs.py --phase categorical` |
| Earlier numerical/precision regression measurements | [Diffusion and token-scaling audit](diffusion_audit.md) | `benchmark_diffusion_audit.py`, `benchmark_transformer_audit.py` |
| DeepSet, induced (ISAB), latent (MLA) and full self-attention on set tasks | [Set transformer benchmark](set_transformers.md) | `benchmark_set_transformers.py` |

## Quick smoke checks

These reduced runs check that a configuration executes; use the full commands
on each results page for meaningful quality comparisons. Write smoke results
to distinct filenames so the published measurements remain intact.

```bash
# Forward and gradient timing for small, compact and standard transformer profiles.
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_nn_defaults \
  --tokens 8 --batch 1 --repeats 1 --output docs/benchmarks/nn_smoke.json

# One short NSF target across the mixing choices.
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_flow_mixing \
  --targets crescent2 --steps 50 --seeds 1 --train-size 1024 --test-size 256 \
  --output docs/benchmarks/flow_smoke.json

# A few oracle diffusion solver cases, without training a network.
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_diffusion_configs \
  --phase oracle --sample-steps 16 --samples 256 --repeats 1 --limit 8 \
  --output docs/benchmarks/diffusion_smoke.json

# All categorical schedule kinds, with a short training budget.
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_diffusion_configs \
  --phase categorical --sample-steps 16 --samples 256 --train-steps 50 --seeds 1 \
  --output docs/benchmarks/categorical_smoke.json

# Set-learning tasks and ISAB invariance audit, reduced budget.
JAX_PLATFORMS=cpu python -m docs.scripts.benchmark_set_transformers \
  --steps 50 --seeds 1 --batch-size 16 --set-size 32 --num-inducing 8 \
  --model-dim 32 --tokens 32 128 --output docs/benchmarks/set_smoke.json
```

The runners synchronize device work and separate first-call compilation from
warm timing. Seeds, shapes, numerical budgets, and finite-value flags are saved
with the results. Synthetic oracle tests isolate numerical error; learned-model
tests additionally reflect training and model-capacity error. CPU timings do
not establish GPU performance, and short synthetic studies are not universal
rankings of model quality.

## Inspect saved measurements

```bash
python -m docs.scripts.benchmark_flow_mixing --summary-only \
  --output docs/benchmarks/flow_mixing_realistic.json
python -m docs.scripts.benchmark_diffusion_configs --phase learned --summary
python -m docs.scripts.benchmark_diffusion_configs --phase categorical --summary \
  --output docs/benchmarks/diffusion_defaults_categorical.json
```

See [configuration reference](../reference/nn.md) for constructors and
[migration notes](../guides/migration.md) for reproducing older checkpoints.

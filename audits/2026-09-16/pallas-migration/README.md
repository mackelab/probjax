# Pallas feature migration and legacy removal

The frozen `probjax/nn/pallas_kernels/old_pallas` package has been removed.
Before removal, its public function/class inventory and callable signatures were
compared to the live package: [inventory](api_inventory.json). No public kernel
symbol or signature was missing. This inventory is structural evidence, not a
substitute for numerical/hardware validation.

## Feature disposition

| Area | Canonical implementation and retained features |
| --- | --- |
| Attention | `kernels/attention.py`: BlockSizes/tuning, masks, bias pytrees, block sparsity, both dropout implementations, reverse AD, forward JVP, batching, batch/head sharding. |
| Flash3 | `kernels/flash_attention3.py`: ordinary/pipeline forward, residual outputs, custom backward, batch/head sharding. **Ported missing public-API forward and VJP fallbacks** for JAX builds without private kernel helpers. GPU/causal compatibility guards remain. |
| Kernel matrix products | `kernels/kernel_mv.py`: generic kernels/VJPs, RBF, naive references, KDE helpers; primitive forward supports batch/query-row sharding. |
| Mamba | `kernels/mamba.py`: same raw recurrence kernels and forward/backward signatures; primitive partitioning retains batch-only sharding. CPU associative-scan support remains. |
| SSD | `kernels/ssd.py`: same raw forward/backward kernels, grouped heads, initial state, linear-scan utilities; primitive partitioning retains batch/head/group sharding. Forward-mode support remains in the new implementation. |
| Masks/biases | Already shared with the live `attention_mask_bias.py`; the legacy file was an import shim. |
| Common utilities | Legacy and live `kernel_utils/common.py` were identical. |
| Partitioning | Legacy `cp_utils.py` heuristics and hand-built partition wrappers are replaced by declarative specs and the shared `core.custom_primitives.sharded_primitive` machinery. Golden sharding-rule tests remain. |

The implementation bodies of the shared raw SSD and Mamba kernels matched
structurally before deletion; their differences were in the public dispatch and
partitioning wrappers. Public API signatures also matched for attention,
Flash3, and kernel matrix products.

## Additional correctness fixes

The old attention A/B JVP test accepted matching NaNs from both implementations.
Replacing that comparison with a mathematical reference exposed two real bugs:

1. The JVP kernel loaded the log-normalizer with a global query offset even
   though its BlockSpec had already sliced the query tile. Later query tiles
   read outside the tile.
2. The softmax directional normalization was computed separately for each key
   tile. It must aggregate across all keys before subtracting its contribution
   from the output tangent.

Both are corrected in the live JVP kernel. The replacement tests require finite
results and compare multi-tile forward/JVP results with dense attention,
including causal masks, dense bias, and materialized dropout. Ordinary
forward/reverse kernels are unaffected by this correction.

## Coverage preserved after removal

`tests/test_pallas_old_vs_new.py` was replaced by
`tests/test_pallas_references.py`. Attention and RBF matrix products now compare
against independent dense/naive formulas; SSD compares against a pure-JAX
recurrence and Mamba against a sequential recurrence. Existing accelerator-only
forward/backward/JVP cases remain. Flash3 public-API fallback tests use backend
stubs on CPU.

The source tree has no remaining imports or references to the retired package
outside historical audit documentation. All 11 legacy Python files (7,859
lines) and their cache directory were removed; no forwarding legacy package is
left behind.

## Validation

Logs are stored beside this report. Main CPU environment: JAX 0.10.0. Supported
JAX 0.9.2 was tested separately with its own virtual environment.

- Replacement reference tests: **12 passed, 3 accelerator cases skipped**.
- JAX 0.9.2 kernel/reference/AD/helper suite: **53 passed, 5 skipped, 1 xpassed**.
  The xpass is an existing non-strict expected numerical failure in the generic
  kernel matrix-product test; no new expected-failure markers were introduced.
- Existing attention JVP/dropout/Flash3 compatibility selection: **19 passed,
  5 hardware-gated cases skipped**.
- Host-device mesh tests cover batch/head-sharded forward, reverse gradients,
  and JVP, with a forward HLO check against all-gathers. See the mesh logs.
- Full test collection succeeded after removal; collection is not execution of
  the full suite.
- Undefined-name/syntax lint and whitespace checks passed.

**GPU numerical execution remains unverified here.** `nvidia-smi` could not
communicate with the NVIDIA driver. CPU Pallas interpretation and host-device
sharding are useful validation but do not establish GPU lowering correctness,
GPU performance, or multi-GPU runtime parity.

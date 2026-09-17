# Worktree consolidation for PR #29

All local worktrees and branches were compared with origin/main at 629d9da.
The original main checkout and its uncommitted changes are preserved.

| Source | Disposition |
| --- | --- |
| codex/complementary-special, 75d66c5 | Special functions, NN/fit/generative changes, audit fixes, Pallas migration, docs and benchmarks |
| fix/attention-consumer-gpu-smem, bf285bc..83373b6 | Fourteen earlier shared-helper refactors integrated, with overlap resolved in favor of the later audited behavior |
| refactoring, d17feea | Unresolved reduction inverse guard and removal of obsolete select-n xfail already present in the newer implementation; retain the newer corrected logdet test |
| audit/pallas-cleanup, 4e5bff9 | GPU shape and tile arguments ported to test_pallas_references.py; legacy A/B test stays deleted |
| main working changes | CI coverage job/Codecov configuration and deterministic/explicit test checks copied without changing the original checkout |
| review-pr-14, review-pr-16, review-pr-17, ci/github-pages-docs, audit/full-codebase-audit | Already ancestors of main |
| origin/fix/attention-consumer-gpu-smem | No commits missing from main |

## Conflict resolutions

- Preserve complementary inverse-beta/gamma solvers and their derivative/tail fixes;
  older initial-guess extraction is superseded by these implementations.
- Preserve SMC evidence, resumability and adaptive execution while sharing progress
  reporting through RunnerMixin. Shared scan refactors already represented by
  the newer _run_fixed implementation are not duplicated.
- Preserve Improved MeanFlow, event specifications, protocol placement and
  validated affine rules while adopting shared flow-loss and taint-walk helpers.
- Preserve normalized Gaussian likelihoods in the shared innovation helper.
- Restore missing merge_dict_state, RunnerMixin, _gaussian_unpack and
  infer_noise_dim helpers referenced by the refactor branch.
- Skip literals when forwarding variable metadata into nested inverse programs;
  this resolves the two failures in the initial PR CI run.

## Pallas follow-up

The older GPU audit identified Mamba's cross-tile carry race. GPU programs do not
provide the ordered grid execution used by the TPU kernel. Multi-tile GPU calls
now use the differentiable associative scan; the fused single-tile path remains.
CPU dispatch regression tests compare values and all six parameter gradients to
a sequential recurrence for both sequence and dimension tiling.

GPU SSD tests now use supported sequence/head dimensions. Their numerical
reference assertions remain strict; the older A/B audit's permissive JVP
threshold is not adopted without a passing strict reference comparison.
Actual accelerator execution and GPU fallback performance require available
hardware. The local NVIDIA driver is unavailable.

## Validation

Validation logs and final CI status are recorded in the PR. The initial strict
docs build passed. The nested-inverse regression selection passed (7 tests),
and the Mamba multi-tile dispatch/gradient regressions passed (2 tests).

The consolidated CI run (35139676908) completed the full CPU suite and reported
3,611 passes, 837 skips, 10 expected failures, one non-strict unexpected pass,
and five stale test expectations. Two strict xfail markers covered newly
supported inverse operations; three Mamba dispatch mocks assumed the former
multi-tile Pallas path. The follow-up removes those two obsolete markers and
uses single-tile inputs for the fused-dispatch tests, retaining the separate
multi-tile value/gradient regressions. Coverage instrumentation reported 73.53%.

The revised documentation harness executes each page once, preserving cumulative
state and block-level error reporting; 12 checks passed and 3 skipped in 34.36s.
The final multi-device attention selection passed both tests (5.82s).

GitHub's main ruleset requires one approving review and approval of the latest
push. Auto-merge is disabled. No repository protection settings were changed.

Final affected-file regression suite: 58 passed, 3 skipped, 2 deselected,
9 expected failures and 1 pre-existing non-strict unexpected pass (51.63s).

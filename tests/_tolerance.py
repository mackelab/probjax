"""Tolerances for float32 kernel-vs-reference comparisons.

On CPU float32 matmuls are exact float32 and the tests keep their tight
tolerances.  On GPUs XLA's default float32 matmul precision is TF32 (10-bit
mantissa), and the Pallas kernels (run in interpret mode by these tests) pin
their own dot precision, so forcing ``jax_default_matmul_precision="highest"``
would only make the reference exact.  Kernel and reference therefore differ at
the TF32 level (~1e-3 on an H100); ``fp32_tol`` keeps the given tolerance on CPU
and widens it to that level on GPU.
"""

from __future__ import annotations

import jax

TF32_ATOL = 5e-3
TF32_RTOL = 5e-3
# Accumulated TF32 error grows with the operands, not with each output element:
# gradients of magnitude ~30 differ by ~3e-2 (about 1e-3 of the tensor's scale).
TF32_SCALE_TOL = 2e-3


def on_tf32_backend() -> bool:
    return jax.default_backend() == "gpu"


def fp32_tol(atol: float, rtol: float = 1e-5, *, scale=None) -> dict[str, float]:
    """``dict(atol=..., rtol=...)`` for ``allclose``: unchanged on CPU, TF32 level on GPU.

    ``scale`` (e.g. the reference's max magnitude) widens the GPU ``atol`` to
    ``TF32_SCALE_TOL * scale`` for outputs that accumulate long TF32 reductions.
    """
    if on_tf32_backend():
        floor = TF32_ATOL if scale is None else max(TF32_ATOL, TF32_SCALE_TOL * float(scale))
        return {"atol": max(atol, floor), "rtol": max(rtol, TF32_RTOL)}
    return {"atol": atol, "rtol": rtol}

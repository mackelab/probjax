"""A progress bar for compiled scans.

The bar lives on the host. The compiled program reports to it through an
unordered ``jax.debug.callback`` at most ``num_updates`` times (and at the last
step), so it costs one cheap callback every few steps and never waits for the
host. Ordered callbacks would serialise the program and are not allowed under
``vmap``.

Unordered reports can arrive late, and under ``vmap`` every chain reports the
same step, so the host only ever moves the bar forward: it ignores reports
older than, or equal to, what it has shown. Redrawing is throttled by wall
time, and the last step always draws the full bar and ends the line.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["ProgressBar", "progress_scan"]


def _format_time(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressBar:
    """Host-side state of one progress bar; ``update`` is the callback.

    A compiled program reuses the bar on every call, so a report of
    ``first_step`` starts it afresh.
    """

    def __init__(
        self,
        total: int,
        names: Sequence[str] = (),
        first_step: int = 0,
        width: int = 30,
        min_interval: float = 0.1,
    ):
        self.total = total
        self.first_step = first_step
        self.names = tuple(names)
        self.width = width
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = -1
        self._start = 0.0
        self._last_draw = 0.0

    def update(self, step: Any, stats: Sequence[Any] = ()) -> None:
        # Under vmap each chain reports separately, with scalar values.
        step = int(np.asarray(step).reshape(-1)[0])
        now = time.perf_counter()
        with self._lock:
            if step == self.first_step and self._last != step:
                self._start, self._last_draw = now, 0.0
            elif step <= self._last:
                return
            self._last = step
            final = step == self.total - 1
            if not final and now - self._last_draw < self.min_interval:
                return
            self._last_draw = now
            line = self._render(step + 1, now - self._start, stats)
            sys.stdout.write(f"\r{line}" + ("\n" if final else ""))
            sys.stdout.flush()

    def _render(self, done: int, elapsed: float, stats: Sequence[Any]) -> str:
        filled = self.width * done // self.total
        bar = "█" * filled + "-" * (self.width - filled)
        line = f"|{bar}| {done}/{self.total} [{_format_time(elapsed)}"
        # The clock starts at the first report, so these exclude compilation.
        timed = done - self.first_step - 1
        if timed > 0 and elapsed > 0:
            rate = timed / elapsed
            line += f"<{_format_time((self.total - done) / rate)}, {rate:.1f} it/s"
        line += "]"
        for name, value in zip(self.names, stats, strict=False):
            line += f" {name}: {float(np.asarray(value).reshape(-1)[0]):.3g}"
        return line


def progress_scan(
    body: Callable,
    carry: Any,
    xs: Any,
    length: int,
    *,
    names: Sequence[str] = (),
    stats_fn: Callable | None = None,
    num_updates: int = 50,
) -> tuple[Any, Any]:
    """``jax.lax.scan(body, carry, xs, length)`` with a progress bar.

    The steps run in chunks of about ``length / num_updates``, each an inner
    scan followed by one report, so the steps themselves carry no extra work.

    Args:
        body, carry, xs, length: As for :func:`jax.lax.scan`.
        names: Names of the stats shown next to the bar.
        stats_fn: ``y -> tuple`` of one scalar per name, read from each step's
            output; the bar shows their mean over the last chunk.
        num_updates: How often the program reports to the host, at most.
    """
    chunk = -(-length // min(num_updates, length))
    num_chunks, rest = divmod(length, chunk)
    bar = ProgressBar(length, names, first_step=(chunk if num_chunks else rest) - 1)

    def run_chunk(carry, xs, size, done):
        carry, ys = jax.lax.scan(body, carry, xs, size)
        stats = ()
        if names:
            stats = tuple(jnp.mean(jnp.asarray(v, jnp.float32)) for v in stats_fn(ys))
        jax.debug.callback(bar.update, done + size - 1, stats)
        return carry, ys

    def outer(state, xs):
        done, carry = state
        carry, ys = run_chunk(carry, xs, chunk, done)
        return (done + chunk, carry), ys

    split = num_chunks * chunk
    head = jax.tree.map(
        lambda x: x[:split].reshape((num_chunks, chunk) + x.shape[1:]), xs
    )
    (done, carry), ys = jax.lax.scan(outer, (jnp.int32(0), carry), head, num_chunks)
    ys = jax.tree.map(lambda y: y.reshape((split,) + y.shape[2:]), ys)
    if rest:
        tail = jax.tree.map(lambda x: x[split:], xs)
        carry, tail_ys = run_chunk(carry, tail, rest, done)
        ys = jax.tree.map(lambda a, b: jnp.concatenate([a, b]), ys, tail_ys)
    return carry, ys

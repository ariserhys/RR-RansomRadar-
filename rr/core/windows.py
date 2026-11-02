"""Overlapping (sliding) window construction over a normalised event stream.

The previous generation of RansomRadar used 10-second *tumbling* windows, which
imposes a hard 10-second floor on detection latency: no matter how obvious the
burst, no decision exists until the window closes. This module emits windows of
``size`` seconds advancing by ``hop`` seconds (default 1.0 s), so a decision can
be produced every hop while retaining the statistical mass of a long window.

No I/O, no platform-specific code, fully deterministic (no stochastic APIs).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Iterator

from rr.core.schema import NormalisedEvent, Window

__all__ = ["sliding_windows"]


def sliding_windows(
    events: Iterable[NormalisedEvent],
    size: float,
    hop: float,
    fill_idle: bool = True,
) -> Iterator[Window]:
    """Yield overlapping half-open ``[start, start + size)`` windows.

    Semantics:

    * **Grid anchoring.** Starts are anchored to a global grid,
      ``start_0 = floor(first_ts / hop) * hop``, then advance by ``hop``. The
      grid depends only on ``first_ts`` and ``hop``, so replays of the same
      stream are byte-for-byte reproducible and windows from different scales
      line up on shared boundaries.
    * **Coverage.** Window starts run while ``start <= last_ts``, so every event
      (including the last) is covered by at least one window.
    * **Membership.** An event belongs to a window iff ``start <= ts < end``.
      An event therefore appears in up to ``ceil(size / hop)`` windows.
    * **Idle filling.** With ``fill_idle=True`` (default) *every* grid window
      between the first and last event is emitted, including empty ones, so the
      model observes that quiet periods are normal and cannot learn "any window
      at all implies activity". With ``fill_idle=False`` empty windows are
      skipped.

    Complexity: **O(n + W)** time and **O(k)** extra space, where ``n`` is the
    number of events, ``W`` the number of emitted windows, and ``k`` the maximum
    number of events concurrently inside one window. Events are never rescanned:
    a right pointer admits each event exactly once and a left pointer evicts it
    exactly once (both are monotonically non-decreasing because window starts
    increase), with the live set held in a deque.

    Raises:
        ValueError: If ``size`` or ``hop`` is not a positive finite number.

    Edge cases:
        * Empty input yields nothing.
        * A single event yields exactly one window.
        * Events sharing one timestamp also yield exactly one window start.
    """
    size = float(size)
    hop = float(hop)
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError(f"size must be a positive finite number, got {size!r}")
    if not math.isfinite(hop) or hop <= 0.0:
        raise ValueError(f"hop must be a positive finite number, got {hop!r}")

    ordered = sorted(events, key=lambda e: e.ts)
    n = len(ordered)
    if n == 0:
        return

    first_ts = ordered[0].ts
    last_ts = ordered[-1].ts

    start = math.floor(first_ts / hop) * hop
    n_windows = int(math.floor((last_ts - start) / hop)) + 1
    n_windows = max(n_windows, 1)

    live: deque[NormalisedEvent] = deque()
    right = 0

    for step in range(n_windows):
        w_start = start + step * hop
        w_end = w_start + size

        # Admit newly visible events (each event admitted at most once).
        while right < n and ordered[right].ts < w_end:
            live.append(ordered[right])
            right += 1

        # Evict events that fell out of the left edge (each evicted at most once).
        while live and live[0].ts < w_start:
            live.popleft()

        if not live and not fill_idle:
            continue

        yield Window(
            start=w_start,
            end=w_end,
            scale=size,
            events=tuple(live),
        )

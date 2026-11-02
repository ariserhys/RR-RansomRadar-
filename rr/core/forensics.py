"""Cheap byte-level forensic statistics.

Pure numpy (scipy is intentionally not required here so the collector hot path
stays dependency-light). No I/O at import or call time.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BYTE_VALUES", "chi_square_uniformity"]

#: Number of distinct byte values; also the number of histogram bins.
BYTE_VALUES: int = 256


def chi_square_uniformity(sample: bytes) -> float:
    """Chi-square goodness-of-fit of a byte sample against a uniform model.

    The 256 byte-value counts of ``sample`` are compared with the uniform
    expectation ``len(sample) / 256``::

        X^2 = sum_i (observed_i - expected)^2 / expected

    Interpretation (direction matters):

    * **Uniform / encrypted / compressed bytes -> LOW statistic.** The
      expectation under the null is ``df = 255``, so a 4096-byte sample of
      good random data scores roughly 255 +- 23.
    * **Structured or constant bytes -> HIGH statistic.** An all-one-value
      4096-byte buffer scores ``255 * 4096`` ~ 1.04e6.

    Why this complements Shannon entropy: entropy is strongly file-type
    dependent and *saturates* near 8 bits/byte, so already-compressed office
    documents, JPEGs and archives are indistinguishable from ciphertext by
    entropy alone. Chi-square probes a different property of the *same* bytes
    (deviation of the value histogram from uniformity), keeps discriminative
    resolution in the region where entropy has flattened out, and costs one
    ``bincount``. The Data Aware Defense line of work applies chi-square to
    write buffers for exactly this reason, and we mirror that choice here.

    Returns:
        The chi-square statistic as a finite, non-negative float. Returns
        ``0.0`` for an empty sample. Short samples (fewer than
        :data:`BYTE_VALUES` bytes) are still scored, but the value is noisy and
        biased upward simply because most bins must be empty.
    """
    if sample is None:
        return 0.0
    buf = np.frombuffer(memoryview(sample).cast("B"), dtype=np.uint8)
    n = int(buf.size)
    if n == 0:
        return 0.0

    counts = np.bincount(buf, minlength=BYTE_VALUES).astype(np.float64)
    expected = n / float(BYTE_VALUES)
    stat = float(np.sum((counts - expected) ** 2) / expected)

    if not np.isfinite(stat) or stat < 0.0:
        return 0.0
    return stat

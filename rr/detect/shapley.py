"""Exact Shapley attribution by full coalition enumeration.

These values are **exact**, not sampled: every one of the ``2^n`` coalitions
is evaluated, so the result carries no Monte-Carlo error and is reproducible
bit-for-bit. That is affordable only for small ``n``; for a wider feature set
the function refuses rather than silently degrading to an approximation.
"""

from __future__ import annotations

from math import factorial
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd

__all__ = ["exact_shapley"]

BackgroundMode = Literal["mean", "reference"]

ScoreFn = Callable[[Any], Any]


def _extract_row(
    x: Any, feature_names: Sequence[str] | None
) -> tuple[np.ndarray, list[str] | None]:
    """Normalise ``x`` into ``(values, names)``."""
    if isinstance(x, pd.Series):
        names = list(x.index) if feature_names is None else list(feature_names)
        values = x.loc[names].to_numpy(dtype=float)
        return values, names
    if isinstance(x, pd.DataFrame):
        if len(x) != 1:
            raise ValueError("x must be a single row.")
        names = list(x.columns) if feature_names is None else list(feature_names)
        values = x.loc[:, names].to_numpy(dtype=float)[0]
        return values, names
    values = np.asarray(x, dtype=float).ravel()
    names = None if feature_names is None else list(feature_names)
    if names is not None and len(names) != values.size:
        raise ValueError("feature_names length does not match x.")
    return values, names


def _baseline_row(
    background: Any,
    mode: BackgroundMode,
    names: list[str] | None,
    n_features: int,
) -> np.ndarray:
    """Reduce ``background`` to the single row used to impute absent features."""
    if mode not in ("mean", "reference"):
        raise ValueError("background_mode must be 'mean' or 'reference'.")

    if isinstance(background, pd.DataFrame):
        frame = background if names is None else background.loc[:, names]
        arr = frame.to_numpy(dtype=float)
    elif isinstance(background, pd.Series):
        arr = (
            background.to_numpy(dtype=float)
            if names is None
            else background.loc[names].to_numpy(dtype=float)
        ).reshape(1, -1)
    else:
        arr = np.asarray(background, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(
            f"background must have {n_features} columns; got shape {arr.shape}."
        )
    if arr.shape[0] == 0:
        raise ValueError("background contains zero rows.")

    if mode == "mean":
        return arr.mean(axis=0)
    if arr.shape[0] != 1:
        raise ValueError(
            "background_mode='reference' requires exactly one background row; "
            f"got {arr.shape[0]}. Pass a single Series/1-row DataFrame, or use "
            "background_mode='mean'."
        )
    return arr[0]


def exact_shapley(
    score_fn: ScoreFn,
    x: Any,
    background: Any,
    *,
    max_exact: int = 15,
    background_mode: BackgroundMode = "mean",
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Exact Shapley values of ``x`` under ``score_fn`` relative to ``background``.

    Parameters
    ----------
    score_fn:
        Callable mapping a batch of rows to a 1-D array of scores. It is
        invoked **once** with all ``2^n`` coalition rows. If ``x`` carries
        feature names the batch is passed as a :class:`pandas.DataFrame` with
        those columns (so ``Detector.score`` can select by name); otherwise a
        2-D :class:`numpy.ndarray` is passed.
    x:
        The single window to explain (Series, 1-row DataFrame, or 1-D array).
    background:
        Reference distribution used to impute *absent* features.
    max_exact:
        Maximum number of features for which enumeration is attempted
        (default 15, i.e. 32768 coalition rows).
    background_mode:
        * ``"mean"`` (default) -- absent features take the **column mean** of
          ``background``. This is the standard "average benign window"
          reference. Caveat: the mean row may be off-manifold (no real window
          looks like it), which can push the model into an extrapolation
          regime.
        * ``"reference"`` -- absent features take the values of a **single
          reference row** (``background`` must have exactly one row). This
          keeps every coalition on a real, observed window and makes the
          attribution read as "why this window rather than *that* one", which
          is usually what an analyst wants. It is, however, sensitive to which
          reference row you pick.
    feature_names:
        Optional explicit ordering; defaults to the index/columns of ``x``.

    Returns
    -------
    numpy.ndarray
        Shapley value per feature, in the feature order of ``x``.

    Guarantees
    ----------
    These values are **exact**: all ``2^n`` coalitions are enumerated and
    weighted by ``|S|! (n-|S|-1)! / n!``. Nothing is sampled and no kernel
    regression is used, so efficiency holds to floating-point precision::

        sum(phi) == score_fn(x) - score_fn(background_row)

    Raises
    ------
    ValueError
        If ``n > max_exact``. The error names the alternative explicitly
        rather than silently switching to a sampled estimator: use a
        sampling-based estimator (permutation sampling or KernelSHAP), or
        reduce the feature set, or raise ``max_exact`` and accept the
        ``2^n`` cost.
    """
    values, names = _extract_row(x, feature_names)
    n = int(values.size)
    if n == 0:
        raise ValueError("x has no features.")
    if n > max_exact:
        raise ValueError(
            f"exact_shapley enumerates 2^n coalitions and n={n} exceeds "
            f"max_exact={max_exact} (that would be {2 ** n} evaluations). "
            "This function does not silently fall back to sampling. Either "
            "raise max_exact and accept the cost, reduce the feature set, or "
            "use a sampling-based estimator (permutation sampling / "
            "KernelSHAP) and report it as approximate."
        )

    base = _baseline_row(background, background_mode, names, n)

    n_coalitions = 1 << n
    masks = np.arange(n_coalitions, dtype=np.int64)
    bits = ((masks[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(bool)
    rows = np.where(bits, values[None, :], base[None, :])

    if names is not None:
        batch: Any = pd.DataFrame(rows, columns=names)
    else:
        batch = rows
    raw = np.asarray(score_fn(batch), dtype=float).ravel()
    if raw.size != n_coalitions:
        raise ValueError(
            f"score_fn returned {raw.size} scores for {n_coalitions} rows; it "
            "must be a batch-scoring function."
        )

    # Precompute Shapley weights by coalition size.
    n_fact = factorial(n)
    weight_by_size = np.array(
        [factorial(s) * factorial(n - s - 1) / n_fact for s in range(n)],
        dtype=float,
    )

    sizes = bits.sum(axis=1)
    phi = np.zeros(n, dtype=float)
    for i in range(n):
        without = ~bits[:, i]
        idx_without = masks[without]
        idx_with = idx_without | (1 << i)
        deltas = raw[idx_with] - raw[idx_without]
        phi[i] = float(np.sum(weight_by_size[sizes[without]] * deltas))
    return phi

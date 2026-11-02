"""Evaluation protocol: splits that do not leak, and honest uncertainty.

Why this module is opinionated
------------------------------
RansomRadar features are computed over sliding windows with a small hop
(1 second by default). Two adjacent windows therefore share almost all of their
underlying events and are *near-duplicates*. Under a random (shuffled) split,
the near-duplicate partner of nearly every test window ends up in training,
which produces the classic "AUC 1.000" artefact: the model is being asked to
recognise rows it has effectively already seen.

Arp et al. ("Dos and Don'ts of Machine Learning in Computer Security", USENIX
Security 2022) catalogue this as *sampling bias* / *spurious correlation* and
*data snooping*; Pendlebury et al. (TESSERACT, USENIX Security 2019) show that
ignoring temporal ordering inflates reported performance by a wide margin. The
mitigations implemented here are:

1. :func:`temporal_split` -- split strictly by ``window_start``, never shuffle,
   and purge test windows that overlap the training period in wall-clock time.
2. :func:`session_group_kfold` / :func:`leave_one_group_out` -- keep whole
   sessions (or whole ransomware families / benign workloads) together, so a
   session is never on both sides of a split.
3. :func:`assert_no_overlap` -- a loud, callable guardrail that refuses a split
   sharing ``session_id`` values or overlapping in time.
4. :func:`select_alpha_on_benign` -- choose the operating point using benign
   validation data only, never the test set (the data-snooping fix).
5. :func:`bootstrap_ci` -- resample at the *session* level, because windows
   within a session are strongly dependent and window-level resampling
   understates the confidence interval.

This module imports only numpy/pandas/scipy/scikit-learn and
:mod:`rr.eval.metrics`. It does not import ``rr.sim`` or ``rr.respond``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator, NamedTuple, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from rr.eval.metrics import alert_episodes

__all__ = [
    "AlphaSelection",
    "BootstrapCI",
    "DetectorProtocol",
    "LeakageError",
    "McNemarResult",
    "DEFAULT_ALPHA_GRID",
    "assert_no_overlap",
    "bootstrap_ci",
    "iter_fold_frames",
    "leave_one_group_out",
    "mcnemar",
    "select_alpha_on_benign",
    "session_group_kfold",
    "temporal_split",
]

#: Candidate alphas scanned by :func:`select_alpha_on_benign` (decades x {1,2,5}).
DEFAULT_ALPHA_GRID: tuple[float, ...] = tuple(
    float(m * d)
    for d in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
    for m in (1.0, 2.0, 5.0)
    if m * d <= 0.5
)


class LeakageError(ValueError):
    """Raised when a split (or a requested operation) would leak information.

    Deliberately a subclass of :class:`ValueError` so that callers who only
    guard against bad arguments still fail closed rather than silently
    publishing an optimistic number.
    """


@runtime_checkable
class DetectorProtocol(Protocol):
    """The duck-typed detector surface this package relies on.

    Any object exposing these methods can be evaluated, which keeps the
    evaluation layer decoupled from AGENT-B's concrete implementation.
    """

    def score(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - protocol
        ...

    def pvalue(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover - protocol
        ...

    def decide(
        self, pvalues: np.ndarray, alpha: float = 0.01, k: int = 3
    ) -> np.ndarray:  # pragma: no cover - protocol
        ...


class BootstrapCI(NamedTuple):
    """Point estimate with a percentile bootstrap confidence interval."""

    point: float
    low: float
    high: float
    level: float
    n_resamples: int
    grouped: bool


class McNemarResult(NamedTuple):
    """Result of a paired McNemar test between two detectors."""

    statistic: float
    pvalue: float
    n01: int
    n10: int
    method: str


class AlphaSelection(NamedTuple):
    """Operating point chosen on benign validation data only."""

    alpha: float
    achieved_fa_per_hour: float
    n_episodes: int
    benign_hours: float
    meets_target: bool
    table: pd.DataFrame


# --------------------------------------------------------------------------- #
# internal helpers
# --------------------------------------------------------------------------- #
def _require_columns(df: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    """Raise if ``df`` is missing any required column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{what} expects a pandas DataFrame, got {type(df)!r}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{what}: DataFrame is missing required column(s) {missing}; "
            f"available columns: {list(df.columns)}"
        )


def _to_seconds(values: Any) -> np.ndarray:
    """Convert a timestamp-like array to float epoch seconds.

    Accepts numeric epoch seconds, ``datetime64`` (naive or tz-aware) and
    ``timedelta64``. Returns ``float`` seconds so that all downstream
    comparisons are on one scale.
    """
    series = pd.Series(values)
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        series = series.dt.tz_convert("UTC").dt.tz_localize(None)
        return series.astype("int64").to_numpy() / 1e9
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64").to_numpy() / 1e9
    if pd.api.types.is_timedelta64_dtype(series):
        return series.astype("int64").to_numpy() / 1e9
    if series.dtype == object:
        converted = pd.to_datetime(series, errors="coerce", utc=True)
        if converted.notna().all():
            return converted.dt.tz_localize(None).astype("int64").to_numpy() / 1e9
    return pd.to_numeric(series, errors="raise").astype(float).to_numpy()


def _validate_groups(df: pd.DataFrame, group_col: str, what: str) -> np.ndarray:
    """Return group labels, refusing loudly when grouping is impossible."""
    _require_columns(df, [group_col], what)
    raw = df[group_col]
    if raw.isna().all():
        raise LeakageError(
            f"{what}: column {group_col!r} is entirely null, so grouping cannot be "
            "enforced. Refusing to produce a split that would silently mix a "
            "session across train and test (Arp et al., USENIX Security 2022)."
        )
    if raw.isna().any():
        n_null = int(raw.isna().sum())
        raise LeakageError(
            f"{what}: column {group_col!r} has {n_null} null value(s). Every row must "
            "carry a group label, otherwise unlabelled rows can leak across folds."
        )
    return raw.to_numpy()


def _consecutive_k(flags: np.ndarray, k: int) -> np.ndarray:
    """Fallback k-of-k persistence rule for detectors without ``decide``."""
    flags = np.asarray(flags, dtype=bool)
    if k <= 1:
        return flags.copy()
    out = np.zeros(flags.shape, dtype=bool)
    run = 0
    for i, value in enumerate(flags):
        run = run + 1 if value else 0
        out[i] = run >= k
    return out


def _decide(detector: Any, pvalues: np.ndarray, alpha: float, k: int) -> np.ndarray:
    """Call ``detector.decide`` when available, else apply a persistence rule."""
    decide = getattr(detector, "decide", None)
    if callable(decide):
        return np.asarray(decide(pvalues, alpha=alpha, k=k), dtype=bool)
    return _consecutive_k(np.asarray(pvalues, dtype=float) <= alpha, k)


# --------------------------------------------------------------------------- #
# T1: splits that do not leak
# --------------------------------------------------------------------------- #
def temporal_split(
    df: pd.DataFrame,
    frac: float = 0.7,
    *,
    time_col: str = "window_start",
    end_col: str = "window_end",
    gap_seconds: float = 0.0,
    purge_overlapping: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` strictly by time, never by shuffling.

    The first ``frac`` of windows (ordered by ``time_col``) become the training
    set; later windows become the test set. Rows sharing the boundary timestamp
    are all assigned to train so that one instant never straddles the split.

    With a 1-second hop and a multi-second window, consecutive windows overlap
    in wall-clock time and are near-duplicates. A random split would therefore
    place a test window's twin in training -- the exact leak that produced the
    meaningless AUC 1.000. When ``purge_overlapping`` is true (default), test
    windows whose ``time_col`` precedes the maximum training ``end_col`` are
    dropped, so no test window observes events already seen in training. This is
    the temporal-consistency requirement argued by Pendlebury et al. (TESSERACT,
    USENIX Security 2019) and the "temporal snooping" don't in Arp et al.
    (USENIX Security 2022).

    Args:
        df: Featurised windows, one row per window.
        frac: Fraction of rows (by count, in time order) assigned to train.
            Must be strictly between 0 and 1.
        time_col: Window start column. Numeric epoch seconds or datetimes.
        end_col: Window end column, used for purging. Ignored if absent.
        gap_seconds: Extra embargo added after the purge cutoff.
        purge_overlapping: Whether to purge time-overlapping test windows.

    Returns:
        ``(train_df, test_df)``, both time-ordered copies of subsets of ``df``.
        Guarantees ``train.time_col.max() < test.time_col.min()``.

    Raises:
        ValueError: If ``frac`` is out of range, the frame is too small, every
            row shares one timestamp, timestamps are null, or purging leaves the
            test set empty.
        KeyError: If ``time_col`` is absent.
    """
    _require_columns(df, [time_col], "temporal_split")
    if not (0.0 < float(frac) < 1.0):
        raise ValueError(f"temporal_split: frac must be in (0, 1), got {frac!r}")
    if float(gap_seconds) < 0.0:
        raise ValueError("temporal_split: gap_seconds must be >= 0")

    n_rows = len(df)
    if n_rows < 2:
        raise ValueError(
            f"temporal_split: need at least 2 rows to split, got {n_rows}"
        )

    starts = _to_seconds(df[time_col])
    if np.isnan(starts).any():
        raise ValueError(
            f"temporal_split: column {time_col!r} contains null/unparseable "
            "timestamps; a time-ordered split cannot be verified."
        )

    order = np.argsort(starts, kind="stable")
    ordered = df.iloc[order].copy()
    ordered_starts = starts[order]

    n_train = int(math.floor(n_rows * float(frac)))
    n_train = max(1, min(n_train, n_rows - 1))
    boundary = float(ordered_starts[n_train - 1])
    # Keep all rows sharing the boundary timestamp on the training side.
    n_train = int(np.searchsorted(ordered_starts, boundary, side="right"))
    if n_train >= n_rows:
        raise ValueError(
            "temporal_split: all rows share the same window_start value, so no "
            "strictly-later test set exists. Refusing to fall back to shuffling."
        )

    train = ordered.iloc[:n_train].copy()
    candidate = ordered.iloc[n_train:]

    cutoff = boundary
    if purge_overlapping and end_col in ordered.columns:
        train_ends = _to_seconds(train[end_col])
        if not np.isnan(train_ends).all():
            cutoff = max(cutoff, float(np.nanmax(train_ends)))
    cutoff += float(gap_seconds)

    candidate_starts = _to_seconds(candidate[time_col])
    keep = np.flatnonzero(candidate_starts >= cutoff)
    test = candidate.iloc[keep].copy()

    if test.empty:
        raise ValueError(
            "temporal_split: purging overlapping/embargoed windows emptied the "
            f"test set (cutoff={cutoff:g}). Lower frac, shorten the window, or "
            "reduce gap_seconds."
        )
    return train, test


def session_group_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    *,
    group_col: str = "session_id",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group k-fold over sessions: a session is never split across folds.

    Thin wrapper around :class:`sklearn.model_selection.GroupKFold` that refuses
    unsafe usage. Windows inside one session share the same underlying process
    behaviour, so splitting a session across train/test leaks (Arp et al.,
    USENIX Security 2022, "sampling bias" and "spurious correlations").

    Args:
        df: Featurised windows.
        n_splits: Number of folds. Must be >= 2 and <= number of unique groups.
        group_col: Session/group column.

    Returns:
        List of ``(train_idx, test_idx)`` pairs of **positional** indices into
        ``df`` (use ``df.iloc[...]``).

    Raises:
        LeakageError: If ``group_col`` is entirely or partially null.
        ValueError: If ``n_splits`` is invalid for the number of groups.
        KeyError: If ``group_col`` is absent.
    """
    groups = _validate_groups(df, group_col, "session_group_kfold")
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"session_group_kfold: n_splits must be >= 2, got {n_splits}")
    n_groups = int(pd.unique(groups).size)
    if n_groups < n_splits:
        raise ValueError(
            f"session_group_kfold: n_splits={n_splits} exceeds the number of unique "
            f"{group_col!r} values ({n_groups}). Collect more sessions instead of "
            "splitting one session across folds."
        )

    splitter = GroupKFold(n_splits=n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    positions = np.arange(len(df))
    for train_idx, test_idx in splitter.split(positions, groups=groups):
        folds.append(
            (np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int))
        )
    return folds


def leave_one_group_out(
    df: pd.DataFrame,
    group_col: str = "session_id",
) -> list[tuple[Any, np.ndarray, np.ndarray]]:
    """Leave-one-group-out splits, e.g. leave-one-family / one-workload out.

    Use ``group_col="family"`` for leave-one-ransomware-family-out generalisation
    and ``group_col="workload"`` for leave-one-benign-workload-out false-alarm
    stress testing. Holding out an entire family is the only way to claim
    detection of *unseen* behaviour rather than memorised behaviour.

    Args:
        df: Featurised windows.
        group_col: Column identifying the held-out group.

    Returns:
        List of ``(group_value, train_idx, test_idx)`` with positional indices.

    Raises:
        LeakageError: If ``group_col`` has null values.
        ValueError: If fewer than two distinct groups exist.
        KeyError: If ``group_col`` is absent.
    """
    groups = _validate_groups(df, group_col, "leave_one_group_out")
    unique = pd.unique(groups)
    if unique.size < 2:
        raise ValueError(
            f"leave_one_group_out: need >= 2 distinct {group_col!r} values, got "
            f"{unique.size}. A single group leaves no training data."
        )
    positions = np.arange(len(df))
    splits: list[tuple[Any, np.ndarray, np.ndarray]] = []
    for value in unique:
        mask = groups == value
        splits.append(
            (value, positions[~mask].astype(int), positions[mask].astype(int))
        )
    return splits


def iter_fold_frames(
    df: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Materialise ``(train_df, test_df)`` frames from positional fold indices.

    Args:
        df: The frame the folds were computed from.
        folds: ``(train_idx, test_idx)`` pairs.

    Yields:
        ``(train_df, test_df)`` copies for each fold.
    """
    for train_idx, test_idx in folds:
        yield df.iloc[np.asarray(train_idx, dtype=int)].copy(), df.iloc[
            np.asarray(test_idx, dtype=int)
        ].copy()


def assert_no_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    group_col: str = "session_id",
    time_col: str = "window_start",
    end_col: str = "window_end",
    check_groups: bool = True,
    check_time: bool = True,
) -> None:
    """Verify a split shares no session and (optionally) no wall-clock time.

    Call this immediately before fitting. It is cheap and it catches the two
    leaks that matter for RansomRadar:

    * the same ``session_id`` appearing on both sides (near-duplicate windows);
    * a test window whose time range overlaps the training period.

    For :func:`session_group_kfold` results, sessions from different folds are
    interleaved in time by construction, so pass ``check_time=False`` there and
    rely on the group check. For :func:`temporal_split` results, both checks
    should pass.

    Args:
        train_df: Training rows.
        test_df: Test rows.
        group_col: Session/group column to compare.
        time_col: Window start column.
        end_col: Window end column; falls back to ``time_col`` when absent.
        check_groups: Enforce disjoint group values.
        check_time: Enforce disjoint time intervals.

    Raises:
        LeakageError: On any detected overlap.
        KeyError: If a required column is missing for an enabled check.
    """
    if train_df.empty or test_df.empty:
        raise LeakageError(
            "assert_no_overlap: one side of the split is empty, which makes the "
            "resulting numbers meaningless."
        )

    if check_groups:
        _require_columns(train_df, [group_col], "assert_no_overlap")
        _require_columns(test_df, [group_col], "assert_no_overlap")
        train_groups = train_df[group_col]
        test_groups = test_df[group_col]
        if train_groups.isna().all() or test_groups.isna().all():
            raise LeakageError(
                f"assert_no_overlap: {group_col!r} is entirely null on one side, so "
                "group disjointness cannot be verified."
            )
        shared = set(train_groups.dropna().unique()) & set(
            test_groups.dropna().unique()
        )
        if shared:
            sample = sorted(map(str, shared))[:5]
            raise LeakageError(
                f"assert_no_overlap: {len(shared)} {group_col!r} value(s) appear in "
                f"both train and test (e.g. {sample}). Windows from one session are "
                "near-duplicates; this split leaks."
            )

    if check_time:
        _require_columns(train_df, [time_col], "assert_no_overlap")
        _require_columns(test_df, [time_col], "assert_no_overlap")
        train_start = _to_seconds(train_df[time_col])
        test_start = _to_seconds(test_df[time_col])
        train_end = (
            _to_seconds(train_df[end_col])
            if end_col in train_df.columns
            else train_start
        )
        test_end = (
            _to_seconds(test_df[end_col]) if end_col in test_df.columns else test_start
        )
        tr_lo, tr_hi = float(np.nanmin(train_start)), float(np.nanmax(train_end))
        te_lo, te_hi = float(np.nanmin(test_start)), float(np.nanmax(test_end))
        if tr_hi > te_lo and te_hi > tr_lo:
            raise LeakageError(
                "assert_no_overlap: train interval "
                f"[{tr_lo:g}, {tr_hi:g}] overlaps test interval "
                f"[{te_lo:g}, {te_hi:g}]. Overlapping windows are near-duplicates; "
                "use temporal_split(..., purge_overlapping=True)."
            )


# --------------------------------------------------------------------------- #
# T2: benign-only threshold selection
# --------------------------------------------------------------------------- #
def select_alpha_on_benign(
    detector: Any,
    X_benign_val: pd.DataFrame,
    target_fa_per_hour: float,
    window_hop_seconds: float,
    *,
    k: int = 3,
    candidate_alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
    episode_gap_seconds: float | None = None,
    timestamps: Any | None = None,
) -> AlphaSelection:
    """Choose the operating point using benign validation data only.

    This is the data-snooping fix. The threshold must never be tuned on the test
    set or on any attack data: doing so silently optimises the reported numbers
    (Arp et al., USENIX Security 2022, "data snooping"). Here we scan candidate
    alphas on held-out *benign* windows, convert firing windows into alert
    episodes (a 1-second hop would otherwise count one sustained alarm hundreds
    of times), and pick the **most sensitive** alpha whose benign false-alarm
    rate still satisfies the operator's budget.

    Args:
        detector: Any object exposing ``pvalue(X)``; ``decide(pvalues, alpha, k)``
            is used when present, otherwise a k-consecutive-window rule is applied.
        X_benign_val: Benign validation windows. Must not include attack data.
        target_fa_per_hour: Operator budget, alarms per benign hour. Must be >= 0.
        window_hop_seconds: Hop between consecutive windows, in seconds.
        k: Persistence parameter forwarded to ``decide``.
        candidate_alphas: Alphas to scan, ascending order not required.
        episode_gap_seconds: Gap that separates two alert episodes. Defaults to
            ``window_hop_seconds``, i.e. contiguous firing windows collapse.
        timestamps: Optional per-window timestamps for episode grouping. When
            omitted, window index positions scaled by the hop are used.

    Returns:
        :class:`AlphaSelection` with the chosen alpha, the achieved benign
        false-alarm rate, and the full scan table for the report appendix.

    Raises:
        ValueError: If inputs are empty/invalid or no candidate alphas are given.
        AttributeError: If ``detector`` has no ``pvalue`` method.
    """
    if not isinstance(X_benign_val, pd.DataFrame):
        raise TypeError(
            "select_alpha_on_benign: X_benign_val must be a pandas DataFrame"
        )
    n_windows = len(X_benign_val)
    if n_windows == 0:
        raise ValueError("select_alpha_on_benign: X_benign_val is empty")
    hop = float(window_hop_seconds)
    if hop <= 0:
        raise ValueError(
            f"select_alpha_on_benign: window_hop_seconds must be > 0, got {hop!r}"
        )
    target = float(target_fa_per_hour)
    if target < 0 or not math.isfinite(target):
        raise ValueError(
            "select_alpha_on_benign: target_fa_per_hour must be a finite value >= 0"
        )
    alphas = sorted({float(a) for a in candidate_alphas})
    if not alphas:
        raise ValueError("select_alpha_on_benign: candidate_alphas is empty")
    if any(not (0.0 < a <= 1.0) for a in alphas):
        raise ValueError("select_alpha_on_benign: alphas must lie in (0, 1]")

    if not hasattr(detector, "pvalue"):
        raise AttributeError(
            "select_alpha_on_benign: detector must expose pvalue(X); got "
            f"{type(detector)!r}"
        )

    pvalues = np.asarray(detector.pvalue(X_benign_val), dtype=float)
    if pvalues.shape[0] != n_windows:
        raise ValueError(
            "select_alpha_on_benign: detector.pvalue returned "
            f"{pvalues.shape[0]} values for {n_windows} windows"
        )

    if timestamps is None:
        ts = np.arange(n_windows, dtype=float) * hop
    else:
        ts = _to_seconds(timestamps)
        if ts.shape[0] != n_windows:
            raise ValueError(
                "select_alpha_on_benign: timestamps length does not match X_benign_val"
            )
    gap = float(episode_gap_seconds) if episode_gap_seconds is not None else hop
    benign_hours = (n_windows * hop) / 3600.0

    records: list[dict[str, float]] = []
    for alpha in alphas:
        flags = _decide(detector, pvalues, alpha=alpha, k=k)
        episodes = alert_episodes(flags, timestamps=ts, gap_seconds=gap)
        rate = len(episodes) / benign_hours if benign_hours > 0 else float("inf")
        records.append(
            {
                "alpha": alpha,
                "n_firing_windows": float(np.count_nonzero(flags)),
                "n_episodes": float(len(episodes)),
                "benign_hours": benign_hours,
                "fa_per_hour": float(rate),
                "meets_target": float(rate <= target),
            }
        )

    table = pd.DataFrame.from_records(records).sort_values("alpha").reset_index(
        drop=True
    )
    feasible = table[table["meets_target"] > 0.5]
    if not feasible.empty:
        # Most sensitive (largest) alpha that still respects the budget.
        chosen = feasible.iloc[int(feasible["alpha"].to_numpy().argmax())]
        meets = True
    else:
        # Nothing meets the budget: report the least noisy option and say so.
        chosen = table.iloc[int(table["fa_per_hour"].to_numpy().argmin())]
        meets = False

    return AlphaSelection(
        alpha=float(chosen["alpha"]),
        achieved_fa_per_hour=float(chosen["fa_per_hour"]),
        n_episodes=int(chosen["n_episodes"]),
        benign_hours=float(benign_hours),
        meets_target=bool(meets),
        table=table,
    )


# --------------------------------------------------------------------------- #
# T4: uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Any,
    stat: Callable[[np.ndarray], float] = np.mean,
    n: int = 1000,
    seed: int = 0,
    *,
    groups: Any | None = None,
    level: float = 0.95,
) -> BootstrapCI:
    """Percentile bootstrap CI, resampling at the session level when possible.

    Windows inside a session are highly dependent (1-second hop, overlapping
    windows). Resampling individual windows therefore treats correlated
    observations as independent and produces intervals that are far too narrow.
    When ``groups`` is supplied, whole groups are resampled with replacement and
    their values pooled, which is the correct unit of independence here.

    Args:
        values: 1-D array of per-observation metric values.
        stat: Statistic to apply to a resampled pool. Defaults to the mean.
        n: Number of bootstrap resamples. Must be >= 1.
        seed: Seed for the deterministic RNG.
        groups: Optional per-value group labels (typically ``session_id``).
        level: Confidence level, e.g. 0.95.

    Returns:
        :class:`BootstrapCI` with the point estimate and percentile interval.

    Raises:
        ValueError: If ``values`` is empty, ``groups`` length mismatches,
            ``n < 1``, or ``level`` is out of range.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("bootstrap_ci: values is empty")
    n = int(n)
    if n < 1:
        raise ValueError(f"bootstrap_ci: n must be >= 1, got {n}")
    if not (0.0 < float(level) < 1.0):
        raise ValueError(f"bootstrap_ci: level must be in (0, 1), got {level!r}")

    rng = np.random.default_rng(int(seed))
    point = float(stat(arr))
    grouped = groups is not None

    if grouped:
        group_arr = np.asarray(groups).ravel()
        if group_arr.shape[0] != arr.shape[0]:
            raise ValueError(
                "bootstrap_ci: groups length "
                f"({group_arr.shape[0]}) must match values length ({arr.shape[0]})"
            )
        unique_groups = pd.unique(group_arr)
        index_by_group = [
            np.flatnonzero(group_arr == g).astype(int) for g in unique_groups
        ]
        n_groups = len(index_by_group)
        draws = np.empty(n, dtype=float)
        for i in range(n):
            picks = rng.integers(0, n_groups, size=n_groups)
            pooled = np.concatenate([index_by_group[p] for p in picks])
            draws[i] = float(stat(arr[pooled]))
    else:
        draws = np.empty(n, dtype=float)
        size = arr.shape[0]
        for i in range(n):
            draws[i] = float(stat(arr[rng.integers(0, size, size=size)]))

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return BootstrapCI(point, float("nan"), float("nan"), float(level), n, grouped)
    tail = (1.0 - float(level)) / 2.0 * 100.0
    low = float(np.percentile(finite, tail))
    high = float(np.percentile(finite, 100.0 - tail))
    return BootstrapCI(point, low, high, float(level), n, grouped)


def mcnemar(
    y_true: Any,
    pred_a: Any,
    pred_b: Any,
    *,
    exact_threshold: int = 25,
    correction: bool = True,
) -> McNemarResult:
    """Paired McNemar test comparing two detectors on the same windows.

    Two detectors evaluated on identical data produce *paired* predictions, so an
    unpaired comparison of accuracies overstates uncertainty. McNemar's test
    looks only at the discordant pairs: cases where exactly one detector is
    correct.

    Args:
        y_true: Ground-truth binary labels.
        pred_a: Detector A binary predictions.
        pred_b: Detector B binary predictions.
        exact_threshold: Use an exact binomial test when the number of
            discordant pairs is at or below this value.
        correction: Apply Yates' continuity correction in the chi-square branch.

    Returns:
        :class:`McNemarResult` with statistic, p-value, discordant counts
        (``n01`` = only B correct, ``n10`` = only A correct) and method used.

    Raises:
        ValueError: If the three arrays differ in length or are empty.
    """
    truth = np.asarray(y_true).astype(bool).ravel()
    a = np.asarray(pred_a).astype(bool).ravel()
    b = np.asarray(pred_b).astype(bool).ravel()
    if not (truth.shape == a.shape == b.shape):
        raise ValueError(
            "mcnemar: y_true, pred_a and pred_b must have identical shapes; got "
            f"{truth.shape}, {a.shape}, {b.shape}"
        )
    if truth.size == 0:
        raise ValueError("mcnemar: inputs are empty")

    a_correct = a == truth
    b_correct = b == truth
    n10 = int(np.count_nonzero(a_correct & ~b_correct))
    n01 = int(np.count_nonzero(~a_correct & b_correct))
    total = n01 + n10

    if total == 0:
        return McNemarResult(0.0, 1.0, n01, n10, "no-discordant-pairs")
    if total <= int(exact_threshold):
        result = stats.binomtest(n10, n=total, p=0.5, alternative="two-sided")
        return McNemarResult(float(n10), float(result.pvalue), n01, n10, "exact-binomial")

    diff = abs(n10 - n01)
    if correction:
        diff = max(0.0, diff - 1.0)
    statistic = float(diff**2) / float(total)
    pvalue = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(
        statistic, pvalue, n01, n10, "chi2-with-correction" if correction else "chi2"
    )

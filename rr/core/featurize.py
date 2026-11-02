"""Feature engineering over sliding windows.

Design notes
------------

**Rate-invariant features.** The original ten features were dominated by raw
per-window counts and rates. Those are precisely what a throttled
"low-and-slow" adversary defeats: halve the request rate and every count halves,
while the *shape* of the behaviour is unchanged. We therefore keep the ten
legacy features byte-for-byte (same names, same meaning, for backwards
comparability and ablation studies) and add ratio features that are invariant to
a global slow-down, timing-regularity features, file-type funnelling features,
and cheap forensic aggregates.

**Statefulness and overlapping windows.** EWMA, cumulative and delta features
are computed over the *window sequence* (ordered by ``window_start``, separately
per scale), not over events. Two reasons: (1) with a 1-second hop the window
sequence is a uniformly sampled signal, so an EWMA over it has a well-defined
time constant, whereas an event-indexed EWMA would silently speed up whenever
the event rate rises - re-introducing exactly the rate dependence the ratio
features exist to remove; (2) each scale is its own sequence, so 10 s and 60 s
rows remain independent and the ``scale`` column stays a clean grouping key.

Because that state lives in the sequence, :func:`featurize_window` is a pure
per-window function: it fills the stateful columns with their *stateless
degenerate* values (``*_ewma`` = current value, ``*_cum`` = current value,
``entropy_delta`` = 0.0). :func:`featurize` overwrites them in a single ordered
pass. Callers who need true EWMA semantics must use :func:`featurize`.

**Finiteness.** Every returned value is a finite float. Ratios use
``max(1.0, denominator)`` guards, aggregates over empty collections return 0.0,
and a final clamp maps NaN/inf to 0.0 and bounds magnitudes.

**Consistency invariants.** ``extension_changes <= renames`` and
``files_touched >= created`` are enforced here, not merely hoped for. Any
counterfactual / evasion search optimises in feature space; without hard
invariants it can propose feature vectors that correspond to no physically
realisable event stream, producing meaningless "evasions".

No I/O and no side effects at import time.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from rr.core.schema import NormalisedEvent, Window
from rr.core.windows import sliding_windows

__all__ = [
    "CANARY_MARKERS",
    "EWMA_ALPHA",
    "FEATURE_ORDER",
    "META_COLUMNS",
    "featurize",
    "featurize_window",
    "is_canary_path",
]

#: Smoothing factor for all EWMA features. With a 1 s hop this is a time
#: constant of roughly 1/alpha = 5 windows (~5 s at the 10 s scale).
EWMA_ALPHA: float = 0.2

#: Substrings that mark a decoy ("canary") file planted by the responder.
CANARY_MARKERS: tuple[str, ...] = (
    "__rr_canary__",
    ".rr_canary",
    "/.canary/",
    "\\.canary\\",
    "_canary_",
    "/canaries/",
    "\\canaries\\",
)

#: Absolute bound applied to every feature value after computation.
_CLAMP_ABS: float = 1e12

_EPS: float = 1e-9

#: Legacy ten features, unchanged in name and meaning.
_LEGACY_FEATURES: tuple[str, ...] = (
    "event_rate",
    "files_touched",
    "created",
    "existing_writes",
    "renames",
    "extension_changes",
    "deletions",
    "canary_touched",
    "entropy_max",
    "type_divergence",
)

#: Ratio features: invariant to a uniform slow-down of the whole attack.
_RATIO_FEATURES: tuple[str, ...] = (
    "rename_write_ratio",
    "extchange_rename_ratio",
    "delete_create_ratio",
)

#: Forensic aggregates over sampled bytes.
_FORENSIC_FEATURES: tuple[str, ...] = (
    "entropy_mean",
    "entropy_delta",
    "chi2_max",
    "chi2_mean",
)

#: Inter-arrival timing: scripted crypto loops have suspiciously regular gaps.
_TIMING_FEATURES: tuple[str, ...] = (
    "inter_event_mean",
    "inter_event_cv",
)

#: File-type funnelling (CryptoDrop) and ransom-note pressure.
_FUNNEL_FEATURES: tuple[str, ...] = (
    "distinct_ext_written",
    "ext_funnel_ratio",
    "note_like_rate",
)

#: base feature -> EWMA column name (computed over the window sequence).
_EWMA_SOURCES: tuple[tuple[str, str], ...] = (
    ("event_rate", "event_rate_ewma"),
    ("files_touched", "files_touched_ewma"),
    ("created", "created_ewma"),
    ("existing_writes", "existing_writes_ewma"),
    ("renames", "renames_ewma"),
    ("extension_changes", "extension_changes_ewma"),
    ("deletions", "deletions_ewma"),
    ("entropy_max", "entropy_max_ewma"),
)

#: base feature -> cumulative column name (running sum over the sequence).
_CUM_SOURCES: tuple[tuple[str, str], ...] = (
    ("files_touched", "files_touched_cum"),
    ("created", "created_cum"),
    ("existing_writes", "existing_writes_cum"),
    ("renames", "renames_cum"),
    ("extension_changes", "extension_changes_cum"),
    ("deletions", "deletions_cum"),
    ("canary_touched", "canary_touched_cum"),
)

#: Canonical feature order. Downstream detectors must consume this list, never a
#: hard-coded copy. ``featurize`` guarantees columns == FEATURE_ORDER + META_COLUMNS.
FEATURE_ORDER: list[str] = list(
    _LEGACY_FEATURES
    + _RATIO_FEATURES
    + _FORENSIC_FEATURES
    + _TIMING_FEATURES
    + _FUNNEL_FEATURES
    + tuple(name for _, name in _EWMA_SOURCES)
    + tuple(name for _, name in _CUM_SOURCES)
)

#: Non-feature columns appended to the feature frame, in order.
META_COLUMNS: list[str] = ["window_start", "window_end", "scale", "session_id"]

_WRITE_KINDS = frozenset({"created", "modified", "moved"})


def is_canary_path(path: str | None) -> bool:
    """Return True when ``path`` looks like a planted decoy (canary) file.

    Matching is substring-based and case-insensitive so it works for both
    POSIX and Windows path separators without any platform-specific code.
    """
    if not path:
        return False
    lowered = str(path).lower().replace("\\", "/")
    for marker in CANARY_MARKERS:
        if marker.replace("\\", "/") in lowered:
            return True
    return False


def _f(value: float) -> float:
    """Clamp ``value`` to a finite float in ``[-_CLAMP_ABS, _CLAMP_ABS]``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    if out > _CLAMP_ABS:
        return _CLAMP_ABS
    if out < -_CLAMP_ABS:
        return -_CLAMP_ABS
    return out


def _ratio(numerator: float, denominator: float) -> float:
    """Ratio with a ``max(1, denominator)`` guard, so zero denominators are safe."""
    return _f(numerator / max(1.0, float(denominator)))


def featurize_window(w: Window) -> dict[str, float]:
    """Compute the feature dictionary for a single window.

    The returned mapping has exactly the keys of :data:`FEATURE_ORDER` and all
    values are finite floats. Sequence-stateful keys (``*_ewma``, ``*_cum``,
    ``entropy_delta``) receive their degenerate single-window values; use
    :func:`featurize` to obtain their true sequence-aware values.

    Enforced invariants:

    * ``extension_changes <= renames``
    * ``files_touched >= created``
    """
    events: Sequence[NormalisedEvent] = w.events
    duration = max(float(w.end) - float(w.start), _EPS)

    created = 0
    existing_writes = 0
    renames = 0
    extension_changes = 0
    deletions = 0
    divergent = 0
    note_like = 0

    paths: set[str] = set()
    canaries: set[str] = set()
    written_exts: Counter[str] = Counter()
    entropies: list[float] = []
    chi2s: list[float] = []
    timestamps: list[float] = []

    for event in events:
        timestamps.append(float(event.ts))

        if event.path:
            paths.add(event.path)
        if event.dest_path:
            paths.add(event.dest_path)

        kind = event.kind
        if kind == "created":
            created += 1
        elif kind == "modified":
            existing_writes += 1
        elif kind == "deleted":
            deletions += 1
        elif kind == "moved":
            renames += 1
            src_ext = event.ext or ""
            dst_ext = event.dest_ext if event.dest_ext is not None else ""
            if dst_ext != src_ext:
                extension_changes += 1

        if event.divergent:
            divergent += 1
        if event.note_like:
            note_like += 1

        if is_canary_path(event.path):
            canaries.add(event.path)
        if is_canary_path(event.dest_path):
            canaries.add(str(event.dest_path))

        if event.entropy is not None:
            entropies.append(_f(event.entropy))
        if event.chi2 is not None:
            chi2s.append(_f(event.chi2))

        if kind in _WRITE_KINDS:
            target_ext = (
                event.dest_ext
                if (kind == "moved" and event.dest_ext is not None)
                else event.ext
            )
            if target_ext:
                written_exts[target_ext] += 1

    # --- invariants -------------------------------------------------------
    extension_changes = min(extension_changes, renames)
    files_touched = max(len(paths), created)

    # --- timing regularity ------------------------------------------------
    if len(timestamps) >= 2:
        gaps = np.diff(np.asarray(sorted(timestamps), dtype=np.float64))
        gap_mean = float(gaps.mean())
        gap_std = float(gaps.std())
        inter_event_mean = _f(gap_mean)
        inter_event_cv = (
            _ratio(gap_std, max(gap_mean, _EPS)) if gap_mean > _EPS else 0.0
        )
    else:
        # Fewer than two events: no gap is observable. 0.0 is the documented
        # sentinel for "undefined", shared with the empty-window case.
        inter_event_mean = 0.0
        inter_event_cv = 0.0

    # --- funnelling -------------------------------------------------------
    total_written = float(sum(written_exts.values()))
    if total_written > 0.0:
        ext_funnel_ratio = _f(max(written_exts.values()) / total_written)
    else:
        ext_funnel_ratio = 0.0

    entropy_max = _f(max(entropies)) if entropies else 0.0
    entropy_mean = _f(sum(entropies) / len(entropies)) if entropies else 0.0
    chi2_max = _f(max(chi2s)) if chi2s else 0.0
    chi2_mean = _f(sum(chi2s) / len(chi2s)) if chi2s else 0.0

    features: dict[str, float] = {
        # legacy ten
        "event_rate": _f(len(events) / duration),
        "files_touched": _f(files_touched),
        "created": _f(created),
        "existing_writes": _f(existing_writes),
        "renames": _f(renames),
        "extension_changes": _f(extension_changes),
        "deletions": _f(deletions),
        "canary_touched": _f(len(canaries)),
        "entropy_max": entropy_max,
        "type_divergence": _f(divergent),
        # rate-invariant ratios
        "rename_write_ratio": _ratio(renames, existing_writes + created),
        "extchange_rename_ratio": _ratio(extension_changes, renames),
        "delete_create_ratio": _ratio(deletions, created),
        # forensic
        "entropy_mean": entropy_mean,
        "entropy_delta": 0.0,  # filled by featurize() over the window sequence
        "chi2_max": chi2_max,
        "chi2_mean": chi2_mean,
        # timing
        "inter_event_mean": inter_event_mean,
        "inter_event_cv": inter_event_cv,
        # funnelling
        "distinct_ext_written": _f(len(written_exts)),
        "ext_funnel_ratio": ext_funnel_ratio,
        "note_like_rate": _f(note_like / duration),
    }

    # Degenerate stateless values for the sequence-stateful columns.
    for base, ewma_name in _EWMA_SOURCES:
        features[ewma_name] = features[base]
    for base, cum_name in _CUM_SOURCES:
        features[cum_name] = features[base]

    return {name: _f(features[name]) for name in FEATURE_ORDER}


class _SequenceState:
    """Running EWMA / cumulative state for one scale sequence.

    Windows overlap, so this state is advanced once per *window*, in
    ``window_start`` order, which is what makes the EWMA time constant
    interpretable as ``1 / EWMA_ALPHA`` hops.
    """

    def __init__(self, alpha: float = EWMA_ALPHA) -> None:
        """Initialise with smoothing factor ``alpha`` in ``(0, 1]``."""
        self.alpha = float(alpha)
        self._ewma: dict[str, float] = {}
        self._cum: dict[str, float] = {}

    def apply(self, features: dict[str, float]) -> None:
        """Overwrite the stateful entries of ``features`` in place."""
        for base, ewma_name in _EWMA_SOURCES:
            current = _f(features[base])
            previous = self._ewma.get(base)
            if previous is None:
                updated = current
                if base == "entropy_max":
                    features["entropy_delta"] = 0.0
            else:
                if base == "entropy_max":
                    # Delta is measured against the EWMA *before* this update,
                    # so it answers "how unusual is this window vs its recent past".
                    features["entropy_delta"] = _f(current - previous)
                updated = _f(self.alpha * current + (1.0 - self.alpha) * previous)
            self._ewma[base] = updated
            features[ewma_name] = updated

        for base, cum_name in _CUM_SOURCES:
            total = _f(self._cum.get(base, 0.0) + _f(features[base]))
            self._cum[base] = total
            features[cum_name] = total


def _dominant_session(events: Sequence[NormalisedEvent]) -> str | None:
    """Return the most frequent non-``None`` ``session_id``, or ``None``."""
    counts = Counter(e.session_id for e in events if e.session_id is not None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _window_session(w: Window, fallback: str | None) -> str | None:
    """Return the dominant ``session_id`` inside ``w``, else ``fallback``.

    Idle-filled windows contain no events, so they inherit the stream-level
    session; otherwise they could not be grouped for session-wise splits.
    """
    session = _dominant_session(w.events)
    return session if session is not None else fallback


def _empty_frame() -> pd.DataFrame:
    """Return a correctly typed, zero-row feature frame."""
    data: dict[str, pd.Series] = {}
    for column in FEATURE_ORDER + ["window_start", "window_end", "scale"]:
        data[column] = pd.Series(dtype="float64")
    data["session_id"] = pd.Series(dtype="object")
    return pd.DataFrame(data, columns=FEATURE_ORDER + META_COLUMNS)


def featurize(
    events: Iterable[NormalisedEvent],
    scales: Sequence[float] = (10.0, 60.0),
    hop: float = 1.0,
) -> pd.DataFrame:
    """Featurise an event stream at multiple window scales.

    :func:`rr.core.windows.sliding_windows` is run once per scale with
    ``fill_idle=True``, each window is featurised, sequence-stateful columns are
    resolved per scale, and the results are concatenated with a ``scale`` tag.
    A 10 s scale catches short encryption bursts; a 60 s scale accumulates
    throttled "low-and-slow" activity that never looks dense in any 10 s slice.

    Returns:
        A DataFrame whose columns are exactly
        ``FEATURE_ORDER + ["window_start", "window_end", "scale", "session_id"]``.
    """
    ordered = tuple(events)
    scale_list = [float(s) for s in scales]
    if not ordered or not scale_list:
        return _empty_frame()

    fallback_session = _dominant_session(ordered)
    rows: list[dict[str, object]] = []

    for scale in scale_list:
        state = _SequenceState()
        for window in sliding_windows(
            ordered, size=scale, hop=float(hop), fill_idle=True
        ):
            features = featurize_window(window)
            state.apply(features)
            row: dict[str, object] = {
                name: _f(features[name]) for name in FEATURE_ORDER
            }
            row["window_start"] = float(window.start)
            row["window_end"] = float(window.end)
            row["scale"] = float(scale)
            row["session_id"] = _window_session(window, fallback_session)
            rows.append(row)

    if not rows:
        return _empty_frame()

    frame = pd.DataFrame(rows, columns=FEATURE_ORDER + META_COLUMNS)
    for column in FEATURE_ORDER + ["window_start", "window_end", "scale"]:
        frame[column] = frame[column].astype("float64")
    frame["session_id"] = frame["session_id"].astype("object")
    return frame.reset_index(drop=True)

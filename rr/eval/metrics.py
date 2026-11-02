"""rr.eval.metrics — the numbers a ransomware victim actually cares about.

Accuracy is the wrong question for ransomware: most windows are benign, so a
detector that always says "normal" scores brilliantly and protects nobody.
These metrics measure the real quantities instead:

* how many files were lost before the alarm raised,
* how long detection took,
* how often the detector cries wolf on benign traffic,
* and a precision-recall summary that survives heavy class imbalance.

The attack side consumes an ``AttackTrace.to_dataframe()`` frame with columns
``ts, op, path, dest_path, bytes_changed`` (see ``rr.sim.evader``).

Deliberately absent: ``point_adjusted_f1``. It lives here as a function that
raises, because a random scorer beats published state-of-the-art under it
(Kim et al., AAAI 2022) — reporting it would flatter us and mean nothing.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

__all__ = [
    "files_lost_before_detection",
    "time_to_detect",
    "false_alarms_per_hour",
    "alert_episodes",
    "pr_auc",
    "point_adjusted_f1_DO_NOT_USE",
]

_DAMAGING_OPS = ("encrypt",)


def alert_episodes(flags: Sequence[bool] | np.ndarray,
                   *,
                   timestamps: Sequence[float] | np.ndarray | None = None,
                   gap_seconds: float = 1.0) -> list[dict[str, float]]:
    """Collapse per-window alert flags into distinct alarm episodes.

    k-of-n sequential rules still fire on many consecutive windows; an
    operator experiences one *incident*, not thirty alerts. Firings whose
    timestamps are within ``gap_seconds`` of the previous firing belong to the
    same episode. Returns one dict per episode:
    ``{start, end, n_firing, start_idx, end_idx}`` — ``rr.eval.protocol``
    counts episodes to express false alarms in per-hour units.
    """
    flags_arr = np.asarray(flags, dtype=bool)
    ts = (np.arange(flags_arr.size, dtype=float) if timestamps is None
          else np.asarray(timestamps, dtype=float))
    if ts.shape[0] != flags_arr.size:
        raise ValueError("timestamps length does not match flags length.")
    gap = float(gap_seconds)

    episodes: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for i, firing in enumerate(flags_arr):
        if not firing:
            continue
        if current is not None and ts[i] - current["end"] <= gap:
            current["end"] = float(ts[i])
            current["n_firing"] += 1
            current["end_idx"] = float(i)
        else:
            if current is not None:
                episodes.append(current)
            current = {"start": float(ts[i]), "end": float(ts[i]),
                       "n_firing": 1, "start_idx": float(i), "end_idx": float(i)}
    if current is not None:
        episodes.append(current)
    return episodes


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"trace frame is missing required columns: {missing}")


def files_lost_before_detection(trace: pd.DataFrame, detection_ts: float) -> int:
    """Count distinct files damaged *before* the first alert.

    A file counts as lost when it was damaged (encrypted) at or before
    ``detection_ts``. Renames and notes do not destroy data; encryption does.
    """
    _require_columns(trace, ["ts", "op", "path"])
    damaged = trace[(trace["op"].isin(_DAMAGING_OPS)) & (trace["ts"] <= detection_ts)]
    return int(damaged["path"].nunique())


def time_to_detect(attack_start_ts: float, detection_ts: float) -> float:
    """Seconds from the first attack operation to the first alert."""
    if detection_ts < attack_start_ts:
        raise ValueError("detection_ts precedes attack_start_ts.")
    return float(detection_ts - attack_start_ts)


def false_alarms_per_hour(alert_ts: Sequence[float], benign_hours: float) -> float:
    """Alert rate on benign traffic, in units an operator can feel.

    ``alert_ts`` holds alert timestamps raised during a purely benign period
    of ``benign_hours`` wall-clock hours. An alert rate is only comparable
    across deployments in this unit — "3 per hour" is actionable, "0.02 AUROC"
    is not.
    """
    if benign_hours <= 0:
        raise ValueError("benign_hours must be > 0.")
    return float(len(list(alert_ts)) / benign_hours)


def pr_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Precision-recall AUC (average precision) — robust to the benign-majority
    class imbalance that makes ROC-AUC and accuracy flattering here."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or y.size != s.size:
        raise ValueError("y_true and scores must be non-empty and equal length.")
    if y.sum() == 0:
        raise ValueError("no positive labels present; PR-AUC is undefined.")
    return float(average_precision_score(y, s))


def point_adjusted_f1_DO_NOT_USE(*args: object, **kwargs: object) -> float:
    """Forbidden metric — always raises.

    Point-adjusted evaluation (used by several time-series anomaly-detection
    papers) credits a detector for an entire anomalous *segment* if any single
    point inside it is flagged. Kim et al. (AAAI 2022, "Towards a Rigorous
    Evaluation of Time-series Anomaly Detection") showed a RANDOM scorer beats
    published state-of-the-art under it. Reporting it here would produce a
    flattering number that predicts nothing about files lost or false alarms.
    We keep the name so the refusal is greppable; we do not keep the number.
    """
    raise RuntimeError(
        "point_adjusted_f1 is deliberately not implemented: a random scorer "
        "beats state-of-the-art results under point-adjusted evaluation (Kim "
        "et al., AAAI 2022). Report files_lost_before_detection, "
        "false_alarms_per_hour and pr_auc instead."
    )


if __name__ == "__main__":  # pragma: no cover - smoke demo
    rng = np.random.default_rng(0)
    base = 1_000_000.0
    trace = pd.DataFrame({
        "ts": base + np.arange(12) * 0.5,
        "op": ["encrypt"] * 10 + ["rename", "note"],
        "path": [f"docs/f{i}.txt" for i in range(10)] + ["docs/f0.txt", "_note.txt"],
        "dest_path": [None] * 10 + ["docs/f0.txt.locked", None],
        "bytes_changed": [512] * 12,
    })
    lost = files_lost_before_detection(trace, base + 4.0)
    print(f"files lost by t+4s: {lost} (expected 9)")
    print(f"time to detect at t+6s: {time_to_detect(base, base + 6.0):.1f}s")
    print(f"false alarms/hour: {false_alarms_per_hour([1.0, 2.5], 0.5):.1f}")
    y = [0, 0, 0, 0, 1, 1]
    s = [0.1, 0.2, 0.15, 0.4, 0.7, 0.9]
    print(f"pr_auc: {pr_auc(y, s):.3f}")
    try:
        point_adjusted_f1_DO_NOT_USE()
    except RuntimeError as exc:
        print(f"point_adjusted_f1 refused: {str(exc)[:60]}…")

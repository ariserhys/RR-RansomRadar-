"""Tests for the modules this project added to satisfy the contract:
rr.eval.metrics, rr.eval.figures, rr.eval.ablation, rr.compat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------- #
# rr.eval.metrics
# --------------------------------------------------------------------- #
def _trace(n_damaged: int = 6) -> pd.DataFrame:
    base = 1_000_000.0
    rows = []
    for i in range(n_damaged):
        rows.append({"ts": base + i * 0.5, "op": "encrypt",
                     "path": f"docs/f{i}.txt", "dest_path": None,
                     "bytes_changed": 512})
    rows.append({"ts": base + 9.0, "op": "rename", "path": "docs/f0.txt",
                 "dest_path": "docs/f0.txt.locked", "bytes_changed": 0})
    rows.append({"ts": base + 9.5, "op": "note", "path": "_note.txt",
                 "dest_path": None, "bytes_changed": 80})
    return pd.DataFrame(rows)


from rr.eval.metrics import (alert_episodes, false_alarms_per_hour,  # noqa: E402
                             files_lost_before_detection, pr_auc,
                             point_adjusted_f1_DO_NOT_USE, time_to_detect)


def test_files_lost_before_detection() -> None:
    trace = _trace(6)
    assert files_lost_before_detection(trace, 1_000_000.0 + 2.4) == 5
    assert files_lost_before_detection(trace, 1_000_000.0 + 2.6) == 6


def test_time_to_detect_and_fa_per_hour() -> None:
    assert time_to_detect(100.0, 103.5) == pytest.approx(3.5)
    assert false_alarms_per_hour([1.0, 2.0, 3.0], 1.5) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        false_alarms_per_hour([1.0], 0.0)


def test_alert_episodes_merge_adjacent_firings() -> None:
    flags = [True, True, False, True, False, False, True, True]
    eps = alert_episodes(flags, gap_seconds=1.0)
    assert len(eps) == 3
    assert eps[0]["n_firing"] == 2 and eps[2]["n_firing"] == 2


def test_pr_auc_and_forbidden_metric() -> None:
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    with pytest.raises(RuntimeError):
        point_adjusted_f1_DO_NOT_USE()


# --------------------------------------------------------------------- #
# rr.eval.figures
# --------------------------------------------------------------------- #
def test_pareto_frontier_and_png(tmp_path) -> None:
    from rr.eval.figures import pareto_curve, pareto_frontier

    pts = [(0.5, 40.0, "quiet"), (2.0, 6.0, "balanced"),
           (8.0, 2.0, "twitchy"), (3.0, 9.0, "dominated")]
    out = tmp_path / "pareto.png"
    frontier = pareto_curve(pts, out)
    assert out.exists() and out.stat().st_size > 0
    labels = {p[2] for p in frontier}
    assert labels == {"quiet", "balanced", "twitchy"}
    # Identical points do not dominate each other: both stay on the frontier.
    assert {p[2] for p in pareto_frontier([(1.0, 1.0, "a"), (1.0, 1.0, "dup")])} == {"a", "dup"}


# --------------------------------------------------------------------- #
# rr.eval.ablation
# --------------------------------------------------------------------- #
def test_leave_one_feature_out_ranks_signal_first() -> None:
    from rr.eval.ablation import leave_one_feature_out

    rng = np.random.default_rng(0)
    n = 160
    signal = rng.normal(0, 1, n) + rng.choice([0.0, 6.0], n, p=[0.7, 0.3])
    frame = pd.DataFrame({
        "signal": signal,
        "noise": rng.normal(0, 1, n),
        "label": (rng.choice([0.0, 6.0], n, p=[0.7, 0.3]) > 0).astype(int),
    })

    def score(subset):
        from sklearn.ensemble import IsolationForest
        det = IsolationForest(n_estimators=64, random_state=0).fit(frame[list(subset)])
        return -det.score_samples(frame[list(subset)])

    result = leave_one_feature_out(frame, label_col="label", score_fn=score)
    assert list(result["feature"])[:1] == ["signal"]   # removing it hurts most
    assert set(result["feature"]) == {"signal", "noise"}


# --------------------------------------------------------------------- #
# rr.compat
# --------------------------------------------------------------------- #
def test_compat_end_to_end(tmp_path, monkeypatch) -> None:
    """Legacy rows -> v2 detector -> conformal p + Shapley + snapshot."""
    from pathlib import Path

    import rr.compat as compat
    from rr.sim.safety import create_arena

    arena = create_arena(tmp_path / "arena")
    (arena / "docs").mkdir()
    for i in range(10):
        (arena / "docs" / f"f{i}.txt").write_text(f"data {i}\n" * 40,
                                                  encoding="utf-8")
    rows = []
    base = 1_000_000.0
    for i in range(60):                       # benign events over ~30 s
        rows.append({"ts": base + i * 0.5, "session_id": "t_benign",
                     "event_type": "modified",
                     "src_path": str(arena / "docs" / f"f{i % 10}.txt"),
                     "dest_path": None, "entropy": 4.5, "divergent": 0,
                     "note_like": 0})

    det, full_frame, benign_frame = compat.fit_v2_detector(rows)
    assert det.is_calibrated and det.n_calibration >= 2
    assert list(full_frame.columns)[:2] == ["event_rate", "files_touched"]
    # Benign-session filtering must not remove attack windows from the frame
    # we score: full_frame keeps every window, benign_frame only calibration.
    assert len(benign_frame) <= len(full_frame)

    suffix, extra = compat.enhance_window_alert(
        {"source": "rule", "severity": "HIGH"},
        pd.Series({"window_start": base + 25.0, "window_end": base + 35.0}),
        rows, root=arena)
    assert "Conformal" in suffix and "Response" in suffix
    assert extra.get("files_snapshotted", 0) >= 10

    line = compat.protect_files([str(arena / "docs" / "f0.txt")], arena)
    assert "shadow journal" in line

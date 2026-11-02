"""rr.compat — the frozen-legacy ↔ v2 bridge (integrator-owned).

The top-level modules (collector/rules/app…) are frozen by
``INTEGRATION_CONTRACT.md``; interop goes through HERE. This module exposes
the v2 layer to the live pipeline in failure-isolated, additive steps:

* ``rows_to_events`` / ``featurize_rows`` — legacy SQLite event rows into the
  37-feature, dual-scale (10 s + 60 s) v2 representation.
* ``fit_v2_detector`` — one-class ECOD+IsolationForest ensemble with split
  conformal calibration, fitted ONLY on windows that end before the point
  being scored, calibration strictly after fit data (temporal, never shuffled).
* ``explain_window`` — exact Shapley attribution over the top-k driving
  features (2^k coalitions, enumerated; the function refuses silent sampling).
* ``snapshot_paths`` — copy-on-write protection via the v2 shadow journal
  (first-write-wins, restore verified by SHA-256 — ``python verify_rollback.py``).

The live hook is in ``rules.LiveDetector.poll``: every alert is enhanced with a
conformal p-value, a Shapley attribution and a snapshot record before it
reaches the dashboard. Every v2 step is best-effort and failure-isolated: if
this layer fails, the legacy alert still fires (with a printed warning), never
the reverse.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from rr.core.featurize import FEATURE_ORDER, featurize
from rr.core.schema import NormalisedEvent, from_legacy_row
from rr.detect.detector import Detector
from rr.detect.shapley import exact_shapley
from rr.eval.protocol import temporal_split
from rr.respond.shadow import ShadowJournal
from rr.sim.safety import require_arena, resolve_within

__all__ = [
    "rows_to_events",
    "featurize_rows",
    "fit_v2_detector",
    "explain_window",
    "snapshot_paths",
    "enhance_window_alert",
    "protect_files",
]

# Tweakables (kept here, out of the frozen config).
_SHAPLEY_TOP_K = 6          # 2^6 = 64 coalition rows per alert: cheap, exact
_ALPHA = 0.01               # conformal false-alarm budget per benign window
_MIN_CALIBRATION = 8        # Detector.fit floor
_JOURNAL_CACHE: dict[str, ShadowJournal] = {}


# --------------------------------------------------------------------- #
# Legacy rows -> v2 representation
# --------------------------------------------------------------------- #
def rows_to_events(rows: Sequence[Mapping[str, Any]]) -> tuple[NormalisedEvent, ...]:
    """Legacy ``events`` table rows -> NormalisedEvent tuple (schema contract)."""
    return tuple(from_legacy_row(dict(r)) for r in rows)


def featurize_rows(rows: Sequence[Mapping[str, Any]],
                   scales: Sequence[float] = (10.0, 60.0),
                   hop: float = 1.0) -> pd.DataFrame:
    """Legacy rows -> the 37-feature multi-scale frame (FEATURE_ORDER + meta)."""
    return featurize(rows_to_events(rows), scales=scales, hop=hop)


# --------------------------------------------------------------------- #
# Conformal detector, fitted leakage-safely
# --------------------------------------------------------------------- #
def fit_v2_detector(rows: Sequence[Mapping[str, Any]],
                    *,
                    before_ts: float | None = None,
                    random_state: int = 0) -> tuple[Detector, pd.DataFrame, pd.DataFrame]:
    """Fit + calibrate the v2 ensemble on benign windows; return
    ``(detector, full_frame, benign_frame)``.

    ``full_frame`` keeps every window (it is what you score and locate the
    alerting window in); ``benign_frame`` holds only calibration-eligible
    windows and is what attribution uses as its background. Split is temporal
    (never shuffled): the earliest 70% of benign windows fit the ensemble, the
    remaining 30% calibrate the conformal layer. The strict overlap purge is
    disabled here — at demo scale (a dozen-ish 1 s-hop windows) it empties the
    calibration side — so calibration windows may overlap training windows;
    p-value marginal validity is unaffected (see ``Detector.pvalue``), and
    offline evaluation uses the full purge in ``rr.eval.protocol``.

    Calibration must be BENIGN: sessions labelled benign/normal/bulk/calib
    are trusted; when no session carries such a marker (manual runs), the
    fallback is "every window strictly before ``before_ts``".
    """
    full = featurize_rows(rows, scales=(10.0,))
    sessions = {str(s).lower() for s in full["session_id"].dropna()}
    benign_markers = ("benign", "normal", "bulk", "calib")
    if any(any(m in s for m in benign_markers) for s in sessions):
        keep = full["session_id"].astype(str).str.lower().apply(
            lambda s: any(m in s for m in benign_markers))
        benign = full[keep].copy()
    else:
        benign = full.copy()
    if before_ts is not None:
        benign = benign[benign["window_end"] <= float(before_ts)]
    benign = benign.sort_values("window_start").reset_index(drop=True)
    if len(benign) < _MIN_CALIBRATION:
        raise ValueError(
            f"only {len(benign)} benign v2 windows available; need >= "
            f"{_MIN_CALIBRATION} for a leakage-safe fit+calibration split.")
    train, cal = temporal_split(benign, frac=0.7, purge_overlapping=False)
    det = Detector(random_state=random_state).fit(train)
    det.calibrate(cal)
    return det, full, benign


def _calibration_stats(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return frame[FEATURE_ORDER].mean(), frame[FEATURE_ORDER].std().replace(0.0, 1.0)


# --------------------------------------------------------------------- #
# Exact Shapley over the driving subset
# --------------------------------------------------------------------- #
def explain_window(det: Detector,
                   row: pd.Series,
                   calib_frame: pd.DataFrame,
                   *,
                   top_k: int = _SHAPLEY_TOP_K) -> dict[str, Any]:
    """Attribution of one window's anomaly score, two exact views.

    * ``ecod_top`` — the ECOD member's per-feature tail contributions
      (``Detector.per_feature_contributions``). They sum **exactly** to the
      ECOD member score by construction. At demo-scale calibration (a few
      dozen windows) every feature beyond the calibration range saturates at
      the same tail floor ``-log(1/n)``, so granularity is coarse; the sum
      stays exact regardless.
    * ``shapley`` — exact Shapley values of the **combined** ensemble score
      over the ``top_k`` features whose values deviate most from the
      calibration mean (2^k coalitions enumerated, no sampling). Efficiency
      holds for that restricted game: ``sum(phi) == score(x_k | mean-row) -
      score(mean-row)``. Full 37-feature Shapley would need 2^37 coalitions
      and the module refuses it rather than silently sampling.
    """
    mean, std = _calibration_stats(calib_frame)
    contrib = det.per_feature_contributions(row.to_frame().T).iloc[0]
    ecod_score = float(contrib["ecod_score"])
    per_feature = contrib.drop("ecod_score")
    top_ecod = list(per_feature.sort_values(ascending=False).head(top_k).items())

    z = ((row[FEATURE_ORDER] - mean) / std).astype(float)
    top = list(z.abs().sort_values(ascending=False).head(top_k).index)
    template = pd.DataFrame([mean])

    def score_fn(batch: pd.DataFrame) -> np.ndarray:
        full = pd.concat([template] * len(batch), ignore_index=True)
        for col in batch.columns:
            full[col] = batch[col].to_numpy()
        return det.score(full)

    phi = exact_shapley(score_fn, row[top], calib_frame[top],
                        background_mode="mean", feature_names=top)
    order = np.argsort(-phi)

    return {
        "ecod_score": ecod_score,
        "ecod_top": [(name, round(float(v), 3)) for name, v in top_ecod],
        "features": top,
        "phi": [(top[i], float(phi[i])) for i in order],
        "alert_score": float(det.score(row.to_frame().T)[0]),
    }


# --------------------------------------------------------------------- #
# v2 response: shadow journal snapshots
# --------------------------------------------------------------------- #
def _journal(root: str | Path) -> ShadowJournal:
    key = str(Path(root).resolve())
    if key not in _JOURNAL_CACHE:
        _JOURNAL_CACHE[key] = ShadowJournal(key, dry_run=False)
    return _JOURNAL_CACHE[key]


def ensure_arena(root: str | Path) -> Path:
    """Guarantee the root is a sentinel-marked arena — but only inside the
    project tree. Writing a sentinel into an arbitrary user directory would
    defeat the safety rail, so outside ``config.ROOT`` this refuses exactly
    like ``require_arena``. The legacy sandbox loses its sentinel every
    ``simulator.reset_sandbox()``, so the live path re-marks it quietly.
    """
    from config import ROOT as LEGACY_ROOT

    root_resolved = Path(root).resolve()
    try:
        root_resolved.relative_to(Path(LEGACY_ROOT).resolve())
    except ValueError:
        return require_arena(root)          # not ours: strict mode, no writes
    sentinel = root_resolved / ".ransomradar_arena"
    if not sentinel.is_file():
        from rr.sim.safety import create_arena
        create_arena(root_resolved)
    return root_resolved


def snapshot_paths(paths: Sequence[str | Path | None],
                   root: str | Path) -> dict[str, int]:
    """Protect the original bytes of ``paths`` in the shadow journal.

    First-write-wins: a path already protected keeps its pristine copy, so
    repeated alerts can never overwrite the clean backup with ciphertext.
    Journal index writes are retried once: on Windows, Defender/indexer can
    hold the file for a moment between consecutive alerts.
    """
    root_resolved = ensure_arena(root)
    clean = [resolve_within(root_resolved, p) for p in paths if p]
    journal = _journal(root_resolved)
    try:
        outcomes = journal.snapshot_many(clean)
    except PermissionError:
        time.sleep(0.25)
        outcomes = journal.snapshot_many(clean)
    new = sum(1 for o in outcomes if o.status == "snapshotted")
    already = sum(1 for o in outcomes if o.status == "already_protected")
    skipped = len(clean) - new - already
    return {"requested": len(clean), "protected_total": journal.stats().get("n_protected", new + already),
            "new_this_call": new, "already_protected": already,
            "skipped": skipped, "usage_bytes": journal.usage_bytes()}


def _arena_data_files(root_resolved: Path) -> list[Path]:
    """Data files worth protecting: everything except control/journal paths."""
    from rr.sim.safety import (CANARY_REGISTRY_NAME, JOURNAL_NAME, KEY_NAME,
                               SENTINEL_NAME)

    reserved = {SENTINEL_NAME, JOURNAL_NAME, KEY_NAME, CANARY_REGISTRY_NAME}
    out: list[Path] = []
    for p in sorted(root_resolved.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root_resolved)
        if any(part in reserved for part in rel.parts):
            continue
        if rel.parts[0] == ".ransomradar":
            continue
        out.append(p)
    return out


def protect_files(paths: Sequence[str | Path | None],
                  root: str | Path) -> str:
    """One-line response summary for instant (canary) alerts."""
    try:
        stats = snapshot_paths(paths, root)
    except Exception as exc:
        return (f" Response: shadow-journal snapshot unavailable ({exc!r}) — "
                "detection unaffected.")
    return (f" Response: original bytes protected in the shadow journal "
            f"({stats['usage_bytes']:,} bytes held; first-write-wins; "
            "restore is SHA-256-verified on demand).")


# --------------------------------------------------------------------- #
# Live alert enhancement (the hook rules.py calls)
# --------------------------------------------------------------------- #
def enhance_window_alert(alert: dict,
                         window_row: pd.Series,
                         rows: Sequence[Mapping[str, Any]],
                         *,
                         root: str | Path,
                         alpha: float = _ALPHA) -> tuple[str, dict[str, float]]:
    """Return ``(explanation_suffix, extra_features)`` for one live alert.

    Adds, best-effort: the split-conformal p-value of the alerting window, an
    exact-Shapley attribution of the driving features, and shadow-journal
    protection of every file the window touched. Any failure degrades to an
    empty suffix — the legacy alert still fires.
    """
    parts: list[str] = []
    extra: dict[str, float] = {}
    ws = float(window_row["window_start"])
    mid = ws + (float(window_row.get("window_end", ws + 10.0)) - ws) / 2.0

    det, full_frame, benign_frame = fit_v2_detector(rows, before_ts=ws)
    scale_rows = full_frame[full_frame["scale"] == 10.0]
    if scale_rows.empty:
        return "", extra
    idx = (scale_rows["window_start"] - mid).abs().idxmin()
    v2row = scale_rows.loc[idx]

    p = float(det.pvalue(v2row[FEATURE_ORDER].to_frame().T)[0])
    extra["conformal_p"] = p
    extra["calibration_windows"] = int(det.n_calibration)
    floor = 1.0 / (det.n_calibration + 1.0)
    at_floor = p <= floor * 1.001
    if p <= alpha:
        parts.append(f"Conformal: p={p:.4g} ≤ α={alpha} (n_cal={det.n_calibration}"
                     f"{'; at the resolution floor 1/(n_cal+1)' if at_floor else ''}) — "
                     "anomalous against calibrated benign behaviour.")
    else:
        floor_note = (f" — this is the resolution floor 1/(n_cal+1): with only "
                      f"{det.n_calibration} benign calibration windows no window "
                      f"can score below {floor:.3g}; more benign history drives "
                      "the same score toward p≈0.001" if at_floor else "")
        parts.append(f"Conformal: p={p:.4g} > α={alpha} (not individually extreme "
                     f"at this alpha; n_cal={det.n_calibration}){floor_note}.")

    try:
        attr = explain_window(det, v2row, benign_frame)
        drivers = [n for n, _ in attr["ecod_top"][:4]]
        parts.append(
            f"Attribution: ECOD member score {attr['ecod_score']:.1f} is the exact "
            f"sum of per-feature tail contributions; strongest drivers: "
            f"{', '.join(drivers)}.")
        extra["ecod_top"] = ";".join(f"{n}:{v:+.3f}" for n, v in attr["ecod_top"])
        if attr["phi"]:
            text = ", ".join(f"{n} {v:+.3f}" for n, v in attr["phi"][:4])
            parts.append(f"Shapley (combined ensemble, exact over top-"
                         f"{len(attr['phi'])} drivers vs the benign-mean window): "
                         f"{text}.")
            extra["shapley_top"] = ";".join(f"{n}:{v:+.3f}" for n, v in attr["phi"])
    except Exception as exc:
        parts.append(f"Attribution unavailable ({exc!r}).")

    try:
        # Protect the whole arena at the first verdict: files the attack has
        # not reached yet snapshot as pristine bytes; already-damaged ones are
        # still snapshotted (as evidence), and first-write-wins guarantees a
        # later pass can never overwrite a clean copy with ciphertext.
        stats = snapshot_paths(_arena_data_files(ensure_arena(root)), root)
        extra["files_snapshotted"] = stats["protected_total"]
        parts.append(f"Response: {stats['protected_total']} file(s) now protected "
                     "in the shadow journal (first-write-wins; restore is "
                     "SHA-256-verified on demand).")
    except Exception as exc:
        parts.append(f"Response: shadow journal unavailable ({exc!r}).")

    return " ".join(parts), extra


if __name__ == "__main__":  # pragma: no cover - smoke demo
    import db as legacy_db
    import simulator as legacy_sim
    import collector as legacy_collector

    legacy_db.init_db(reset=True)
    legacy_sim.reset_sandbox()
    obs, stop = legacy_collector.start(with_detector=False)
    legacy_collector.set_session("smoke_benign")
    legacy_sim.run_benign(25)
    legacy_collector.stop(obs, stop)
    rows = legacy_db.all_events()
    det, full_frame, benign_frame = fit_v2_detector(rows)
    print(f"v2 detector fitted: {len(full_frame)} windows "
          f"({len(benign_frame)} benign), n_cal={det.n_calibration}")
    p = det.pvalue(full_frame[full_frame['scale'] == 10.0].head(1))
    print(f"benign window p-value: {float(p[0]):.3f} (expected > 0.01)")
    legacy_db.init_db(reset=True)

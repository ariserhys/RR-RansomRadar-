"""ML detection: a learned "normal envelope" + Isolation Forest score.

Two components live in the persisted bundle:

1. The normal envelope (the calibrated detector). For every feature we store
   the largest value observed across the normal training windows, padded for
   headroom; features that were ~0 throughout normal training get absolute
   floors instead (see config.ENVELOPE_FLOORS). A window alerts when any
   feature exceeds its envelope - i.e. when activity goes beyond anything
   ever seen in normal behaviour. Because normal training deliberately
   includes bulk uploads, the envelope covers them and ML does not
   false-alarm on legitimate bulk operations, while attacks exceed the
   envelope on the *shape* features (edits of existing files, extension
   changes, deletions). The violating feature IS the plain-language evidence.

2. Isolation Forest (sklearn), the spec'd anomaly model. With the small
   training sets a demo produces its raw verdict is unreliable, so it is
   used as a corroborating 0-100 anomaly score shown alongside the envelope.
"""
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

import features
from config import ENVELOPE_FLOORS, ENVELOPE_PAD, FEATURES, MODEL_PATH

FEATURE_LABELS = {
    "event_rate": "event rate",
    "files_touched": "files touched",
    "created": "new files created",
    "existing_writes": "edits to existing files",
    "renames": "file renames",
    "extension_changes": "extension changes",
    "deletions": "file deletions",
    "canary_touched": "canary touches",
    "entropy_max": "peak content entropy",
    "type_divergence": "files whose type no longer matches their content",
}


def train(windows_df):
    """Learn the envelope + fit the forest on normal windows; persist bundle."""
    X = windows_df[FEATURES].astype(float)
    envelope = {f: max(float(ENVELOPE_FLOORS.get(f, 0.0)),
                       ENVELOPE_PAD * float(X[f].max()))
                for f in FEATURES}

    contamination = 0.05 if len(X) >= 20 else 0.10
    forest = IsolationForest(n_estimators=100, contamination=contamination,
                             random_state=42)
    forest.fit(X)
    raw = -forest.decision_function(X)          # higher = more anomalous

    bundle = {
        "envelope": envelope,
        "features": FEATURES,
        "baseline": features.learn_baseline(windows_df),   # raw units, for text
        "model": forest,
        "trained_on_windows": int(len(X)),
        "raw_min": float(raw.min()),
        "raw_span": float(max(raw.max() - raw.min(), 1e-6)),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def worst_violation(bundle, row):
    """Return (feature, value, allowed_max) for the biggest envelope breach,
    or None if the window stays inside everything seen in normal training."""
    best = None
    for f in bundle["features"]:
        v = float(row[f])
        a = float(bundle["envelope"][f])
        if v > a and (best is None or (v - a) > (best[1] - best[2])):
            best = (f, v, a)
    return best


def ml_alert(bundle, row):
    return worst_violation(bundle, row) is not None


def anomaly_score(bundle, row):
    """Isolation Forest output mapped to a 0-100 anomaly score (display)."""
    X = pd.DataFrame([[float(row[f]) for f in bundle["features"]]],
                     columns=bundle["features"])
    raw = -bundle["model"].decision_function(X)[0]
    s = (raw - bundle["raw_min"]) / bundle["raw_span"] * 100.0
    return max(0.0, min(100.0, s))


def describe_ml(bundle, row):
    """Human-readable ML evidence: the worst envelope breach + IF score."""
    parts = []
    violation = worst_violation(bundle, row)
    if violation:
        f, val, allowed = violation
        label = FEATURE_LABELS.get(f, f)
        fmt = lambda x: f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"
        parts.append(f"{label} {fmt(val)} exceeds the learned normal envelope "
                     f"(max seen in normal training ~{fmt(allowed)})")
    parts.append(f"Isolation Forest anomaly score "
                 f"{anomaly_score(bundle, row):.0f}/100")
    return "; ".join(parts)


if __name__ == "__main__":
    import db

    db.init_db()
    events = db.all_events()
    win, _ = features.windows_from_events(events, fill_idle=True)
    if len(win) < 5:
        raise SystemExit("Not enough data yet. Run a benign simulation with the "
                         "collector running first (need at least 5 windows).")
    bundle = train(win)
    print(f"Trained on {len(win)} windows ({len(events)} events) -> {MODEL_PATH}")
    for f, a in bundle["envelope"].items():
        print(f"  envelope {f:<18} <= {a:.1f}")

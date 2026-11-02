"""Rule engine (canary + burst rules), rules/ML fusion, and the live detector.

Detection has three flavours, used both live and in the experiments:
  rules-only  - static thresholds + instant canary alerts
  ML-only     - Isolation Forest anomaly verdict per window
  combined    - fused alert carrying both kinds of evidence
"""
import os
import threading
import time

import db
import features
import model
import notify
from config import (CANARY_ALERT_COOLDOWN, DETECTOR_POLL_SECONDS, FEATURES,
                    RULE_DELETIONS, RULE_EVENT_RATE_PER_MIN,
                    RULE_EXTENSION_CHANGES, RULE_RENAMES, SANDBOX,
                    WINDOW_SECONDS)

# v2 layer (rr/compat): adds conformal p-values, Shapley attribution and
# shadow-journal snapshots to alerts. Best-effort and failure-isolated: if it
# is unavailable, the legacy alert path works exactly as before.
try:
    import rr.compat as _v2
except Exception as _v2_exc:                        # pragma: no cover
    _v2 = None
    print(f"[detector] rr.compat unavailable ({_v2_exc!r}); "
          "alerts run legacy-only", flush=True)

SEVERITY_RANK = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}


# --- static rules -------------------------------------------------------------

def rule_hits(row):
    """Return [(rule_name, severity, description)] for one feature window."""
    hits = []
    if row["canary_touched"]:
        hits.append(("canary_touched", "CRITICAL",
                     "a protected canary file was touched"))
    if row["extension_changes"] >= RULE_EXTENSION_CHANGES:
        hits.append(("mass_extension_change", "HIGH",
                     f"{int(row['extension_changes'])} file extensions changed "
                     "in one window"))
    if row["deletions"] >= RULE_DELETIONS:
        hits.append(("mass_deletion", "HIGH",
                     f"{int(row['deletions'])} deletions in one window"))
    if row["event_rate"] >= RULE_EVENT_RATE_PER_MIN:
        hits.append(("event_burst", "HIGH",
                     f"{row['event_rate']:.0f} events/min "
                     f"(threshold {RULE_EVENT_RATE_PER_MIN})"))
    elif row["renames"] >= RULE_RENAMES:
        hits.append(("rename_burst", "MEDIUM",
                     f"{int(row['renames'])} renames in one window"))
    if row["ransom_notes"] >= 1:
        hits.append(("ransom_note", "HIGH",
                     f"{int(row['ransom_notes'])} ransom-note-style file(s) "
                     "created (matches known ransomware note patterns)"))
    return hits


def evaluate_window(row, evidence=None, bundle=None):
    """Fuse rules + ML for one window. Returns an alert dict or None.

    bundle=None -> rules only; bundle=model bundle -> rules + ML fused.
    """
    hits = rule_hits(row)
    ml_score = None
    ml_flag = False
    if bundle:
        ml_score = model.anomaly_score(bundle, row)
        ml_flag = model.ml_alert(bundle, row)
    if not hits and not ml_flag:
        return None

    parts = []
    if hits:
        parts.append("Rules: " + "; ".join(t for _, _, t in hits) + ".")
    if ml_flag:
        parts.append("ML: " + model.describe_ml(bundle, row) + ".")
    baseline = bundle.get("baseline") if bundle else None
    parts.append("Evidence: " + features.explain_window(row, baseline) + ".")
    if evidence:
        parts.append("Sample activity: " + " | ".join(evidence[:4]))

    if hits and ml_flag:
        source = "fused"
        severity = max((h[1] for h in hits), key=lambda s: SEVERITY_RANK[s])
    elif hits:
        source, severity = "rule", max((h[1] for h in hits),
                                       key=lambda s: SEVERITY_RANK[s])
    else:
        source, severity = "ml", "HIGH"

    return {
        "source": source,
        "rule_name": ",".join(h[0] for h in hits) or "isolation_forest",
        "severity": severity,
        "ml_score": float(ml_score) if ml_score is not None else 0.0,
        "explanation": " ".join(parts),
    }


# --- live detector (runs inside the collector process) --------------------------

class LiveDetector:
    """Canary events alert instantly; closed 10 s windows go through fusion."""

    def __init__(self):
        self.bundle = model.load_bundle()
        self._last_canary = 0.0

    def _canary_check(self, events, now):
        recent = [e for e in events if e["ts"] >= now - 2 * WINDOW_SECONDS]
        cooldown = max(self._last_canary, db.last_canary_alert_ts())
        for e in recent:
            if features.is_canary_path(e["src_path"]) or features.is_canary_path(e["dest_path"]):
                if now - cooldown >= CANARY_ALERT_COOLDOWN:
                    self._last_canary = now
                    cooldown = now
                    name = os.path.basename(e["src_path"] or "?")
                    stamp = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
                    response = ""
                    if _v2 is not None:
                        try:
                            response = _v2.protect_files(
                                [e["src_path"], e["dest_path"]], SANDBOX)
                        except Exception as exc:    # legacy alert must survive
                            print(f"[detector] v2 snapshot skipped: {exc!r}",
                                  flush=True)
                    db.insert_alert(
                        ts=e["ts"], session_id=e.get("session_id"),
                        source="rule", rule_name="canary_touched",
                        severity="CRITICAL", ml_score=0.0,
                        window_start=None, window_end=None,
                        explanation=(
                            f"Canary decoy '{name}' was {e['event_type']} at {stamp}. "
                            "Canary files are never used by legitimate software, so "
                            "touching one is a strong ransomware early-warning signal. "
                            "Investigate immediately." + response),
                    )
                    notify.alert("CRITICAL", "RansomRadar canary triggered",
                                 f"Canary decoy '{name}' was {e['event_type']} - "
                                 "possible ransomware activity in the sandbox.")
                    print(f"[detector] CANARY ALERT: {name} {e['event_type']}", flush=True)

    def poll(self):
        if self.bundle is None and model.MODEL_PATH.exists():
            self.bundle = model.load_bundle()   # picks up training done while live
        now = time.time()
        events = db.all_events()
        if not events:
            return 0

        self._canary_check(events, now)

        wins, evidence = features.windows_from_events(events)
        n_new = 0
        for _, row in wins.iterrows():
            ws = row["window_start"]
            if ws + WINDOW_SECONDS > now - 2:      # not closed yet (grace for stragglers)
                continue
            if db.alert_exists_window(ws):
                continue
            alert = evaluate_window(row, evidence.get(ws), self.bundle)
            if alert:
                suffix, extra = "", {}
                if _v2 is not None:
                    try:
                        suffix, extra = _v2.enhance_window_alert(
                            alert, row, events, root=SANDBOX)
                    except Exception as exc:    # legacy alert must survive
                        print(f"[detector] v2 enhancement skipped: {exc!r}",
                              flush=True)
                db.insert_alert(
                    ts=ws + WINDOW_SECONDS, session_id=row.get("session_id"),
                    source=alert["source"], rule_name=alert["rule_name"],
                    severity=alert["severity"], ml_score=alert["ml_score"],
                    window_start=ws, window_end=ws + WINDOW_SECONDS,
                    explanation=alert["explanation"] + suffix,
                    features={**{f: float(row[f]) for f in FEATURES}, **extra},
                )
                n_new += 1
                notify.alert(alert["severity"], f"RansomRadar {alert['source']} alert",
                             alert["explanation"])
                print(f"[detector] {alert['severity']} alert ({alert['source']}): "
                      f"{alert['rule_name']}", flush=True)
        return n_new

    def loop(self, stop_event):
        while not stop_event.is_set():
            try:
                self.poll()
            except Exception as exc:            # keep the collector alive no matter what
                print(f"[detector] error: {exc!r}", flush=True)
            stop_event.wait(DETECTOR_POLL_SECONDS)

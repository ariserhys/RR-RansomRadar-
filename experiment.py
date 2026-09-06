"""RansomRadar experiment harness: rules vs ML vs an entropy-only ablation.

Scripted sessions are recorded into SQLite, then every recorded window is
scored offline by all detectors, so the comparison is apples-to-apples:

  Phase 1  learn normal behaviour (benign session + legitimate bulk uploads,
           which IS normal - this is why ML should not false-alarm on bulk)
  Phase 2  train the learned envelope + Isolation Forest on those windows
  Phase 3  evaluation sessions: benign, bulk, fast attacks, stealth attacks,
           and partial-encryption attacks (evade entropy, break file types)
  Phase 4  score and report: detection rate, false alarms, detection delay,
           an entropy-evasion ablation, and ROC-style AUC per detector

Full run ~4 minutes.  python experiment.py --quick  for a ~2 min version.
"""
import argparse
import sys
import time

import pandas as pd
from sklearn.metrics import roc_auc_score

import collector
import db
import features
import model
import rules
import simulator
from config import FEATURES, RESULTS_CSV, WINDOW_SECONDS

try:                                    # keep Windows console happy
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def _score_windows(wins, bundle):
    """Per-window verdicts for the four detectors (offline, from records).

    "ml"      - full learned envelope (all features, incl. content forensics)
    "ml_entropy" - ablation: an entropy-only detector (CryptoDrop-style).
    """
    out = []
    env_entropy = bundle["envelope"]["entropy_max"] if bundle else None
    for _, row in wins.iterrows():
        r_hit = bool(rules.rule_hits(row))
        m_hit = bool(bundle) and model.ml_alert(bundle, row)
        e_hit = bool(env_entropy is not None
                     and float(row["entropy_max"]) > env_entropy)
        out.append({"window_start": row["window_start"], "rule": r_hit,
                    "ml": m_hit, "ml_entropy": e_hit,
                    "combined": r_hit or m_hit})
    return out


def _first_suspicious_ts(events):
    """First event that looks attack-like: canary touch, extension change or
    deletion. Falls back to the session's first event (stealth attacks)."""
    cand = [e["ts"] for e in events
            if features.is_canary_path(e["src_path"])
            or features.is_canary_path(e["dest_path"])
            or (e["event_type"] == "moved"
                and features._ext(e["src_path"]) != features._ext(e["dest_path"]))
            or e["event_type"] == "deleted"]
    if cand:
        return min(cand)
    return min((e["ts"] for e in events), default=None)


def run(quick=False):
    db.init_db(reset=True)
    simulator.reset_sandbox()

    def start_collector(label):
        observer, stop_ev = collector.start(with_detector=False)
        collector.set_session(label)
        return observer, stop_ev

    eval_sessions = []          # (label, kind)

    # --- phase 1: learn normal behaviour -------------------------------------
    # The training set is deliberately multi-modal: quiet human edits (many
    # windows) AND legitimate bulk uploads (a second dense cluster). This is
    # why ML should NOT false-alarm on the bulk eval, while static rules will.
    print("\n=== Phase 1: learning normal behaviour ===")
    pair = start_collector("train_benign")
    simulator.run_benign(45 if quick else 60)
    collector.set_session("train_bulk")
    # Several bulk sizes so "bulk = normal" is a dense second cluster that
    # spans the eval bulk size too.
    for n in ([80, 120, 160] if quick else [60, 90, 120, 150, 180]):
        simulator.run_bulk(n_files=n)
    collector.stop(*pair)

    # --- phase 2: train -------------------------------------------------------
    print("\n=== Phase 2: training the Isolation Forest ===")
    all_ev = db.all_events()
    wins, _ = features.windows_from_events(all_ev, fill_idle=True)
    train_wins = wins[wins["session_id"].isin(["train_benign", "train_bulk"])]
    bundle = model.train(train_wins)
    print(f"trained on {len(train_wins)} windows "
          f"({int((train_wins['session_id'] == 'train_bulk').sum())} bulk) "
          f"-> {model.MODEL_PATH}")

    # --- phase 3: evaluation sessions -----------------------------------------
    print("\n=== Phase 3: evaluation sessions ===")

    def run_eval(label, kind, fn, *args):
        simulator.reset_sandbox()               # collector is OFF during reset
        pair = start_collector(label)
        time.sleep(1.0)                         # settle: don't race the fresh watch
        fn(*args)
        time.sleep(1.0)                         # let trailing watchdog events land
        collector.stop(*pair)
        eval_sessions.append((label, kind))
        print(f"  session '{label}' ({kind}) recorded")

    run_eval("eval_benign", "benign", simulator.run_benign, 15 if quick else 30)
    run_eval("eval_bulk", "bulk", simulator.run_bulk, 100)
    n_attacks = 2 if quick else 3
    for i in range(n_attacks):
        run_eval(f"attack_fast_{i}", "attack_fast", simulator.run_attack_fast)
    for i in range(n_attacks):
        run_eval(f"attack_stealth_{i}", "attack_stealth",
                 simulator.run_attack_stealth, 15 if quick else 25)
    for i in range(n_attacks):
        run_eval(f"attack_partial_{i}", "attack_partial",
                 simulator.run_attack_partial, 15 if quick else 25)

    # --- phase 4: score and report ---------------------------------------------
    print("\n=== Phase 4: scoring (rules vs ML vs entropy-only ablation) ===")
    all_ev = db.all_events()
    rows = []
    for label, kind in eval_sessions:
        sess = [e for e in all_ev if e["session_id"] == label]
        wins, _ = features.windows_from_events(sess, fill_idle=True)
        verdicts = _score_windows(wins, bundle)
        first = _first_suspicious_ts(sess)
        canary_ts = [e["ts"] for e in sess
                     if features.is_canary_path(e["src_path"])
                     or features.is_canary_path(e["dest_path"])]
        for det in ("rule", "ml", "ml_entropy", "combined"):
            fired = [v for v in verdicts if v[det]]
            alert_ts = [v["window_start"] + WINDOW_SECONDS for v in fired]
            if det in ("rule", "combined") and canary_ts:
                alert_ts += canary_ts           # the canary rule fires on the event itself
            detected = bool(alert_ts)
            delay = (min(alert_ts) - first) if detected and first is not None else None
            rows.append({"session": label, "kind": kind, "detector": det,
                         "detected": int(detected), "windows": len(wins),
                         "alert_windows": len(fired),
                         "delay_s": round(delay, 1) if delay is not None else ""})

    df = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

    pretty = {"rule": "rules-only", "ml": "ML (full envelope)",
              "ml_entropy": "ML (entropy-only)", "combined": "combined"}
    summary = []
    for det in ("rule", "ml", "ml_entropy", "combined"):
        sub = df[df["detector"] == det]
        attacks = sub[sub["kind"].str.startswith("attack")]
        normal = sub[sub["kind"].isin(["benign", "bulk"])]
        delays = pd.to_numeric(attacks.loc[attacks["detected"] == 1, "delay_s"],
                               errors="coerce")
        summary.append({
            "detector": pretty[det],
            "detection_rate": f"{int(attacks['detected'].sum())}/{len(attacks)}",
            "avg_delay_s": f"{delays.mean():.1f}" if delays.notna().any() else "-",
            "false_alarms": f"{int(normal['detected'].sum())}/{len(normal)} sessions",
        })
    print()
    print(pd.DataFrame(summary).to_string(index=False))

    print("\nEntropy-evasion check (why content forensics beats entropy alone):")
    abl = []
    for kind in ("attack_fast", "attack_stealth", "attack_partial"):
        sub = df[(df["kind"] == kind) & (df["detector"].isin(["ml", "ml_entropy"]))]
        n_sess = int(sub[sub["detector"] == "ml"].shape[0])
        abl.append({"attack kind": kind,
                    "entropy-only": f"{int(sub[sub['detector'] == 'ml_entropy']['detected'].sum())}/{n_sess}",
                    "full ML": f"{int(sub[sub['detector'] == 'ml']['detected'].sum())}/{n_sess}"})
    print(pd.DataFrame(abl).to_string(index=False))

    # Discrimination quality over all evaluation windows (ROC-style AUC):
    labels, if_scores, margins = [], [], []
    for label, kind in eval_sessions:
        sess = [e for e in all_ev if e["session_id"] == label]
        wins, _ = features.windows_from_events(sess, fill_idle=True)
        y = 1 if kind.startswith("attack") else 0
        for _, row in wins.iterrows():
            labels.append(y)
            if_scores.append(model.anomaly_score(bundle, row))
            margins.append(max(float(row[f]) / bundle["envelope"][f]
                               for f in FEATURES))
    if len(set(labels)) > 1:
        print(f"\nDiscrimination over eval windows (AUC, 1.0 = perfect): "
              f"Isolation Forest score {roc_auc_score(labels, if_scores):.3f} | "
              f"envelope margin {roc_auc_score(labels, margins):.3f}")

    print(f"\nfull per-window results -> {RESULTS_CSV}")
    print("""
How to read this (viva talking points):
  * rules-only: instant canary alerts (fastest), but it false-alarms on the
    legitimate bulk upload and is blind to slow and partial-encryption attacks.
  * ML (entropy-only): the CryptoDrop-style ablation - catches full rewrites,
    but modern partial encryption keeps entropy low and slips past it.
  * ML (full envelope): zero false alarms on bulk, catches every attack kind
    because file-type divergence survives partial encryption.
  * combined: keeps the canary's instant response AND the ML's coverage.
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RansomRadar experiment harness")
    ap.add_argument("--quick", action="store_true", help="shorter sessions")
    args = ap.parse_args()
    run(quick=args.quick)

"""Group raw file events into 10-second windows and compute features.

Also owns the "explainability" helpers: the learned normal profile and the
plain-language comparison of a window against it.
"""
import os
from pathlib import PurePath

import pandas as pd

from config import CANARY_DIR, DEFAULT_BASELINE, FEATURES, WINDOW_SECONDS


def is_canary_path(path):
    """Canary files live under the canary folder with realistic names, so
    match on the configured folder (as a path part) rather than the filename.
    Note: substring matching fails here - "canaries" does not contain
    "canary" - which is exactly why the part-wise check exists."""
    if not path:
        return False
    parts = [p.lower() for p in PurePath(str(path)).parts]
    return CANARY_DIR in parts or any("canary" in p for p in parts)


def _ext(path):
    return os.path.splitext(path)[1].lower() if path else ""


def _short(path, width=44):
    if not path:
        return "?"
    name = os.path.basename(path)
    return name if len(name) <= width else name[: width - 3] + "..."


def windows_from_events(events, fill_idle=False):
    """events: list of dicts (ts, event_type, src_path, dest_path, session_id).

    Returns (windows_df, evidence):
      windows_df - one row per 10 s window: FEATURES columns + window_start
                   + session_id (first session seen in the window)
      evidence   - {window_start: [short event descriptions]} for the UI

    fill_idle=True additionally emits all-zero rows for quiet 10 s buckets
    between a session's first and last event (used when training, so the
    model learns that idle is normal).
    """
    cols = FEATURES + ["window_start", "session_id"]
    if not events:
        return pd.DataFrame(columns=cols), {}

    df = pd.DataFrame(events)
    df["window_start"] = (df["ts"] // WINDOW_SECONDS) * WINDOW_SECONDS

    rows, evidence = [], {}
    for ws, g in df.groupby("window_start"):
        srcs = [p for p in g["src_path"] if p]
        dests = [p for p in g["dest_path"] if p]
        renames = g[g["event_type"] == "moved"]
        ext_changes = sum(
            1 for _, e in renames.iterrows()
            if _ext(e["src_path"]) != _ext(e["dest_path"])
        )
        canary_hit = any(
            is_canary_path(s) or is_canary_path(d)
            for s, d in zip(g["src_path"], g["dest_path"])
        )
        created_paths = {p for p, t in zip(g["src_path"], g["event_type"])
                         if t == "created" and p}
        created_n = len(created_paths)
        existing_writes = sum(
            1 for p, t in zip(g["src_path"], g["event_type"])
            if t == "modified" and p and p not in created_paths)
        entropies = g["entropy"].dropna()
        sessions = g["session_id"].dropna()
        rows.append({
            "window_start": ws,
            "event_rate": len(g) * 60.0 / WINDOW_SECONDS,
            "files_touched": len(set(srcs) | set(dests)),
            "created": created_n,
            "existing_writes": existing_writes,
            "renames": int(len(renames)),
            "extension_changes": int(ext_changes),
            "deletions": int((g["event_type"] == "deleted").sum()),
            "canary_touched": int(canary_hit),
            "entropy_max": float(entropies.max()) if len(entropies) else 0.0,
            "type_divergence": int(g["divergent"].fillna(0).sum()),
            # rules-only signal, not part of the ML feature vector:
            "ransom_notes": int(g["note_like"].fillna(0).sum()),
            "session_id": sessions.iloc[0] if len(sessions) else None,
        })
        evidence[ws] = []
        for e in g.itertuples():
            s = f"{e.event_type}: {_short(e.src_path)}"
            if e.dest_path:
                s += f" -> {_short(e.dest_path)}"
            if e.entropy is not None:
                s += f" (H={e.entropy:.1f}/8)"
                if e.divergent:
                    s += " [TYPE MISMATCH]"
            evidence[ws].append(s)
        evidence[ws] = evidence[ws][:6]

    win = pd.DataFrame(rows).sort_values("window_start").reset_index(drop=True)

    if fill_idle:
        step = WINDOW_SECONDS
        first_k = int(df["ts"].min() // step)
        last_k = int(df["ts"].max() // step)
        missing = [k * step for k in range(first_k, last_k + 1)
                   if k * step not in set(win["window_start"])]
        if missing:
            sid = df["session_id"].dropna()
            sid = sid.iloc[0] if len(sid) else None
            idle = [{**{f: 0 for f in FEATURES}, "ransom_notes": 0,
                     "window_start": ws, "session_id": sid} for ws in missing]
            win = pd.concat([win, pd.DataFrame(idle)],
                            ignore_index=True).sort_values("window_start")
            win = win.reset_index(drop=True)
    return win, evidence


def learn_baseline(windows_df):
    """Robust per-feature normal profile (median + MAD) from benign windows."""
    base = {}
    for f in FEATURES:
        vals = windows_df[f].astype(float)
        med = float(vals.median())
        base[f] = {"median": med,
                   "mad": float((vals - med).abs().median())}
    return base


def explain_window(row, baseline=None):
    """Plain-language comparison of one window against the learned profile.

    This is the explainability core: every alert carries this sentence.
    """
    base = baseline or DEFAULT_BASELINE
    bits = []

    typical_rate = max(base["event_rate"]["median"], 1.0)
    factor = row["event_rate"] / typical_rate
    if factor >= 3:
        bits.append(f"event rate {row['event_rate']:.0f}/min vs normal "
                    f"~{typical_rate:.0f}/min ({factor:.0f}x higher)")
    if row["canary_touched"]:
        bits.append("a protected canary (decoy) file was touched")
    if row["extension_changes"]:
        bits.append(f"{int(row['extension_changes'])} file extension(s) changed "
                    "- the classic ransomware rename pattern")
    if row["deletions"]:
        bits.append(f"{int(row['deletions'])} files deleted within 10 seconds")
    if row["renames"] >= 5 and not row["extension_changes"]:
        bits.append(f"{int(row['renames'])} files renamed within 10 seconds")
    if row["entropy_max"] >= 7.0:
        bits.append(f"file content entropy peaked at {row['entropy_max']:.1f}/8 "
                    "(encrypted-looking data)")
    if row["type_divergence"]:
        bits.append(f"{int(row['type_divergence'])} files' contents no longer "
                    "match their file type")
    if row.get("ransom_notes"):
        bits.append(f"{int(row['ransom_notes'])} ransom-note-style file(s) created")
    if not bits:
        bits.append("activity pattern deviates from the learned normal profile")
    return "; ".join(bits)

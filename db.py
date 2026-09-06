"""SQLite storage layer: file events in, explainable alerts out."""
import json
import sqlite3
from contextlib import closing

import pandas as pd

from config import DB_PATH, DB_SCHEMA_CREATE


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def init_db(reset=False):
    """Create the schema; with reset=True, wipe events and alerts first."""
    with closing(connect()) as conn:
        if reset:
            conn.executescript("DROP TABLE IF EXISTS alerts; DROP TABLE IF EXISTS events;")
        conn.executescript(DB_SCHEMA_CREATE)
        _migrate(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.commit()


def _migrate(conn):
    """Add columns introduced after the first release to existing databases."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
    for name, decl in (("entropy", "REAL"), ("divergent", "INTEGER"),
                       ("note_like", "INTEGER")):
        if name not in cols:
            conn.execute(f"ALTER TABLE events ADD COLUMN {name} {decl}")


# --- events -----------------------------------------------------------------

def insert_event(ts, event_type, src_path, dest_path=None, session_id=None,
                 entropy=None, divergent=None, note_like=None):
    with closing(connect()) as conn:
        conn.execute(
            "INSERT INTO events (ts, session_id, event_type, src_path, dest_path, "
            "entropy, divergent, note_like) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, session_id, event_type, src_path, dest_path,
             entropy, divergent, note_like),
        )
        conn.commit()


def all_events(session_id=None):
    q = ("SELECT id, ts, session_id, event_type, src_path, dest_path, "
         "entropy, divergent, note_like FROM events")
    params = []
    if session_id:
        q += " WHERE session_id = ?"
        params.append(session_id)
    q += " ORDER BY id"
    with closing(connect()) as conn:
        return [dict(r) for r in conn.execute(q, params)]


def recent_events_df(limit=15):
    q = ("SELECT * FROM (SELECT * FROM events ORDER BY id DESC LIMIT ?) "
         "ORDER BY id ASC")
    with closing(connect()) as conn:
        return pd.read_sql_query(q, conn, params=(int(limit),))


# --- alerts -------------------------------------------------------------------

def insert_alert(ts, session_id, source, rule_name, severity, ml_score,
                 window_start, window_end, explanation, features=None):
    with closing(connect()) as conn:
        conn.execute(
            "INSERT INTO alerts (ts, session_id, source, rule_name, severity, "
            "ml_score, window_start, window_end, explanation, features_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, session_id, source, rule_name, severity, ml_score,
             window_start, window_end, explanation,
             json.dumps(features) if features else None),
        )
        conn.commit()


def all_alerts_df():
    with closing(connect()) as conn:
        return pd.read_sql_query("SELECT * FROM alerts ORDER BY ts DESC", conn)


def all_alerts():
    return all_alerts_df().to_dict("records")


def alert_exists_window(window_start):
    with closing(connect()) as conn:
        row = conn.execute("SELECT 1 FROM alerts WHERE window_start = ? LIMIT 1",
                           (window_start,)).fetchone()
        return row is not None


def last_canary_alert_ts():
    with closing(connect()) as conn:
        row = conn.execute(
            "SELECT MAX(ts) AS m FROM alerts WHERE rule_name = 'canary_touched'"
        ).fetchone()
        return row["m"] or 0.0


# --- misc ---------------------------------------------------------------------

def count(table):
    assert table in ("events", "alerts")
    with closing(connect()) as conn:
        return conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]

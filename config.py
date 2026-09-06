"""RansomRadar - shared configuration and paths.

Defensive academic prototype: every file RansomRadar touches lives inside the
sandbox folder below. Detection is passive (rows in a local SQLite database);
it never interferes with the system.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SANDBOX = ROOT / "sandbox"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "ransomradar.db"
MODEL_PATH = DATA_DIR / "model.joblib"
RESULTS_CSV = DATA_DIR / "experiment_results.csv"

# --- detection parameters -------------------------------------------------
WINDOW_SECONDS = 10           # sliding-window size for feature extraction
DETECTOR_POLL_SECONDS = 3     # live detector poll interval
CANARY_ALERT_COOLDOWN = 20    # min seconds between duplicate canary alerts

# --- canary (decoy) files ---------------------------------------------------
# Realistic names and content: a careful attacker may inspect candidate files,
# and a decoy that screams "decoy" is a wasted tripwire. Detection keys on the
# canary FOLDER (see features.is_canary_path), not on the word in the name.
CANARY_DIR = "canaries"       # subfolder of SANDBOX holding the decoys
CANARY_FILES = [
    "Q4_Payroll_Report.docx",
    "FY27_Budget_Allocation.xlsx",
    "server_admin_passwords.txt",
]

# --- content forensics (CryptoDrop-style checks, sampled by the collector) --
CONTENT_SAMPLE_BYTES = 4096   # per-file sample size (keeps I/O negligible)
ENTROPY_RESAMPLE_SECS = 2.0   # min seconds between content samples per file

# Ransom-note heuristics (mirrors Elastic's "ransomware note" rules)
NOTE_NAME_KEYWORDS = ("readme", "recover", "restore", "decrypt", "ransom",
                      "how_to", "payment", "help_me")
NOTE_EXTENSIONS = (".txt", ".html", ".hta")
NOTE_CONTENT_KEYWORDS = ("encrypt", "decrypt", "bitcoin", "ransom",
                         "recover your files", "payment")

# --- alert delivery -----------------------------------------------------------
TOAST_ENABLED = True          # Windows balloon toast on CRITICAL/HIGH alerts
TOAST_MIN_SECS = 10           # rate limit between toasts
WEBHOOK_URL = None            # opt-in: POST alert summaries (e.g. Discord/Slack)

# --- watchdog noise we never treat as activity (editor/OS temp files) ------
IGNORED_PREFIXES = ("~$", ".")
IGNORED_SUFFIXES = (".tmp", ".crswap", ".swp")
IGNORED_EXACT = ("desktop.ini",)

# --- feature vector per window (order matters; persisted with the model) ---
# The shape features matter: a legitimate bulk copy is extreme in event_rate /
# files_touched / created, while attacks are extreme in existing_writes /
# extension_changes / deletions - dimensions where bulk stays low. Content
# forensics (entropy, type divergence) catches rewrites that rename nothing.
FEATURES = [
    "event_rate",        # events per minute in this window
    "files_touched",     # unique files seen in events
    "created",           # new files appearing (bulk copies, note-taking)
    "existing_writes",   # edits to files that already existed (edits, rewrites)
    "renames",           # moved events
    "extension_changes", # renames that changed the file extension
    "deletions",         # deleted events
    "canary_touched",    # 1 if any canary file appears in an event
    "entropy_max",       # peak content entropy sampled in this window (0-8)
    "type_divergence",   # writes whose content no longer matches the file type
]

# --- static rule thresholds per 10 s window --------------------------------
RULE_EVENT_RATE_PER_MIN = 240
RULE_DELETIONS = 5
RULE_EXTENSION_CHANGES = 3
RULE_RENAMES = 12

# Learned-normal-envelope policy: a feature flags ML when it exceeds the
# largest value seen in normal training, padded for headroom. Features that
# were ~0 throughout normal training get absolute floors instead.
ENVELOPE_PAD = 1.4
ENVELOPE_FLOORS = {
    "extension_changes": 3,
    "deletions": 5,
    "renames": 4,
    "existing_writes": 8,
    "canary_touched": 2,     # canary touches belong to the canary rule
    "entropy_max": 7.0,      # text ~2-5; compressed/encrypted content ~7.5-8
    "type_divergence": 2,    # benign writes never break their file signature
}

# Fallback "normal" profile used for explanations before any training
DEFAULT_BASELINE = {
    "event_rate": {"median": 20.0, "mad": 10.0},
    "files_touched": {"median": 3.0, "mad": 2.0},
    "created": {"median": 0.0, "mad": 0.0},
    "existing_writes": {"median": 3.0, "mad": 2.0},
    "renames": {"median": 0.0, "mad": 0.0},
    "extension_changes": {"median": 0.0, "mad": 0.0},
    "deletions": {"median": 0.0, "mad": 0.0},
    "canary_touched": {"median": 0.0, "mad": 0.0},
}

DB_SCHEMA_CREATE = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT,
    event_type TEXT NOT NULL,
    src_path TEXT,
    dest_path TEXT,
    entropy REAL,                   -- bits/byte of a content sample (modified events)
    divergent INTEGER,              -- 1 if content no longer matches its file type
    note_like INTEGER               -- 1 if the file looks like a ransom note
);
CREATE INDEX IF NOT EXISTS idx_events_id ON events(id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT,
    source TEXT NOT NULL,          -- rule | ml | fused
    rule_name TEXT,
    severity TEXT,                 -- CRITICAL | HIGH | MEDIUM
    ml_score REAL,                 -- 0-100 (0 for pure rule alerts)
    window_start REAL,
    window_end REAL,
    explanation TEXT,
    features_json TEXT
);
"""

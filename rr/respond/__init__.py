"""Response layer: protect originals, act proportionately, prove recovery.

Two pieces:

* :mod:`rr.respond.shadow` -- copy-on-write shadow journal. First-write-wins
  snapshots taken before anything escalates, restored with SHA-256
  verification so "we got the files back" is checkable.
* :mod:`rr.respond.ladder` -- graduated response ladder (alert, snapshot,
  suspend, quarantine, kill) with a dry-run default, a never-kill allowlist,
  a kill switch, TTL auto-revert and a hash-chained audit log.

Both are stdlib-only and safe to import anywhere: nothing touches the
filesystem at import time, and every mutating default is dry-run.
"""

from __future__ import annotations

from rr.respond.ladder import (
    ALWAYS_PROTECTED_PIDS,
    DEFAULT_NEVER_KILL,
    GENESIS_HASH,
    QUARANTINE_DIRNAME,
    QUARANTINE_MANIFEST,
    SEVERITY_MAX_TIER,
    ActionRecord,
    AuditVerification,
    FakeController,
    ProcessController,
    PsutilController,
    ResponseLadder,
    ResponsePolicy,
    SafeNoOpController,
    Verdict,
)
from rr.respond.shadow import (
    INDEX_FILENAME,
    INDEX_VERSION,
    RR_DIRNAME,
    SHADOW_DIRNAME,
    RestoreResult,
    ShadowJournal,
    SnapshotEntry,
    SnapshotOutcome,
    sha256_bytes,
    sha256_file,
)

__all__ = [
    "ALWAYS_PROTECTED_PIDS",
    "DEFAULT_NEVER_KILL",
    "GENESIS_HASH",
    "INDEX_FILENAME",
    "INDEX_VERSION",
    "QUARANTINE_DIRNAME",
    "QUARANTINE_MANIFEST",
    "RR_DIRNAME",
    "SEVERITY_MAX_TIER",
    "SHADOW_DIRNAME",
    "ActionRecord",
    "AuditVerification",
    "FakeController",
    "ProcessController",
    "PsutilController",
    "ResponseLadder",
    "ResponsePolicy",
    "RestoreResult",
    "SafeNoOpController",
    "ShadowJournal",
    "SnapshotEntry",
    "SnapshotOutcome",
    "Verdict",
    "sha256_bytes",
    "sha256_file",
]

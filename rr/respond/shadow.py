"""Copy-on-write shadow journal: the visible rollback story.

When the detector fires, the interesting question is not "was it right?" but
"can I get my files back?". This module answers that by snapshotting a file's
original bytes *before* the response ladder lets anything else happen, then
restoring them on demand and verifying the restore by SHA-256.

Design decisions that matter:

* **First-write-wins.** A path is snapshotted at most once per journal. If an
  encryptor rewrites the same file five times, re-snapshotting would overwrite
  the pristine copy with ciphertext -- the exact failure that would make
  rollback useless. :meth:`ShadowJournal.is_protected` gates every snapshot.
* **Cap refuses, never evicts.** When the storage cap is reached the journal
  *declines* new snapshots (status ``skipped_over_cap``) instead of evicting
  older ones. Evicting would silently discard the earliest -- and most likely
  pristine -- copies during an attack.
* **Dry-run by default.** ``dry_run=True`` is the constructor default, so
  importing and wiring this module cannot touch a user's disk by accident.
* **Root confinement.** Every path is resolved and checked against the journal
  root; anything outside is refused with ``refused_outside_root``.

The journal lives in ``<root>/.ransomradar/shadow/`` with a JSON index, so a
restore works in a fresh process after a crash.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "INDEX_FILENAME",
    "INDEX_VERSION",
    "RR_DIRNAME",
    "SHADOW_DIRNAME",
    "RestoreResult",
    "ShadowJournal",
    "SnapshotEntry",
    "SnapshotOutcome",
    "sha256_bytes",
    "sha256_file",
]

RR_DIRNAME = ".ransomradar"
SHADOW_DIRNAME = "shadow"
INDEX_FILENAME = "index.json"
INDEX_VERSION = 1

_MAX_KEY_LEN = 180
_HASHED_KEY_PREFIX = "h_"
_COPY_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks.

    Chunked reading keeps memory flat on large media files.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_COPY_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _encode_key(rel_path: str) -> str:
    """Encode an arena-relative path into a flat, filesystem-safe journal key.

    Long paths are hashed rather than truncated, because truncation could make
    two distinct files collide onto one snapshot -- silently losing one of them.
    """
    safe = rel_path.replace("\\", "/")
    encoded = safe.replace("%", "%25").replace("/", "%2F").replace(":", "%3A")
    if len(encoded) <= _MAX_KEY_LEN:
        return encoded
    return _HASHED_KEY_PREFIX + sha256_bytes(safe.encode("utf-8"))


def _decode_key(key: str) -> str | None:
    """Decode a journal key back to a relative path, or ``None`` if hashed."""
    if key.startswith(_HASHED_KEY_PREFIX):
        return None
    return key.replace("%2F", "/").replace("%3A", ":").replace("%25", "%")


@dataclass(frozen=True)
class SnapshotEntry:
    """One protected file in the journal.

    Attributes:
        rel_path: Path relative to the journal root (POSIX separators).
        key: Flat journal key used as the stored filename.
        sha256: Digest of the *original* bytes.
        size: Original size in bytes.
        ts: Epoch seconds when the snapshot was taken.
    """

    rel_path: str
    key: str
    sha256: str
    size: int
    ts: float

    def to_dict(self) -> dict[str, object]:
        return {
            "rel_path": self.rel_path,
            "key": self.key,
            "sha256": self.sha256,
            "size": self.size,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "SnapshotEntry":
        return cls(
            rel_path=str(data["rel_path"]),
            key=str(data["key"]),
            sha256=str(data["sha256"]),
            size=int(data["size"]),  # type: ignore[arg-type]
            ts=float(data["ts"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SnapshotOutcome:
    """Result of attempting to snapshot one path.

    ``status`` is one of ``snapshotted``, ``already_protected``,
    ``skipped_over_cap``, ``missing``, ``refused_outside_root``, ``simulated``
    or ``failed``. Every outcome is explicit: the journal never silently drops a
    file, because a silent drop is indistinguishable from successful protection
    until the moment you need to restore.
    """

    path: str
    status: str
    reason: str = ""
    entry: SnapshotEntry | None = None

    @property
    def ok(self) -> bool:
        """Whether the path ends up protected (or would be, in dry-run)."""
        return self.status in {"snapshotted", "already_protected", "simulated"}


@dataclass(frozen=True)
class RestoreResult:
    """Outcome of a restore operation.

    Attributes:
        restored: Relative paths successfully written back.
        failed: Mapping of relative path -> failure reason.
        skipped: Relative paths deliberately not restored.
        bytes_restored: Total bytes written back.
        sha256_by_path: Digest of each restored file, re-read from disk.
        verified_identical: True only if every restored file's re-read digest
            matches the recorded original digest. This is the claim that makes
            "we recovered the files" checkable rather than asserted.
        dry_run: Whether this was a simulation.
    """

    restored: tuple[str, ...] = ()
    failed: dict[str, str] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()
    bytes_restored: int = 0
    sha256_by_path: dict[str, str] = field(default_factory=dict)
    verified_identical: bool = False
    dry_run: bool = True

    @property
    def n_restored(self) -> int:
        return len(self.restored)


class ShadowJournal:
    """Copy-on-write snapshot store rooted at a protected directory.

    Args:
        root: Directory whose files may be protected. Snapshots live under
            ``<root>/.ransomradar/shadow/``.
        cap_bytes: Maximum total snapshot bytes. On reaching the cap the journal
            refuses new snapshots rather than evicting old ones.
        dry_run: When True (the default) nothing is written to disk; every
            snapshot returns ``simulated``. Defaulting to dry-run means wiring
            this into a pipeline cannot damage a real filesystem by accident.

    Raises:
        ValueError: If ``cap_bytes`` is not positive.
    """

    def __init__(
        self,
        root: str | Path,
        cap_bytes: int = 512 * 1024 * 1024,
        *,
        dry_run: bool = True,
    ) -> None:
        if cap_bytes <= 0:
            raise ValueError(f"cap_bytes must be > 0, got {cap_bytes!r}")
        self.root = Path(root).expanduser().resolve()
        self.cap_bytes = int(cap_bytes)
        self.dry_run = bool(dry_run)
        self._entries: dict[str, SnapshotEntry] = {}
        self._skipped_over_cap: list[str] = []
        if not self.dry_run:
            self.shadow_dir.mkdir(parents=True, exist_ok=True)
        self.reload()

    @property
    def shadow_dir(self) -> Path:
        """Directory holding the snapshot blobs and index."""
        return self.root / RR_DIRNAME / SHADOW_DIRNAME

    @property
    def index_path(self) -> Path:
        """Path of the JSON index."""
        return self.shadow_dir / INDEX_FILENAME

    @property
    def skipped_over_cap(self) -> tuple[str, ...]:
        """Paths refused because the cap was reached."""
        return tuple(self._skipped_over_cap)

    def _relative(self, path: str | Path) -> str | None:
        """Resolve ``path`` and return its root-relative form, or None if outside.

        ``Path.resolve()`` is applied first so symlinks and ``..`` cannot be
        used to walk outside the protected root.
        """
        try:
            resolved = Path(path).expanduser().resolve()
        except OSError:
            return None
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            return None

    def is_protected(self, path: str | Path) -> bool:
        """Whether a pristine snapshot of ``path`` already exists."""
        rel = self._relative(path)
        return rel is not None and rel in self._entries

    def entries(self) -> tuple[SnapshotEntry, ...]:
        """All snapshot entries, ordered by capture time."""
        return tuple(sorted(self._entries.values(), key=lambda e: e.ts))

    def usage_bytes(self) -> int:
        """Total original bytes currently protected."""
        return sum(entry.size for entry in self._entries.values())

    def stats(self) -> dict[str, object]:
        """Summary counters for dashboards and the demo."""
        return {
            "n_protected": len(self._entries),
            "usage_bytes": self.usage_bytes(),
            "cap_bytes": self.cap_bytes,
            "cap_fraction": self.usage_bytes() / self.cap_bytes,
            "n_skipped_over_cap": len(self._skipped_over_cap),
            "dry_run": self.dry_run,
        }

    def reload(self) -> None:
        """Re-read the on-disk index, so a restore works after a crash."""
        self._entries = {}
        if self.dry_run or not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if int(raw.get("version", 0)) != INDEX_VERSION:
            return
        for item in raw.get("entries", []):
            try:
                entry = SnapshotEntry.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue
            self._entries[entry.rel_path] = entry

    def _persist(self) -> None:
        """Atomically write the index via a temp file + replace."""
        if self.dry_run:
            return
        self.shadow_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "root": str(self.root),
            "entries": [e.to_dict() for e in self.entries()],
        }
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def snapshot(self, path: str | Path) -> SnapshotOutcome:
        """Snapshot one file's original bytes, first-write-wins.

        Args:
            path: File to protect.

        Returns:
            A :class:`SnapshotOutcome` explaining exactly what happened.
        """
        text = str(path)
        rel = self._relative(path)
        if rel is None:
            return SnapshotOutcome(
                text,
                "refused_outside_root",
                f"resolved path is outside the protected root {self.root}",
            )
        if rel in self._entries:
            # First-write-wins: never overwrite a pristine copy with ciphertext.
            return SnapshotOutcome(
                text, "already_protected", "pristine copy already held",
                self._entries[rel],
            )
        source = self.root / rel
        if not source.is_file():
            return SnapshotOutcome(text, "missing", "not an existing regular file")
        try:
            size = source.stat().st_size
        except OSError as exc:
            return SnapshotOutcome(text, "failed", f"stat failed: {exc}")
        if self.usage_bytes() + size > self.cap_bytes:
            # Refuse, never evict.
            self._skipped_over_cap.append(rel)
            return SnapshotOutcome(
                text,
                "skipped_over_cap",
                f"cap {self.cap_bytes} bytes reached; refusing rather than evicting",
            )
        key = _encode_key(rel)
        if self.dry_run:
            return SnapshotOutcome(text, "simulated", "dry_run=True; nothing written")
        try:
            digest = sha256_file(source)
            self.shadow_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, self.shadow_dir / key)
        except OSError as exc:
            return SnapshotOutcome(text, "failed", f"copy failed: {exc}")
        entry = SnapshotEntry(
            rel_path=rel, key=key, sha256=digest, size=int(size), ts=time.time()
        )
        self._entries[rel] = entry
        self._persist()
        return SnapshotOutcome(text, "snapshotted", "", entry)

    def snapshot_many(self, paths: Iterable[str | Path]) -> tuple[SnapshotOutcome, ...]:
        """Snapshot several paths, returning one outcome each."""
        return tuple(self.snapshot(p) for p in paths)

    def plan_restore(
        self, paths: Iterable[str | Path] | None = None
    ) -> tuple[SnapshotEntry, ...]:
        """Return the entries a restore would write back, without touching disk.

        Args:
            paths: Optional subset to restore; ``None`` means everything held.

        Returns:
            Entries in capture order. Unknown or out-of-root paths are omitted.
        """
        if paths is None:
            return self.entries()
        wanted: list[SnapshotEntry] = []
        for path in paths:
            rel = self._relative(path)
            if rel is not None and rel in self._entries:
                wanted.append(self._entries[rel])
        return tuple(sorted(wanted, key=lambda e: e.ts))

    def restore(
        self, paths: Iterable[str | Path] | None = None, *, verify: bool = True
    ) -> RestoreResult:
        """Write pristine bytes back over the current files.

        Each restored file is re-read from disk and re-hashed, so
        ``verified_identical`` is evidence rather than an assumption. That is
        what lets the demo claim recovery and then *prove* it on screen.

        Args:
            paths: Optional subset to restore; ``None`` restores everything.
            verify: Re-hash each restored file and compare to the original.

        Returns:
            A :class:`RestoreResult`.
        """
        planned = self.plan_restore(paths)
        if self.dry_run:
            return RestoreResult(
                restored=tuple(e.rel_path for e in planned),
                skipped=(),
                bytes_restored=sum(e.size for e in planned),
                verified_identical=False,
                dry_run=True,
            )

        restored: list[str] = []
        failed: dict[str, str] = {}
        digests: dict[str, str] = {}
        total = 0
        all_match = True
        for entry in planned:
            blob = self.shadow_dir / entry.key
            target = self.root / entry.rel_path
            if not blob.is_file():
                failed[entry.rel_path] = "snapshot blob missing from journal"
                all_match = False
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(blob, target)
            except OSError as exc:
                failed[entry.rel_path] = f"restore write failed: {exc}"
                all_match = False
                continue
            restored.append(entry.rel_path)
            total += entry.size
            if verify:
                try:
                    actual = sha256_file(target)
                except OSError as exc:
                    failed[entry.rel_path] = f"verify read failed: {exc}"
                    all_match = False
                    continue
                digests[entry.rel_path] = actual
                if actual != entry.sha256:
                    failed[entry.rel_path] = (
                        "restored bytes do not match the recorded original digest"
                    )
                    all_match = False
        return RestoreResult(
            restored=tuple(restored),
            failed=failed,
            skipped=(),
            bytes_restored=total,
            sha256_by_path=digests,
            verified_identical=bool(verify) and all_match and not failed,
            dry_run=False,
        )

    def restore_all(self, *, verify: bool = True) -> RestoreResult:
        """Restore every protected file."""
        return self.restore(None, verify=verify)


if __name__ == "__main__":  # pragma: no cover - manual smoke demo
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "protected"
        (root / "docs").mkdir(parents=True)
        victim = root / "docs" / "report.txt"
        victim.write_text("original contents\n" * 20, encoding="utf-8")
        original_digest = sha256_file(victim)

        journal = ShadowJournal(root, dry_run=False)
        print("snapshot :", journal.snapshot(victim).status)
        print("again    :", journal.snapshot(victim).status, "(first-write-wins)")

        victim.write_bytes(os.urandom(4096))  # simulate encryption
        print("tampered :", sha256_file(victim) != original_digest)

        result = journal.restore_all()
        print(
            f"restored={result.n_restored} "
            f"verified_identical={result.verified_identical} "
            f"bytes={result.bytes_restored}"
        )
        print("digest match:", sha256_file(victim) == original_digest)
        print("stats:", journal.stats())

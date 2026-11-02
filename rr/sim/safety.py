"""rr.sim.safety — arena confinement and path-escape guards.

Defensive research artefact (NOT ransomware). This module is the single choke
point through which every filesystem-mutating helper in ``rr.sim`` routes, so
that all writes, renames and deletes are *provably* confined to a
caller-supplied *arena* directory. The arena must:

* exist and be a directory,
* not be a system path or a user home root,
* contain a sentinel marker file (``.ransomradar_arena``) created by
  :func:`create_arena`, so the tooling can never be aimed at real data.

Safety invariants enforced here:

* :func:`require_arena` refuses non-sentinel / forbidden roots.
* :func:`resolve_within` calls ``Path.resolve()`` (which also resolves
  symlinks and ``..``) and raises :class:`PathEscapeError` if the resolved
  target is not inside the resolved arena root.

No I/O or network happens at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

SENTINEL_NAME = ".ransomradar_arena"
JOURNAL_NAME = ".ransomradar_journal.json"
KEY_NAME = ".ransomradar_key.json"
CANARY_REGISTRY_NAME = ".ransomradar_canaries.json"
NOTE_BASENAME = "_RANSOMRADAR_RECOVERY_NOTE.txt"

#: Control files that are never treated as user data (never hashed, targeted or renamed).
RESERVED_NAMES = frozenset(
    {SENTINEL_NAME, JOURNAL_NAME, KEY_NAME, CANARY_REGISTRY_NAME, NOTE_BASENAME}
)


class ArenaError(RuntimeError):
    """Raised when a root is not a valid, sentinel-marked arena."""


class PathEscapeError(ArenaError):
    """Raised when a target path resolves outside the arena root."""


def _forbidden_roots() -> set[Path]:
    """Absolute paths that must never be used as an arena root."""
    candidates = [
        "/",
        "/root",
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/boot",
        "/dev",
        "/proc",
        "/sys",
        "/var",
        "/opt",
        "/home",
        "/Users",
        "/System",
        "/Applications",
    ]
    out: set[Path] = set()
    for c in candidates:
        try:
            out.add(Path(c).resolve())
        except OSError:
            continue
    try:
        out.add(Path.home().resolve())
    except (RuntimeError, OSError):
        pass
    return out


def create_arena(root: os.PathLike[str] | str) -> Path:
    """Create (if needed) an arena directory and write its sentinel marker.

    Returns the resolved arena path. This is the ONLY supported way to make a
    directory usable by the rest of ``rr.sim``.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    resolved = root_path.resolve()
    if resolved in _forbidden_roots():
        raise ArenaError(f"refusing to create an arena at a protected path: {resolved}")
    sentinel = resolved / SENTINEL_NAME
    sentinel.write_text(
        "RansomRadar defensive-research arena.\n"
        "This directory is a disposable test sandbox. Safe to delete.\n",
        encoding="utf-8",
    )
    return resolved


def require_arena(root: os.PathLike[str] | str) -> Path:
    """Validate that ``root`` is a legitimate arena and return its resolved path.

    Raises :class:`ArenaError` if the root does not exist, is not a directory,
    is a forbidden/system/home path, or is missing the sentinel marker.
    """
    resolved = Path(root).resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ArenaError(f"arena root does not exist or is not a directory: {resolved}")
    if resolved in _forbidden_roots():
        raise ArenaError(f"refusing to operate on a protected system/home path: {resolved}")
    if resolved == Path(resolved.anchor).resolve():
        raise ArenaError(f"refusing to operate on a filesystem root anchor: {resolved}")
    if not (resolved / SENTINEL_NAME).is_file():
        raise ArenaError(
            f"missing sentinel marker '{SENTINEL_NAME}' in {resolved}; "
            "use rr.sim.safety.create_arena() to initialise a sandbox first."
        )
    return resolved


def resolve_within(root_resolved: Path, target: os.PathLike[str] | str) -> Path:
    """Resolve ``target`` (relative to the arena when not absolute) and ensure
    it stays inside ``root_resolved``.

    ``Path.resolve()`` collapses ``..`` and follows symlinks, so both escape
    vectors are caught. Raises :class:`PathEscapeError` on any escape.
    """
    t = Path(target)
    if not t.is_absolute():
        t = root_resolved / t
    resolved = t.resolve()
    if resolved != root_resolved and not resolved.is_relative_to(root_resolved):
        raise PathEscapeError(
            f"path escapes arena: {resolved!s} is not inside {root_resolved!s}"
        )
    return resolved


def is_reserved(path: os.PathLike[str] | str) -> bool:
    """True if the basename is a reserved control/note file managed by rr.sim."""
    return Path(path).name in RESERVED_NAMES


def iter_arena_files(root_resolved: Path) -> Iterator[Path]:
    """Yield regular data files inside the arena in a stable, sorted order.

    Reserved control files and out-of-arena symlinks are skipped. The order is
    deterministic (directories and filenames sorted) so callers can rely on it.
    """
    for dirpath, dirnames, filenames in os.walk(root_resolved):
        dirnames.sort()
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if is_reserved(p):
                continue
            if p.is_symlink():
                try:
                    resolve_within(root_resolved, p)
                except PathEscapeError:
                    continue
            if p.is_file():
                yield p

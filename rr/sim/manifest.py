"""rr.sim.manifest — file-level ground truth for RansomRadar.

Defensive research artefact (NOT ransomware). This module turns "N files
saved" from an *assertion* into a *measurement*: it records the SHA-256 hash
and size of every data file in an arena, and later verifies which files are
byte-identical, modified, missing or newly added. This is what proves that a
restored tree is byte-for-byte identical to the original.

No network access. Files are read in chunks; unreadable files are skipped
gracefully. All operations are confined to a sentinel-marked arena via
``rr.sim.safety``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .safety import require_arena, iter_arena_files

_CHUNK = 1 << 20  # 1 MiB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ManifestData:
    """Full ground-truth snapshot: hash + size per relative POSIX path."""

    hashes: dict[str, str] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {"version": 1, "hashes": self.hashes, "sizes": self.sizes}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ManifestData":
        hashes = dict(data.get("hashes", {}))  # type: ignore[arg-type]
        sizes = {k: int(v) for k, v in dict(data.get("sizes", {})).items()}  # type: ignore[arg-type]
        return cls(hashes=hashes, sizes=sizes)

    def save(self, path: os.PathLike[str] | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "ManifestData":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class VerifyResult:
    """Outcome of comparing an arena against a manifest.

    ``bytes_at_risk`` is the total size (from the manifest when available,
    else the current on-disk size for modified files) of all modified and
    missing files — i.e. the bytes that a real attacker would have damaged.
    """

    intact_paths: list[str]
    modified: list[str]
    missing: list[str]
    added: list[str]
    bytes_at_risk: int

    @property
    def intact(self) -> bool:
        """True only when the arena is byte-for-byte identical to the manifest:
        nothing modified, nothing missing, and nothing added."""
        return not (self.modified or self.missing or self.added)

    @property
    def n_intact(self) -> int:
        return len(self.intact_paths)

    @property
    def n_modified(self) -> int:
        return len(self.modified)

    @property
    def n_missing(self) -> int:
        return len(self.missing)

    @property
    def n_added(self) -> int:
        return len(self.added)

    @property
    def all_recovered(self) -> bool:
        """True when nothing is modified or missing (a clean restore)."""
        return not self.modified and not self.missing


def build_manifest_data(root: os.PathLike[str] | str) -> ManifestData:
    """Walk the arena and return a :class:`ManifestData` (hashes + sizes)."""
    root_resolved = require_arena(root)
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for p in iter_arena_files(root_resolved):
        rel = p.relative_to(root_resolved).as_posix()
        try:
            sizes[rel] = p.stat().st_size
            hashes[rel] = _sha256(p)
        except (OSError, PermissionError):
            # Unreadable file: skip gracefully, do not abort the whole manifest.
            continue
    return ManifestData(
        hashes=dict(sorted(hashes.items())),
        sizes=dict(sorted(sizes.items())),
    )


def build_manifest(root: os.PathLike[str] | str) -> dict[str, str]:
    """Frozen-contract entry point: return ``{relative_posix_path: sha256_hex}``.

    Sizes are available in a documented way via :func:`build_manifest_data`
    (the :class:`ManifestData` dataclass). This function returns only the hash
    map to satisfy the integration contract.
    """
    return build_manifest_data(root).hashes


def _coerce(manifest: "ManifestData | Mapping[str, str]") -> tuple[dict[str, str], dict[str, int]]:
    if isinstance(manifest, ManifestData):
        return dict(manifest.hashes), dict(manifest.sizes)
    return dict(manifest), {}


def verify_manifest(
    root: os.PathLike[str] | str,
    manifest: "ManifestData | Mapping[str, str]",
) -> VerifyResult:
    """Compare the current arena state against ``manifest``.

    Accepts either a :class:`ManifestData` (preferred, so ``bytes_at_risk``
    uses the original sizes) or a plain ``{path: sha256}`` mapping.
    """
    root_resolved = require_arena(root)
    exp_hashes, exp_sizes = _coerce(manifest)

    cur_hashes: dict[str, str] = {}
    cur_sizes: dict[str, int] = {}
    for p in iter_arena_files(root_resolved):
        rel = p.relative_to(root_resolved).as_posix()
        try:
            cur_sizes[rel] = p.stat().st_size
            cur_hashes[rel] = _sha256(p)
        except (OSError, PermissionError):
            continue

    intact: list[str] = []
    modified: list[str] = []
    missing: list[str] = []
    for rel, expected in exp_hashes.items():
        if rel not in cur_hashes:
            missing.append(rel)
        elif cur_hashes[rel] == expected:
            intact.append(rel)
        else:
            modified.append(rel)
    added = [rel for rel in cur_hashes if rel not in exp_hashes]

    bytes_at_risk = 0
    for rel in modified:
        bytes_at_risk += exp_sizes.get(rel, cur_sizes.get(rel, 0))
    for rel in missing:
        bytes_at_risk += exp_sizes.get(rel, 0)

    return VerifyResult(
        intact_paths=sorted(intact),
        modified=sorted(modified),
        missing=sorted(missing),
        added=sorted(added),
        bytes_at_risk=bytes_at_risk,
    )

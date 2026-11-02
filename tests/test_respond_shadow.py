"""Tests for the copy-on-write shadow journal.

These tests exist to defend the three properties that make rollback
trustworthy: first-write-wins, cap-refuses-never-evicts, and
verified-identical restores.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rr.respond.shadow import ShadowJournal, sha256_bytes, sha256_file


def _arena(tmp_path: Path, name: str = "report.txt", payload: str = "original\n") -> tuple[Path, Path]:
    root = tmp_path / "protected"
    (root / "docs").mkdir(parents=True)
    victim = root / "docs" / name
    victim.write_text(payload * 10, encoding="utf-8")
    return root, victim


def test_dry_run_is_the_default_and_writes_nothing(tmp_path: Path) -> None:
    root, victim = _arena(tmp_path)
    journal = ShadowJournal(root)
    assert journal.dry_run is True
    outcome = journal.snapshot(victim)
    assert outcome.status == "simulated"
    assert outcome.ok is True
    assert not journal.shadow_dir.exists()


def test_snapshot_then_restore_recovers_exact_bytes(tmp_path: Path) -> None:
    root, victim = _arena(tmp_path)
    original = sha256_file(victim)
    journal = ShadowJournal(root, dry_run=False)

    assert journal.snapshot(victim).status == "snapshotted"
    victim.write_bytes(os.urandom(2048))
    assert sha256_file(victim) != original

    result = journal.restore_all()
    assert result.n_restored == 1
    assert result.verified_identical is True
    assert not result.failed
    assert sha256_file(victim) == original


def test_first_write_wins_never_overwrites_pristine_copy(tmp_path: Path) -> None:
    """Re-snapshotting must not capture ciphertext over the clean original."""
    root, victim = _arena(tmp_path)
    original = sha256_file(victim)
    journal = ShadowJournal(root, dry_run=False)
    journal.snapshot(victim)

    victim.write_bytes(b"ENCRYPTED" * 100)
    second = journal.snapshot(victim)
    assert second.status == "already_protected"

    journal.restore_all()
    assert sha256_file(victim) == original


def test_cap_refuses_rather_than_evicting(tmp_path: Path) -> None:
    root, first = _arena(tmp_path, "a.txt", "aaaa\n")
    second = root / "docs" / "b.txt"
    second.write_text("bbbb\n" * 10, encoding="utf-8")

    journal = ShadowJournal(root, cap_bytes=first.stat().st_size, dry_run=False)
    assert journal.snapshot(first).status == "snapshotted"
    refused = journal.snapshot(second)
    assert refused.status == "skipped_over_cap"
    assert refused.ok is False
    # The earlier, pristine copy survives.
    assert journal.is_protected(first)
    assert journal.skipped_over_cap == ("docs/b.txt",)


def test_paths_outside_root_are_refused(tmp_path: Path) -> None:
    root, _ = _arena(tmp_path)
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("not yours\n", encoding="utf-8")
    journal = ShadowJournal(root, dry_run=False)
    outcome = journal.snapshot(outside)
    assert outcome.status == "refused_outside_root"


def test_missing_file_is_reported_not_raised(tmp_path: Path) -> None:
    root, _ = _arena(tmp_path)
    journal = ShadowJournal(root, dry_run=False)
    assert journal.snapshot(root / "docs" / "nope.txt").status == "missing"


def test_index_survives_a_fresh_process(tmp_path: Path) -> None:
    """A crash must not lose the ability to restore."""
    root, victim = _arena(tmp_path)
    original = sha256_file(victim)
    ShadowJournal(root, dry_run=False).snapshot(victim)

    victim.write_bytes(os.urandom(1024))
    reopened = ShadowJournal(root, dry_run=False)
    assert reopened.is_protected(victim)
    assert reopened.restore_all().verified_identical is True
    assert sha256_file(victim) == original


def test_long_paths_are_hashed_not_truncated(tmp_path: Path) -> None:
    """Truncation could collide two files onto one snapshot."""
    root = tmp_path / "protected"
    deep = root / ("d" * 60) / ("e" * 60) / ("f" * 60)
    try:
        deep.mkdir(parents=True)
    except OSError:
        pytest.skip(
            "platform refuses paths over MAX_PATH (Windows LongPathsEnabled "
            "off) - the tmp prefix itself exceeds the limit; the key-hashing "
            "behaviour is platform-independent and passes on Linux."
        )
    victim = deep / "long_name_file.txt"
    victim.write_text("payload\n", encoding="utf-8")

    journal = ShadowJournal(root, dry_run=False)
    entry = journal.snapshot(victim).entry
    assert entry is not None
    assert entry.key.startswith("h_")
    assert journal.restore_all().verified_identical is True


def test_sha256_bytes_matches_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    data = b"some bytes" * 500
    path.write_bytes(data)
    assert sha256_bytes(data) == sha256_file(path)


def test_stats_reports_usage_and_cap(tmp_path: Path) -> None:
    root, victim = _arena(tmp_path)
    journal = ShadowJournal(root, dry_run=False)
    journal.snapshot(victim)
    stats = journal.stats()
    assert stats["n_protected"] == 1
    assert stats["usage_bytes"] == victim.stat().st_size
    assert stats["dry_run"] is False

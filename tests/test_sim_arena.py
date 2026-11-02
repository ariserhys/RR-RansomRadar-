"""Tests for the attack arena: safety rails, manifests, canaries, evader.

The evader is the most dangerous code in the project, so the highest-value
tests here are the confinement rails and the proof that every transformation is
byte-for-byte reversible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rr.sim.canaries import CANARY_MARKERS, CANARY_STRATEGIES, place_canaries
from rr.sim.evader import EvaderConfig, run_evader, undo_all
from rr.sim.manifest import build_manifest, verify_manifest
from rr.sim.safety import (
    SENTINEL_NAME,
    ArenaError,
    PathEscapeError,
    create_arena,
    require_arena,
    resolve_within,
)


def _arena(tmp_path: Path, n_files: int = 12) -> Path:
    root = create_arena(tmp_path / "arena")
    for index in range(n_files):
        sub = root / ("docs" if index % 2 else "projects")
        sub.mkdir(parents=True, exist_ok=True)
        (sub / f"file_{index:02d}.txt").write_text(
            f"contents of file {index}\n" * 20, encoding="utf-8"
        )
    return root


def test_create_arena_writes_a_sentinel(tmp_path: Path) -> None:
    root = create_arena(tmp_path / "arena")
    assert (root / SENTINEL_NAME).exists()
    assert require_arena(root) == root.resolve()


def test_require_arena_refuses_a_directory_without_a_sentinel(tmp_path: Path) -> None:
    """The sentinel is what stops the evader running in a real home directory."""
    plain = tmp_path / "not_an_arena"
    plain.mkdir()
    with pytest.raises(ArenaError):
        require_arena(plain)


def test_resolve_within_blocks_traversal(tmp_path: Path) -> None:
    root = create_arena(tmp_path / "arena")
    with pytest.raises(PathEscapeError):
        resolve_within(root, "../../etc/passwd")


def test_manifest_detects_modification(tmp_path: Path) -> None:
    root = _arena(tmp_path, 4)
    manifest = build_manifest(root)
    victim = next(root.rglob("file_*.txt"))
    victim.write_bytes(b"tampered")

    result = verify_manifest(root, manifest)
    assert result.intact is False
    assert result.n_modified == 1
    assert result.bytes_at_risk > 0


def test_manifest_of_untouched_arena_is_intact(tmp_path: Path) -> None:
    root = _arena(tmp_path, 4)
    manifest = build_manifest(root)
    assert verify_manifest(root, manifest).intact is True


@pytest.mark.parametrize("strategy", CANARY_STRATEGIES)
def test_every_canary_path_contains_a_marker(strategy: str) -> None:
    """The featurizer detects canaries by path substring, so the placement
    strategy must guarantee a marker is present or detection silently fails."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = _arena(Path(tmp), 6)
        placement = place_canaries(root, strategy=strategy)
        for path in placement.paths:
            lowered = str(path).lower()
            assert any(marker in lowered for marker in CANARY_MARKERS)


def test_none_strategy_places_nothing(tmp_path: Path) -> None:
    root = _arena(tmp_path, 4)
    assert place_canaries(root, strategy="none").n_placed == 0


def test_canary_placement_is_deterministic(tmp_path: Path) -> None:
    root_a = _arena(tmp_path / "a", 6)
    root_b = _arena(tmp_path / "b", 6)
    names_a = sorted(Path(p).name for p in place_canaries(root_a, seed=7).paths)
    names_b = sorted(Path(p).name for p in place_canaries(root_b, seed=7).paths)
    assert names_a == names_b


def test_evader_refuses_a_directory_that_is_not_an_arena(tmp_path: Path) -> None:
    plain = tmp_path / "real_documents"
    plain.mkdir()
    (plain / "thesis.docx").write_text("precious\n", encoding="utf-8")
    with pytest.raises(ArenaError):
        run_evader(plain, EvaderConfig(rate_hz=1000.0))


def test_evader_is_byte_for_byte_reversible(tmp_path: Path) -> None:
    """This is what makes the adversarial experiments safe to run repeatedly."""
    root = _arena(tmp_path, 10)
    manifest = build_manifest(root)
    config = EvaderConfig(rate_hz=5000.0, encrypt_fraction=1.0, seed=3)

    trace = run_evader(root, config)
    assert trace.n_damaged_before(trace.end_ts + 1.0) > 0
    assert verify_manifest(root, manifest).intact is False

    undo_all(root, config)
    assert verify_manifest(root, manifest).intact is True


def test_partial_encryption_touches_fewer_files(tmp_path: Path) -> None:
    """Intermittent encryption is the headline evasion; it must actually reduce
    the damage footprint, which is what collapses entropy-only detection."""
    full_root = _arena(tmp_path / "full", 20)
    part_root = _arena(tmp_path / "part", 20)
    full = run_evader(full_root, EvaderConfig(rate_hz=5000.0, encrypt_fraction=1.0, seed=1))
    part = run_evader(part_root, EvaderConfig(rate_hz=5000.0, encrypt_fraction=0.15, seed=1))
    full_bytes = sum(op.bytes_changed for op in full.operations)
    part_bytes = sum(op.bytes_changed for op in part.operations)
    assert part_bytes < full_bytes


def test_decoy_avoidance_skips_canary_files(tmp_path: Path) -> None:
    root = _arena(tmp_path, 10)
    placement = place_canaries(root, strategy="per_depth")
    assert placement.n_placed > 0

    trace = run_evader(
        root, EvaderConfig(rate_hz=5000.0, decoy_avoidance=True, seed=2)
    )
    touched = {str(op.path).lower() for op in trace.operations}
    for path in placement.paths:
        assert str(path).lower() not in touched


def test_trace_dataframe_exposes_the_columns_the_metrics_need(tmp_path: Path) -> None:
    """rr.eval.metrics reads bytes_changed/op/path/ts from this frame."""
    root = _arena(tmp_path, 6)
    trace = run_evader(root, EvaderConfig(rate_hz=5000.0, seed=4))
    frame = trace.to_dataframe()
    for column in ("ts", "op", "path", "bytes_changed"):
        assert column in frame.columns
    assert len(frame) == len(trace.operations)


def test_evader_config_rejects_incoherent_settings() -> None:
    with pytest.raises(ValueError):
        EvaderConfig(rate_hz=0.0)
    with pytest.raises(ValueError):
        EvaderConfig(encrypt_fraction=1.5)
    with pytest.raises(ValueError):
        EvaderConfig(in_place=True, rename_policy="extension_append")


def test_same_seed_gives_the_same_attack(tmp_path: Path) -> None:
    root_a = _arena(tmp_path / "a", 8)
    root_b = _arena(tmp_path / "b", 8)
    config = EvaderConfig(rate_hz=5000.0, seed=11)
    first = run_evader(root_a, config)
    second = run_evader(root_b, config)
    assert first.signature == second.signature

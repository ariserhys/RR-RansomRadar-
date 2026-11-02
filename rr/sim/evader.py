"""rr.sim.evader — the adversary we built to try to defeat our own detector.

This is defensive research code, NOT ransomware: it runs only inside a
sentinel-marked arena (``rr.sim.safety``), it throttles itself, and every
transformation it applies is *byte-for-byte reversible* — the "encryption" is
XOR with a seed-derived keystream, so applying it again restores the original
bytes. ``undo_all`` replays the recorded operation journal backwards and then
deletes the journal, returning the arena to exactly its pre-attack state.

Evasion knobs (each maps to a real ransomware behaviour):

* ``rate_hz``            — throttled "low-and-slow" operation rate.
* ``encrypt_fraction``   — intermittent encryption (LockFile/BlackCat/Play/
  Qilin-style partial encryption); 0.15 is the headline setting.
* ``in_place``           — rewrite in place instead of renaming.
* ``rename_policy``      - ``"extension_append"`` (``file.txt -> file.txt.locked``).
* ``note_timing``        - when the ransom note is dropped.
* ``decoy_avoidance``    - fingerprint-and-skip canary paths (attackers do this).
* ``n_processes``        - shards traversal across workers; execution stays
  sequential and fully deterministic per seed (research compromise, documented).

Determinism: the same ``seed`` + config produces the same operation sequence
(same ``AttackTrace.signature``) on any machine and any arena with the same
tree shape. Timing (wall-clock ``ts``) is not part of the signature.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from rr.core.featurize import CANARY_MARKERS
from rr.sim.safety import (
    JOURNAL_NAME,
    CANARY_REGISTRY_NAME,
    KEY_NAME,
    NOTE_BASENAME,
    SENTINEL_NAME,
    ArenaError,
    require_arena,
    resolve_within,
)

__all__ = ["AttackOp", "AttackTrace", "EvaderConfig", "run_evader", "undo_all"]

_JOURNAL_RELPATH = ".ransomradar/evader_journal.json"
_RENAME_SUFFIX = ".locked"


@dataclass(frozen=True)
class EvaderConfig:
    """Configuration of a simulated attacker. Incoherent settings raise."""

    rate_hz: float = 1000.0
    sleep_jitter: float = 0.0
    encrypt_fraction: float = 1.0
    in_place: bool = False
    rename_policy: Literal["none", "extension_append"] = "extension_append"
    note_timing: Literal["start", "end", "none"] = "end"
    decoy_avoidance: bool = False
    n_processes: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be > 0 (the attacker has to act).")
        if not 0.0 < self.encrypt_fraction <= 1.0:
            raise ValueError("encrypt_fraction must be in (0.0, 1.0].")
        if not 0.0 <= self.sleep_jitter <= 1.0:
            raise ValueError("sleep_jitter must be in [0.0, 1.0].")
        if self.in_place and self.rename_policy == "extension_append":
            raise ValueError(
                "in_place=True is incompatible with rename_policy="
                "'extension_append' (renaming contradicts staying in place); "
                "use rename_policy='none'."
            )
        if self.n_processes < 1:
            raise ValueError("n_processes must be >= 1.")

    def signature_payload(self) -> dict:
        return {
            "rate_hz": self.rate_hz,
            "sleep_jitter": self.sleep_jitter,
            "encrypt_fraction": self.encrypt_fraction,
            "in_place": self.in_place,
            "rename_policy": self.rename_policy,
            "note_timing": self.note_timing,
            "decoy_avoidance": self.decoy_avoidance,
            "n_processes": self.n_processes,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class AttackOp:
    """One operation the evader performed (or would perform)."""

    ts: float
    op: str                      # "encrypt" | "rename" | "note"
    path: Path
    dest_path: Path | None = None
    bytes_changed: int = 0


@dataclass(frozen=True)
class AttackTrace:
    """What the evader did, and when — the input to ``rr.eval.metrics``."""

    operations: tuple[AttackOp, ...]
    start_ts: float
    end_ts: float
    signature: str
    config: EvaderConfig = field(repr=False)

    def n_damaged_before(self, ts: float) -> int:
        """Distinct files damaged (encrypted) at or before ``ts``."""
        return len({op.path for op in self.operations
                    if op.op == "encrypt" and op.ts <= ts})

    def to_dataframe(self) -> pd.DataFrame:
        """Columns: ``ts, op, path, dest_path, bytes_changed`` (rr.eval contract)."""
        rows = [{
            "ts": op.ts,
            "op": op.op,
            "path": str(op.path),
            "dest_path": str(op.dest_path) if op.dest_path else None,
            "bytes_changed": op.bytes_changed,
        } for op in self.operations]
        return pd.DataFrame(rows, columns=["ts", "op", "path", "dest_path",
                                           "bytes_changed"])


def _keystream(seed: int, rel_path: str, n_bytes: int) -> bytes:
    """Deterministic XOR keystream for one file. XOR twice == original bytes."""
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        block = hashlib.sha256(
            f"rr-evader:{seed}:{rel_path}:{counter}".encode("utf-8")
        ).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:n_bytes])


def _enumerate_targets(root: Path, cfg: EvaderConfig) -> list[Path]:
    """All arena data files, optionally skipping canary-looking paths.

    Only control files (sentinel, registries, the evader/shadow journals) are
    structurally excluded — a *dumb* attacker will happily hit a planted
    canary, which is exactly what decoy_avoidance=False models.
    """
    reserved = {SENTINEL_NAME, JOURNAL_NAME, KEY_NAME, CANARY_REGISTRY_NAME}
    targets: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in reserved for part in rel.parts):
            continue
        if rel.parts[0] == ".ransomradar":
            continue                       # journal home of rr.respond / evader
        lowered = p.as_posix().lower().replace("\\", "/")
        if cfg.decoy_avoidance and any(m in lowered for m in CANARY_MARKERS):
            continue                       # fingerprint-and-skip the decoys
        targets.append(p)
    return targets


def _plan(root: Path, cfg: EvaderConfig, sizes: dict[str, int]) -> list[dict]:
    """Pure op plan from (config, relative paths, sizes). No I/O, no clock."""
    rels = sorted(sizes)
    rng = random.Random(cfg.seed)
    rng.shuffle(rels)

    if cfg.n_processes > 1:                # shard traversal, deterministic order
        shards: list[list[str]] = [[] for _ in range(cfg.n_processes)]
        for i, rel in enumerate(rels):
            shards[i % cfg.n_processes].append(rel)
        rels = [rel for shard in shards for rel in shard]

    plan: list[dict] = []
    if cfg.note_timing == "start":
        plan.append({"op": "note", "rel": NOTE_BASENAME, "bytes": len(_note_text())})
    renamed: list[str] = []
    n_renames = 0 if cfg.in_place or cfg.rename_policy == "none" \
        else int(len(rels) * 0.6)
    for rel in rels:
        n_bytes = max(1, int(sizes[rel] * cfg.encrypt_fraction))
        plan.append({"op": "encrypt", "rel": rel, "bytes": n_bytes})
    for rel in rels[:n_renames]:
        plan.append({"op": "rename", "rel": rel, "bytes": 0})
        renamed.append(rel)
    if cfg.note_timing == "end":
        plan.append({"op": "note", "rel": NOTE_BASENAME, "bytes": len(_note_text())})
    return plan


def _note_text() -> str:
    return ("All your files have been encrypted!\n"
            "To decrypt them, send 0.5 bitcoin to the address below.\n"
            "After payment you will receive your personal decryption key.\n")


def run_evader(root: str | Path, cfg: EvaderConfig) -> AttackTrace:
    """Run the simulated attacker inside the arena and return the trace."""
    root_resolved = require_arena(root)          # refuses non-arenas outright
    targets = _enumerate_targets(root_resolved, cfg)
    if not targets:
        raise ArenaError(f"arena {root_resolved} has no files to attack.")

    sizes = {p.relative_to(root_resolved).as_posix(): p.stat().st_size
             for p in targets}
    plan = _plan(root_resolved, cfg, sizes)
    signature = hashlib.sha256(json.dumps(
        {"config": cfg.signature_payload(), "plan": plan},
        sort_keys=True).encode("utf-8")).hexdigest()[:32]

    journal_rel = [_journal_entry(root_resolved, cfg, step) for step in plan]

    rng = random.Random(cfg.seed ^ 0x5EED)
    ops: list[AttackOp] = []
    start = time.time()
    t = start
    for step, jentry in zip(plan, journal_rel):
        t += (1.0 / cfg.rate_hz) * (1.0 + cfg.sleep_jitter * rng.random())
        ops.append(_apply_step(root_resolved, step, jentry, t))
        if cfg.sleep_jitter or cfg.rate_hz < 500.0:
            time.sleep(max(0.0, (1.0 / cfg.rate_hz)
                           * (1.0 + cfg.sleep_jitter * rng.random())))
    _persist_journal(root_resolved, cfg, journal_rel)
    return AttackTrace(operations=tuple(ops), start_ts=start,
                       end_ts=t, signature=signature, config=cfg)


def _journal_entry(root: Path, cfg: EvaderConfig, step: dict) -> dict:
    """Serialisable record of one planned step (path, bytes, keystream seed)."""
    rel = step["rel"]
    return {
        "op": step["op"],
        "rel": rel,
        "bytes": int(step["bytes"]),
        "dest": (rel + _RENAME_SUFFIX) if step["op"] == "rename" else None,
        "key": f"{cfg.seed}:{rel}",
    }


def _apply_step(root: Path, step: dict, jentry: dict, ts: float) -> AttackOp:
    rel = step["rel"]
    if step["op"] == "note":
        note = resolve_within(root, rel)
        note.write_text(_note_text(), encoding="utf-8")
        return AttackOp(ts=ts, op="note", path=note, bytes_changed=step["bytes"])
    if step["op"] == "rename":
        src = resolve_within(root, rel)
        dst = resolve_within(root, rel + _RENAME_SUFFIX)
        if src.exists():
            src.rename(dst)
        return AttackOp(ts=ts, op="rename", path=src, dest_path=dst)
    target = resolve_within(root, rel)
    n_bytes = min(int(step["bytes"]), target.stat().st_size)
    key_seed, key_rel = jentry["key"].split(":", 1)
    stream = _keystream(int(key_seed), key_rel, n_bytes)
    with open(target, "r+b") as fh:
        fh.seek(0)
        blob = fh.read(n_bytes)
        fh.seek(0)
        fh.write(bytes(a ^ b for a, b in zip(blob, stream)))
    return AttackOp(ts=ts, op="encrypt", path=target, bytes_changed=n_bytes)


def _persist_journal(root: Path, cfg: EvaderConfig, entries: list[dict]) -> None:
    journal = resolve_within(root, _JOURNAL_RELPATH)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps(
        {"config": cfg.signature_payload(), "entries": entries}, indent=2),
        encoding="utf-8")


def _read_journal(root: Path) -> dict:
    journal = resolve_within(root, _JOURNAL_RELPATH)
    if not journal.is_file():
        raise FileNotFoundError(
            f"no evader journal at {journal} - nothing to undo (or the arena "
            "was already restored)."
        )
    return json.loads(journal.read_text(encoding="utf-8"))


def undo_all(root: str | Path, config: EvaderConfig | None = None) -> int:
    """Reverse every recorded operation; returns the number of steps undone.

    Replays the operation journal in reverse: ransom notes are deleted,
    renames are reverted, and encrypted regions are XOR-ed a second time with
    the same keystream — which is the original bytes, bit for bit.
    """
    root_resolved = require_arena(root)
    data = _read_journal(root_resolved)
    undone = 0
    for jentry in reversed(data["entries"]):
        if jentry["op"] == "note":
            note = resolve_within(root_resolved, jentry["rel"])
            if note.exists():
                note.unlink()
        elif jentry["op"] == "rename":
            dst = resolve_within(root_resolved, jentry["dest"])
            src = resolve_within(root_resolved, jentry["rel"])
            if dst.exists():
                dst.rename(src)
        else:
            target = resolve_within(root_resolved, jentry["rel"])
            n_bytes = min(int(jentry["bytes"]), target.stat().st_size)
            key_seed, key_rel = jentry["key"].split(":", 1)
            stream = _keystream(int(key_seed), key_rel, n_bytes)
            with open(target, "r+b") as fh:
                fh.seek(0)
                blob = fh.read(n_bytes)
                fh.seek(0)
                fh.write(bytes(a ^ b for a, b in zip(blob, stream)))
        undone += 1
    resolve_within(root_resolved, _JOURNAL_RELPATH).unlink()
    return undone


if __name__ == "__main__":  # pragma: no cover - smoke demo
    import shutil

    from rr.sim.manifest import build_manifest, verify_manifest
    from rr.sim.safety import create_arena

    tmp = Path(os.environ.get("TEMP", "/tmp")) / "rr_evader_smoke"
    if tmp.exists():
        shutil.rmtree(tmp)
    arena = create_arena(tmp)
    (arena / "docs").mkdir()
    for i in range(6):
        (arena / "docs" / f"file_{i}.txt").write_text(f"data {i}\n" * 50,
                                                      encoding="utf-8")
    manifest = build_manifest(arena)
    cfg = EvaderConfig(rate_hz=2000.0, encrypt_fraction=0.15, seed=7)
    print("running evader with 15% partial encryption…")
    trace = run_evader(arena, cfg)
    print(f"ops={len(trace.operations)} bytes={sum(o.bytes_changed for o in trace.operations)} "
          f"signature={trace.signature[:12]} intact={verify_manifest(arena, manifest).intact}")
    print("undoing…")
    undo_all(arena, cfg)
    print(f"intact after undo: {verify_manifest(arena, manifest).intact}")
    shutil.rmtree(tmp, ignore_errors=True)

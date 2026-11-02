"""rr.sim.canaries — deterministic decoy placement for experiment arenas.

The featurizer (``rr.core.featurize.is_canary_path``) detects canaries by
*path substring*, so every placement strategy must guarantee a marker is
present in the path — a canary the detector cannot see is a wasted tripwire
(the exact failure recorded in the legacy README: ``"canary" in "canaries"``
is ``False``). Placement therefore always lives inside a marker-carrying
directory (``.canary/`` or ``canaries/``) or embeds ``_canary_`` in the file
name, while the file *names themselves* stay realistic — a decoy that
announces itself with ``do_not_touch`` is skipped by careful attackers.

Placement is deterministic for a given seed: same seed, same names, same
locations, on any machine.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from rr.core.featurize import CANARY_MARKERS
from rr.sim.safety import CANARY_REGISTRY_NAME, require_arena, resolve_within

__all__ = [
    "CANARY_MARKERS",
    "CANARY_STRATEGIES",
    "CanaryPlacement",
    "place_canaries",
    "read_registry",
]

#: Realistic decoy payloads: plausible names, plausible content, zero secrets.
_CANARY_FILES: tuple[tuple[str, str], ...] = (
    ("Q4_Payroll_Report.docx", "Employee | Department | Net Pay\n"
     "A. Sharma | Engineering | 82,000\nB. Verma | Finance | 64,500\n"),
    ("FY27_Budget_Allocation.xlsx", "Line item,Allocated,Spent\nInfra,1200000,880000\nTravel,150000,41000\n"),
    ("server_admin_passwords.txt", "prod-db-01: (managed vault entry)\nlegacy-adc: (rotated 2026-09)\n"),
    ("client_master_list.xlsx", "Account,Region,ARR\nNorthwind,EMEA,45000\nContoso,APAC,61000\n"),
    ("vpn_credentials.txt", "gateway.rr.local: portal login required\n"),
    ("expense_claims_nov.xlsx", "Date,Who,Amount,Status\n2026-11-03,Intern,120,approved\n"),
    ("roadmap_2027_internal.docx", "Q1: platform consolidation\nQ2: detection latency work\n"),
    ("ssh_keys_prod.txt", "# placeholders only - no real keys live here\n"),
)

#: Strategies understood by :func:`place_canaries`.
CANARY_STRATEGIES: tuple[str, ...] = ("none", "root", "per_depth", "layered")

_HIDDEN_CANARY_DIR = ".canary"       # matches the '/.canary/' featurizer marker
_VISIBLE_CANARY_DIR = "canaries"     # matches the '/canaries/' featurizer marker


@dataclass(frozen=True)
class CanaryPlacement:
    """Result of a placement pass: what was written, where, and why."""

    root: Path
    strategy: str
    paths: tuple[Path, ...] = field(default_factory=tuple)
    n_placed: int = 0

    def __post_init__(self) -> None:
        for p in self.paths:
            lowered = str(p).lower().replace("\\", "/")
            if not any(m in lowered for m in CANARY_MARKERS):
                raise RuntimeError(
                    f"canary placement produced an invisible decoy: {p} carries "
                    "no featurizer marker - fix the strategy, do not silence "
                    "this check."
                )


def _write_canary(root: Path, rel_dir: str, name: str, content: str) -> Path:
    target = resolve_within(root, Path(rel_dir) / name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(content, encoding="utf-8")
    return target


def place_canaries(
    root: str | Path,
    strategy: str = "per_depth",
    *,
    seed: int = 0,
    n: int | None = None,
) -> CanaryPlacement:
    """Plant decoys in the arena according to ``strategy``.

    Strategies:
      * ``none``      - place nothing (control arm of an experiment).
      * ``root``      - a single decoy in a hidden ``.canary/`` folder at the
        arena root.
      * ``per_depth`` - one decoy per existing directory depth (root plus each
        subdirectory), each inside that directory's ``.canary/`` folder.
      * ``layered``   - ``per_depth`` plus a visible ``canaries/`` folder and a
        ``_canary_``-named file, for experiments that vary marker visibility.

    Deterministic: the same ``seed`` selects the same names on every machine.
    Existing files are never overwritten. A registry JSON is written beside
    the arena sentinel so a later pass can audit what was planted.
    """
    if strategy not in CANARY_STRATEGIES:
        raise ValueError(
            f"unknown canary strategy {strategy!r}; expected one of {CANARY_STRATEGIES}"
        )
    root_resolved = require_arena(root)
    rng = random.Random(seed)

    dirs = sorted({p.parent for p in root_resolved.rglob("*") if p.is_file()})
    if not dirs:
        dirs = [root_resolved]

    chosen: list[Path] = []
    if strategy in ("root", "per_depth", "layered"):
        content_by_name = dict(_CANARY_FILES)
        plan: list[tuple[str, str]] = []
        if strategy == "root":
            name, _ = _CANARY_FILES[rng.randrange(len(_CANARY_FILES))]
            plan.append((_HIDDEN_CANARY_DIR, name))
        else:
            pool = list(_CANARY_FILES)
            rng.shuffle(pool)
            targets = dirs if strategy == "per_depth" else dirs[: max(1, len(dirs))]
            for i, d in enumerate(targets):
                name, _ = pool[i % len(pool)]
                rel = d.relative_to(root_resolved).as_posix()
                rel = "" if rel == "." else rel
                plan.append((f"{rel}/{_HIDDEN_CANARY_DIR}".lstrip("/"), name))
        if strategy == "layered":
            name, _ = _CANARY_FILES[rng.randrange(len(_CANARY_FILES))]
            plan.append((_VISIBLE_CANARY_DIR, name))
            deep = dirs[-1]
            rel = deep.relative_to(root_resolved).as_posix()
            rel = "" if rel == "." else rel
            plan.append((rel, f"_canary_{name}"))

        seen: set[str] = set()
        for rel_dir, name in plan:
            key = f"{rel_dir}/{name}"
            if key in seen:
                continue
            seen.add(key)
            content = content_by_name[name.removeprefix("_canary_")]
            chosen.append(_write_canary(root_resolved, rel_dir, name, content))

    placement = CanaryPlacement(
        root=root_resolved,
        strategy=strategy,
        paths=tuple(chosen),
        n_placed=len(chosen),
    )
    registry = resolve_within(root_resolved, CANARY_REGISTRY_NAME)
    registry.write_text(
        json.dumps({
            "strategy": strategy,
            "seed": seed,
            "paths": [p.relative_to(root_resolved).as_posix() for p in chosen],
        }, indent=2),
        encoding="utf-8",
    )
    return placement


def read_registry(root: str | Path) -> dict:
    """Return the registry JSON written by the last :func:`place_canaries` call."""
    root_resolved = require_arena(root)
    registry = resolve_within(root_resolved, CANARY_REGISTRY_NAME)
    if not registry.is_file():
        raise FileNotFoundError(f"no canary registry in {root_resolved}")
    return json.loads(registry.read_text(encoding="utf-8"))

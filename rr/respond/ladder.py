"""Graduated response ladder with a tamper-evident audit chain.

Detection without response is a dashboard. This module turns a verdict into a
proportionate action, and -- critically -- makes every action *reversible*,
*logged*, and *bounded*.

The ladder, by tier:

0. ``log_and_alert``    -- always safe, always taken.
1. ``snapshot_paths``   -- protect originals via :mod:`rr.respond.shadow`.
2. ``suspend_process``  -- pause the suspect (reversible; auto-reverts on TTL).
3. ``quarantine_files`` -- move newly-created artefacts aside (reversible).
4. ``kill_process``     -- terminate. Irreversible, so gated hardest.

Safety invariants enforced here:

* **Dry-run by default** (``ResponsePolicy.dry_run=True``). Nothing touches a
  real machine unless explicitly enabled.
* **Tier ceiling.** ``max_tier`` hard-caps escalation; ``severity`` maps to a
  ceiling too, so a ``low`` verdict can never kill a process.
* **Human gate.** Tiers above ``require_human_above_tier`` return
  ``requires_human`` unless ``human_confirmation=True`` is passed.
* **Never-kill allowlist.** Critical OS processes and PIDs 0/4 are refused
  outright -- killing ``lsass.exe`` is a worse outcome than the ransomware.
* **Kill switch.** If ``kill_switch_path`` exists, every tier above 0 is
  refused. A single file gives an operator an instant stop button.
* **TTL auto-revert.** Suspensions and quarantines expire and are rolled back
  by :meth:`ResponseLadder.revert_expired`, so a false positive self-heals.
* **Hash-chained audit log.** Each line commits to the previous line's hash, so
  silent tampering or deletion is detectable via
  :meth:`ResponseLadder.verify_audit_chain`.

``psutil`` is intentionally *not* a dependency: process control is behind a
:class:`ProcessController` protocol whose default implementation is a safe
no-op, so the module imports and tests anywhere.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from rr.respond.shadow import ShadowJournal

__all__ = [
    "ALWAYS_PROTECTED_PIDS",
    "DEFAULT_NEVER_KILL",
    "GENESIS_HASH",
    "QUARANTINE_DIRNAME",
    "QUARANTINE_MANIFEST",
    "SEVERITY_MAX_TIER",
    "ActionRecord",
    "AuditVerification",
    "FakeController",
    "ProcessController",
    "PsutilController",
    "ResponseLadder",
    "ResponsePolicy",
    "SafeNoOpController",
    "Verdict",
]

QUARANTINE_DIRNAME = "quarantine"
QUARANTINE_MANIFEST = "quarantine-index.json"
GENESIS_HASH = "0" * 64

#: Processes that must never be terminated. Killing any of these bluescreens or
#: bricks a Windows session -- strictly worse than the ransomware itself.
DEFAULT_NEVER_KILL: frozenset[str] = frozenset(
    {
        "csrss.exe",
        "lsass.exe",
        "wininit.exe",
        "winlogon.exe",
        "smss.exe",
        "services.exe",
        "system",
        "system idle process",
    }
)

#: System PIDs that are never valid targets.
ALWAYS_PROTECTED_PIDS: frozenset[int] = frozenset({0, 4})

#: Highest tier each severity may reach, independent of policy max_tier.
SEVERITY_MAX_TIER: dict[str, int] = {
    "none": 0,
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

_TIER_ACTIONS: dict[int, str] = {
    0: "log_and_alert",
    1: "snapshot_paths",
    2: "suspend_process",
    3: "quarantine_files",
    4: "kill_process",
}


@dataclass(frozen=True)
class Verdict:
    """A detection decision handed to the ladder.

    Attributes:
        ts: Epoch seconds of the decision.
        severity: One of the keys of :data:`SEVERITY_MAX_TIER`.
        pvalue: Conformal p-value backing the decision (lower = more anomalous).
        paths: Paths implicated, used by tier 1 snapshotting.
        pid: Suspect process id, if known.
        process_name: Suspect process name, checked against the allowlist.
        evidence: Free-form supporting data. ``evidence["created_paths"]``
            drives tier 3 quarantine, because only *newly created* artefacts
            are safe to move aside -- quarantining a pre-existing user file
            would itself destroy data.
    """

    ts: float
    severity: str
    pvalue: float | None = None
    paths: tuple[str, ...] = ()
    pid: int | None = None
    process_name: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_MAX_TIER:
            raise ValueError(
                f"severity must be one of {sorted(SEVERITY_MAX_TIER)}, got "
                f"{self.severity!r}"
            )
        if self.pvalue is not None and not 0.0 <= float(self.pvalue) <= 1.0:
            raise ValueError(f"pvalue must be in [0, 1], got {self.pvalue!r}")

    @property
    def severity_ceiling(self) -> int:
        """Highest tier this severity may reach."""
        return SEVERITY_MAX_TIER[self.severity]


@dataclass(frozen=True)
class ActionRecord:
    """One action the ladder attempted.

    ``status`` is ``executed``, ``simulated``, ``refused``, ``failed`` or
    ``requires_human``. A refusal always carries a human-readable ``reason``,
    so the audit log explains *why* the system held back.
    """

    tier: int
    action: str
    target: str
    status: str
    reason: str = ""
    ts: float = 0.0
    reverts_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "action": self.action,
            "target": self.target,
            "status": self.status,
            "reason": self.reason,
            "ts": self.ts,
            "reverts_at": self.reverts_at,
        }


@dataclass(frozen=True)
class AuditVerification:
    """Result of verifying the hash-chained audit log."""

    ok: bool
    lines: int
    broken_line: int | None = None
    reason: str = ""


@dataclass
class _PendingReversal:
    """An action scheduled to be undone when its TTL expires."""

    tier: int
    action: str
    target: str
    reverts_at: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponsePolicy:
    """Bounds on what the ladder is permitted to do.

    Attributes:
        dry_run: When True (default), no action mutates the machine; every
            action is recorded as ``simulated``.
        max_tier: Hard ceiling on escalation regardless of severity.
        require_human_above_tier: Tiers strictly above this need
            ``human_confirmation=True``.
        ttl_seconds: How long reversible actions stay in force before
            :meth:`ResponseLadder.revert_expired` undoes them.
        audit_log_path: Where to append the hash-chained audit log.
        never_kill: Process names that must never be terminated.
        kill_switch_path: If this file exists, all tiers above 0 are refused.
    """

    dry_run: bool = True
    max_tier: int = 2
    require_human_above_tier: int = 3
    ttl_seconds: float = 900.0
    audit_log_path: str | Path | None = None
    never_kill: frozenset[str] = DEFAULT_NEVER_KILL
    kill_switch_path: str | Path | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.max_tier <= 4:
            raise ValueError(f"max_tier must be in [0, 4], got {self.max_tier!r}")
        if self.ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {self.ttl_seconds!r}")


@runtime_checkable
class ProcessController(Protocol):
    """Minimal process-control surface the ladder depends on.

    Keeping this a Protocol is what lets the project avoid a hard ``psutil``
    dependency and test tiers 2 and 4 deterministically with a fake.
    """

    def name(self, pid: int) -> str | None:
        """Return the process name for ``pid``, or ``None`` if unknown."""
        ...

    def suspend(self, pid: int) -> bool:
        """Suspend ``pid``; return True on success."""
        ...

    def resume(self, pid: int) -> bool:
        """Resume ``pid``; return True on success."""
        ...

    def kill(self, pid: int) -> bool:
        """Terminate ``pid``; return True on success."""
        ...


class SafeNoOpController:
    """Default controller that performs no process operations.

    Chosen as the default so a misconfigured pipeline cannot suspend or kill
    anything on a real machine.
    """

    def name(self, pid: int) -> str | None:
        return None

    def suspend(self, pid: int) -> bool:
        return False

    def resume(self, pid: int) -> bool:
        return False

    def kill(self, pid: int) -> bool:
        return False


class FakeController:
    """Deterministic in-memory controller for tests and the demo.

    Args:
        names: Mapping of pid -> process name.
        fail: Pids whose operations should report failure.
    """

    def __init__(
        self,
        names: dict[int, str] | None = None,
        fail: frozenset[int] = frozenset(),
    ) -> None:
        self._names = dict(names or {})
        self._fail = frozenset(fail)
        self.suspended: set[int] = set()
        self.killed: set[int] = set()

    def name(self, pid: int) -> str | None:
        return self._names.get(pid)

    def suspend(self, pid: int) -> bool:
        if pid in self._fail:
            return False
        self.suspended.add(pid)
        return True

    def resume(self, pid: int) -> bool:
        if pid in self._fail:
            return False
        self.suspended.discard(pid)
        return True

    def kill(self, pid: int) -> bool:
        if pid in self._fail:
            return False
        self.killed.add(pid)
        self.suspended.discard(pid)
        return True


class PsutilController:
    """Real process control, importing ``psutil`` lazily.

    ``psutil`` is outside the frozen dependency allowlist, so the import happens
    inside ``__init__`` and raises a clear error if unavailable rather than
    breaking module import for everyone else.
    """

    def __init__(self) -> None:
        try:
            import psutil  # noqa: PLC0415 - deliberate lazy import
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PsutilController requires the optional 'psutil' package, which "
                "is not part of the frozen dependency set. Install it explicitly "
                "or use SafeNoOpController/FakeController."
            ) from exc
        self._psutil = psutil

    def name(self, pid: int) -> str | None:
        try:
            return self._psutil.Process(pid).name()
        except Exception:  # noqa: BLE001 - process may vanish at any moment
            return None

    def suspend(self, pid: int) -> bool:
        try:
            self._psutil.Process(pid).suspend()
            return True
        except Exception:  # noqa: BLE001
            return False

    def resume(self, pid: int) -> bool:
        try:
            self._psutil.Process(pid).resume()
            return True
        except Exception:  # noqa: BLE001
            return False

    def kill(self, pid: int) -> bool:
        try:
            self._psutil.Process(pid).kill()
            return True
        except Exception:  # noqa: BLE001
            return False


class ResponseLadder:
    """Turns a :class:`Verdict` into bounded, reversible, audited actions.

    Args:
        policy: Bounds on behaviour; defaults to a dry-run policy.
        root: Directory the ladder may touch (snapshots, quarantine). Required
            for tiers 1 and 3.
        journal: Shadow journal for tier 1; created from ``root`` if omitted.
        controller: Process control for tiers 2 and 4; defaults to the safe
            no-op controller.
        alert: Callable invoked at tier 0 with a message string.
        clock: Time source, injectable so TTL behaviour is testable.
    """

    def __init__(
        self,
        policy: ResponsePolicy | None = None,
        *,
        root: str | Path | None = None,
        journal: ShadowJournal | None = None,
        controller: ProcessController | None = None,
        alert: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy or ResponsePolicy()
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.controller: ProcessController = controller or SafeNoOpController()
        self.alert = alert
        self.clock = clock
        self._pending: list[_PendingReversal] = []
        self._last_hash = GENESIS_HASH
        if journal is not None:
            self.journal: ShadowJournal | None = journal
        elif self.root is not None:
            self.journal = ShadowJournal(self.root, dry_run=self.policy.dry_run)
        else:
            self.journal = None

    @property
    def quarantine_dir(self) -> Path | None:
        """Directory where quarantined artefacts are moved."""
        if self.root is None:
            return None
        return self.root / ".ransomradar" / QUARANTINE_DIRNAME

    @property
    def pending_reversals(self) -> tuple[_PendingReversal, ...]:
        """Actions still awaiting TTL expiry."""
        return tuple(self._pending)

    @property
    def protected_pids(self) -> frozenset[int]:
        """PIDs that are never valid targets."""
        return ALWAYS_PROTECTED_PIDS

    @property
    def kill_switch_active(self) -> bool:
        """Whether the operator's kill-switch file is present."""
        path = self.policy.kill_switch_path
        return path is not None and Path(path).exists()

    def allowlist_reason(self, verdict: Verdict) -> str | None:
        """Why this verdict's process must not be touched, or ``None``.

        Checked before tiers 2 and 4. Both the PID and the resolved process
        name are tested, because a caller may supply either.
        """
        if verdict.pid is None:
            return "no pid supplied on the verdict"
        if int(verdict.pid) in ALWAYS_PROTECTED_PIDS:
            return f"pid {verdict.pid} is a protected system pid"
        name = verdict.process_name or self.controller.name(int(verdict.pid))
        if name and name.strip().lower() in self.policy.never_kill:
            return f"process {name!r} is on the never-kill allowlist"
        return None


class ResponseLadder:
    """Turns a Verdict into bounded, reversible, audited actions.

    Args:
        policy: Bounds on behaviour; defaults to a dry-run policy.
        root: Directory the ladder may touch (snapshots, quarantine).
        journal: Shadow journal for tier 1; created from root if omitted.
        controller: Process control for tiers 2 and 4; defaults to no-op.
        alert: Callable invoked at tier 0 with a message string.
        clock: Time source, injectable so TTL behaviour is testable.
    """

    def __init__(
        self,
        policy: ResponsePolicy | None = None,
        *,
        root: str | Path | None = None,
        journal: ShadowJournal | None = None,
        controller: ProcessController | None = None,
        alert: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy or ResponsePolicy()
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.controller: ProcessController = controller or SafeNoOpController()
        self.alert = alert
        self.clock = clock
        self._pending: list[_PendingReversal] = []
        self._last_hash = GENESIS_HASH
        if journal is not None:
            self.journal: ShadowJournal | None = journal
        elif self.root is not None:
            self.journal = ShadowJournal(self.root, dry_run=self.policy.dry_run)
        else:
            self.journal = None

    @property
    def quarantine_dir(self) -> Path | None:
        """Directory where quarantined artefacts are moved."""
        if self.root is None:
            return None
        return self.root / ".ransomradar" / QUARANTINE_DIRNAME

    @property
    def pending_reversals(self) -> tuple[_PendingReversal, ...]:
        """Actions still awaiting TTL expiry."""
        return tuple(self._pending)

    @property
    def protected_pids(self) -> frozenset[int]:
        """PIDs that are never valid targets."""
        return ALWAYS_PROTECTED_PIDS

    @property
    def kill_switch_active(self) -> bool:
        """Whether the operator's kill-switch file is present."""
        path = self.policy.kill_switch_path
        return path is not None and Path(path).exists()

    def allowlist_reason(self, verdict: Verdict) -> str | None:
        """Why this verdict's process must not be touched, or None.

        Checked before tiers 2 and 4. Both the PID and the resolved process
        name are tested, because a caller may supply either.
        """
        if verdict.pid is None:
            return "no pid supplied on the verdict"
        if int(verdict.pid) in ALWAYS_PROTECTED_PIDS:
            return f"pid {verdict.pid} is a protected system pid"
        name = verdict.process_name or self.controller.name(int(verdict.pid))
        if name and name.strip().lower() in self.policy.never_kill:
            return f"process {name!r} is on the never-kill allowlist"
        return None

    def _append_audit(self, record: ActionRecord) -> None:
        """Append one hash-chained line to the audit log.

        Each line stores the hash of (previous hash + this payload), so removing
        or editing any line breaks every hash after it. That makes tampering
        detectable rather than merely discouraged.
        """
        payload = record.to_dict()
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        chained = hashlib.sha256(
            (self._last_hash + body).encode("utf-8")
        ).hexdigest()
        line = json.dumps(
            {"prev": self._last_hash, "hash": chained, "record": payload},
            sort_keys=True,
            separators=(",", ":"),
        )
        self._last_hash = chained
        path = self.policy.audit_log_path
        if path is None:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read_audit_log(self) -> list[dict[str, Any]]:
        """Return parsed audit-log lines, or an empty list if there is no log."""
        path = self.policy.audit_log_path
        if path is None or not Path(path).exists():
            return []
        out: list[dict[str, Any]] = []
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out

    def verify_audit_chain(self) -> AuditVerification:
        """Recompute the hash chain and report the first broken link.

        Returns:
            An AuditVerification. ``ok`` is True only when every line's hash
            matches the recomputed value and the chain starts at the genesis
            hash.
        """
        lines = self.read_audit_log()
        expected_prev = GENESIS_HASH
        for index, item in enumerate(lines, start=1):
            record = item.get("record")
            stored_prev = str(item.get("prev", ""))
            stored_hash = str(item.get("hash", ""))
            if stored_prev != expected_prev:
                return AuditVerification(
                    False, len(lines), index, "previous-hash link does not match"
                )
            body = json.dumps(record, sort_keys=True, separators=(",", ":"))
            recomputed = hashlib.sha256(
                (stored_prev + body).encode("utf-8")
            ).hexdigest()
            if recomputed != stored_hash:
                return AuditVerification(
                    False, len(lines), index, "record hash does not match its content"
                )
            expected_prev = stored_hash
        return AuditVerification(True, len(lines))

    def _record(
        self,
        tier: int,
        target: str,
        status: str,
        reason: str = "",
        reverts_at: float | None = None,
    ) -> ActionRecord:
        """Build, audit and return one action record."""
        record = ActionRecord(
            tier=tier,
            action=_TIER_ACTIONS[tier],
            target=target,
            status=status,
            reason=reason,
            ts=float(self.clock()),
            reverts_at=reverts_at,
        )
        self._append_audit(record)
        return record

    def _ceiling(self, verdict: Verdict) -> int:
        """Lowest of the policy ceiling and the severity ceiling."""
        return min(self.policy.max_tier, verdict.severity_ceiling)

    def handle(
        self, verdict: Verdict, *, human_confirmation: bool = False
    ) -> list[ActionRecord]:
        """Run the ladder for one verdict and return every action attempted.

        Tiers are attempted in order from 0 up to the effective ceiling. Tier 0
        always runs, so an alert is emitted even when every other tier is
        refused -- silence is never an acceptable response to a verdict.

        Args:
            verdict: The detection decision.
            human_confirmation: Whether an operator has approved high tiers.

        Returns:
            One ActionRecord per tier attempted, in ascending tier order.
        """
        records: list[ActionRecord] = [self._tier0(verdict)]
        ceiling = self._ceiling(verdict)
        switch_on = self.kill_switch_active
        for tier in range(1, ceiling + 1):
            if switch_on:
                records.append(
                    self._record(
                        tier,
                        "-",
                        "refused",
                        f"kill switch present at {self.policy.kill_switch_path}",
                    )
                )
                continue
            if tier > self.policy.require_human_above_tier and not human_confirmation:
                records.append(
                    self._record(
                        tier,
                        "-",
                        "requires_human",
                        "tier exceeds require_human_above_tier without confirmation",
                    )
                )
                continue
            if tier == 1:
                records.extend(self._tier1(verdict))
            elif tier == 2:
                records.append(self._tier2(verdict))
            elif tier == 3:
                records.extend(self._tier3(verdict))
            elif tier == 4:
                records.append(self._tier4(verdict))
        return records

    def _tier0(self, verdict: Verdict) -> ActionRecord:
        """Always-safe tier: log and alert."""
        pvalue = "n/a" if verdict.pvalue is None else f"{verdict.pvalue:.4g}"
        message = (
            f"RansomRadar {verdict.severity.upper()} verdict: p={pvalue}, "
            f"{len(verdict.paths)} path(s) implicated"
        )
        status = "executed"
        reason = ""
        if self.alert is not None:
            try:
                self.alert(message)
            except Exception as exc:  # noqa: BLE001 - alerting must never crash response
                status = "failed"
                reason = f"alert callback raised: {exc}"
        return self._record(0, message, status, reason)

    def _tier1(self, verdict: Verdict) -> list[ActionRecord]:
        """Protect original bytes before anything else can escalate."""
        if self.journal is None:
            return [
                self._record(
                    1, "-", "refused", "no shadow journal configured (root missing)"
                )
            ]
        if not verdict.paths:
            return [self._record(1, "-", "refused", "verdict carries no paths")]
        records: list[ActionRecord] = []
        for outcome in self.journal.snapshot_many(verdict.paths):
            if outcome.status == "snapshotted":
                status = "executed"
            elif outcome.status == "simulated":
                status = "simulated"
            elif outcome.status == "already_protected":
                status = "executed"
            else:
                status = "refused"
            records.append(
                self._record(1, outcome.path, status, outcome.reason or outcome.status)
            )
        return records

    def _tier2(self, verdict: Verdict) -> ActionRecord:
        """Suspend the suspect process. Reversible, so TTL-tracked."""
        blocked = self.allowlist_reason(verdict)
        if blocked is not None:
            return self._record(2, str(verdict.pid), "refused", blocked)
        pid = int(verdict.pid)  # type: ignore[arg-type]
        reverts_at = float(self.clock()) + self.policy.ttl_seconds
        if self.policy.dry_run:
            return self._record(
                2, str(pid), "simulated", "dry_run=True", reverts_at
            )
        if not self.controller.suspend(pid):
            return self._record(2, str(pid), "failed", "controller.suspend returned False")
        self._pending.append(
            _PendingReversal(2, _TIER_ACTIONS[2], str(pid), reverts_at, {"pid": pid})
        )
        return self._record(
            2, str(pid), "executed", "suspended; auto-resumes on TTL", reverts_at
        )

    def _tier3(self, verdict: Verdict) -> list[ActionRecord]:
        """Move newly created artefacts aside. Only created paths are eligible.

        Quarantining a pre-existing user file would itself be data loss, so the
        tier reads ``evidence['created_paths']`` and nothing else.
        """
        created = verdict.evidence.get("created_paths") or ()
        if isinstance(created, (str, bytes)):
            created = (created,)
        created = tuple(str(p) for p in created)
        if not created:
            return [
                self._record(
                    3, "-", "refused", "no evidence['created_paths'] to quarantine"
                )
            ]
        qdir = self.quarantine_dir
        if qdir is None:
            return [self._record(3, "-", "refused", "no root configured")]
        reverts_at = float(self.clock()) + self.policy.ttl_seconds
        records: list[ActionRecord] = []
        for raw in created:
            source = Path(raw)
            try:
                resolved = source.expanduser().resolve()
                resolved.relative_to(self.root)  # type: ignore[arg-type]
            except (OSError, ValueError):
                records.append(
                    self._record(3, raw, "refused", "path resolves outside the root")
                )
                continue
            if self.policy.dry_run:
                records.append(
                    self._record(3, raw, "simulated", "dry_run=True", reverts_at)
                )
                continue
            if not resolved.is_file():
                records.append(
                    self._record(3, raw, "refused", "not an existing regular file")
                )
                continue
            dest = qdir / f"{int(self.clock() * 1000)}_{resolved.name}"
            try:
                qdir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(resolved), str(dest))
            except OSError as exc:
                records.append(self._record(3, raw, "failed", f"move failed: {exc}"))
                continue
            self._pending.append(
                _PendingReversal(
                    3,
                    _TIER_ACTIONS[3],
                    str(resolved),
                    reverts_at,
                    {"source": str(resolved), "dest": str(dest)},
                )
            )
            records.append(
                self._record(
                    3, raw, "executed", f"moved to {dest.name}", reverts_at
                )
            )
        return records

    def _tier4(self, verdict: Verdict) -> ActionRecord:
        """Terminate the suspect. Irreversible, therefore never TTL-tracked."""
        blocked = self.allowlist_reason(verdict)
        if blocked is not None:
            return self._record(4, str(verdict.pid), "refused", blocked)
        pid = int(verdict.pid)  # type: ignore[arg-type]
        if self.policy.dry_run:
            return self._record(4, str(pid), "simulated", "dry_run=True")
        if not self.controller.kill(pid):
            return self._record(4, str(pid), "failed", "controller.kill returned False")
        return self._record(4, str(pid), "executed", "terminated (irreversible)")

    def _revert_one(self, item: _PendingReversal) -> ActionRecord:
        """Undo a single expired action."""
        if item.tier == 2:
            pid = int(item.payload["pid"])
            ok = self.controller.resume(pid)
            return self._record(
                2,
                str(pid),
                "executed" if ok else "failed",
                "TTL expired: resumed" if ok else "TTL expired: resume failed",
            )
        if item.tier == 3:
            source = Path(str(item.payload["source"]))
            dest = Path(str(item.payload["dest"]))
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), str(source))
            except OSError as exc:
                return self._record(
                    3, str(source), "failed", f"TTL expired: restore failed: {exc}"
                )
            return self._record(
                3, str(source), "executed", "TTL expired: moved back from quarantine"
            )
        return self._record(
            item.tier, item.target, "refused", "tier is not reversible"
        )

    def revert_expired(self, now: float | None = None) -> list[ActionRecord]:
        """Undo every action whose TTL has passed.

        This is what makes a false positive self-healing: a suspended process
        resumes and quarantined files return without operator intervention.

        Args:
            now: Override the current time; defaults to the injected clock.

        Returns:
            One ActionRecord per reversal attempted.
        """
        current = float(self.clock()) if now is None else float(now)
        still_pending: list[_PendingReversal] = []
        records: list[ActionRecord] = []
        for item in self._pending:
            if item.reverts_at <= current:
                records.append(self._revert_one(item))
            else:
                still_pending.append(item)
        self._pending = still_pending
        return records


if __name__ == "__main__":  # pragma: no cover - manual smoke demo
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "arena"
        root.mkdir()
        victim = root / "notes.txt"
        victim.write_text("important\n", encoding="utf-8")
        dropped = root / "HOW_TO_DECRYPT.txt"
        dropped.write_text("pay us\n", encoding="utf-8")

        audit = Path(tmp) / "audit.jsonl"
        controller = FakeController(names={4242: "evil.exe"})
        ladder = ResponseLadder(
            ResponsePolicy(
                dry_run=False,
                max_tier=4,
                require_human_above_tier=3,
                ttl_seconds=1.0,
                audit_log_path=audit,
            ),
            root=root,
            controller=controller,
            alert=lambda msg: print("  alert:", msg),
        )

        print("-- critical verdict, no human confirmation --")
        for rec in ladder.handle(
            Verdict(
                ts=time.time(),
                severity="critical",
                pvalue=0.0008,
                paths=(str(victim),),
                pid=4242,
                process_name="evil.exe",
                evidence={"created_paths": [str(dropped)]},
            )
        ):
            print(f"  T{rec.tier} {rec.action:<16} {rec.status:<14} {rec.reason}")

        print("-- never-kill allowlist --")
        for rec in ladder.handle(
            Verdict(ts=time.time(), severity="critical", pid=1, process_name="lsass.exe"),
            human_confirmation=True,
        ):
            print(f"  T{rec.tier} {rec.action:<16} {rec.status:<14} {rec.reason}")

        print("-- TTL auto-revert --")
        for rec in ladder.revert_expired(now=time.time() + 10.0):
            print(f"  T{rec.tier} {rec.action:<16} {rec.status:<14} {rec.reason}")
        print("  suspended now:", controller.suspended)
        print("  note restored:", dropped.exists())

        check = ladder.verify_audit_chain()
        print(f"audit chain ok={check.ok} lines={check.lines}")

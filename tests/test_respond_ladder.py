"""Tests for the graduated response ladder.

The ladder is the part of the system that can do real damage, so these tests
focus on refusals: the allowlist, the kill switch, the human gate, the tier
ceilings, and the TTL that makes a false positive self-healing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rr.respond.ladder import (
    ALWAYS_PROTECTED_PIDS,
    SEVERITY_MAX_TIER,
    FakeController,
    ResponseLadder,
    ResponsePolicy,
    Verdict,
)


def _arena(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "arena"
    root.mkdir()
    victim = root / "notes.txt"
    victim.write_text("important\n", encoding="utf-8")
    dropped = root / "HOW_TO_DECRYPT.txt"
    dropped.write_text("pay us\n", encoding="utf-8")
    return root, victim, dropped


def _ladder(tmp_path: Path, **policy_kwargs: object) -> tuple[ResponseLadder, FakeController, Path]:
    root, _, _ = _arena(tmp_path)
    controller = FakeController(names={4242: "evil.exe"})
    defaults: dict[str, object] = {
        "dry_run": False,
        "max_tier": 4,
        "require_human_above_tier": 3,
        "ttl_seconds": 1.0,
        "audit_log_path": tmp_path / "audit.jsonl",
    }
    defaults.update(policy_kwargs)
    ladder = ResponseLadder(
        ResponsePolicy(**defaults),  # type: ignore[arg-type]
        root=root,
        controller=controller,
    )
    return ladder, controller, root


def test_policy_defaults_to_dry_run_and_low_ceiling() -> None:
    policy = ResponsePolicy()
    assert policy.dry_run is True
    assert policy.max_tier == 2


def test_dry_run_never_touches_the_machine(tmp_path: Path) -> None:
    ladder, controller, _ = _ladder(tmp_path, dry_run=True)
    records = ladder.handle(
        Verdict(ts=0.0, severity="critical", pid=4242, process_name="evil.exe"),
        human_confirmation=True,
    )
    assert not controller.suspended
    assert not controller.killed
    assert any(r.status == "simulated" for r in records)


def test_severity_ceiling_blocks_escalation(tmp_path: Path) -> None:
    """A 'low' verdict must never reach suspend or kill."""
    ladder, controller, _ = _ladder(tmp_path)
    records = ladder.handle(
        Verdict(ts=0.0, severity="low", pid=4242, process_name="evil.exe")
    )
    assert max(r.tier for r in records) <= SEVERITY_MAX_TIER["low"]
    assert not controller.suspended


def test_max_tier_caps_escalation(tmp_path: Path) -> None:
    ladder, controller, _ = _ladder(tmp_path, max_tier=1)
    records = ladder.handle(Verdict(ts=0.0, severity="critical", pid=4242))
    assert max(r.tier for r in records) == 1
    assert not controller.suspended


def test_human_gate_blocks_kill_without_confirmation(tmp_path: Path) -> None:
    ladder, controller, _ = _ladder(tmp_path)
    records = ladder.handle(
        Verdict(ts=0.0, severity="critical", pid=4242, process_name="evil.exe")
    )
    tier4 = [r for r in records if r.tier == 4]
    assert tier4 and tier4[0].status == "requires_human"
    assert not controller.killed


def test_human_confirmation_allows_kill(tmp_path: Path) -> None:
    ladder, controller, _ = _ladder(tmp_path)
    records = ladder.handle(
        Verdict(ts=0.0, severity="critical", pid=4242, process_name="evil.exe"),
        human_confirmation=True,
    )
    tier4 = [r for r in records if r.tier == 4]
    assert tier4 and tier4[0].status == "executed"
    assert 4242 in controller.killed


@pytest.mark.parametrize("name", ["lsass.exe", "csrss.exe", "winlogon.exe"])
def test_never_kill_allowlist_refuses_critical_processes(tmp_path: Path, name: str) -> None:
    ladder, controller, _ = _ladder(tmp_path)
    records = ladder.handle(
        Verdict(ts=0.0, severity="critical", pid=1234, process_name=name),
        human_confirmation=True,
    )
    for tier in (2, 4):
        matching = [r for r in records if r.tier == tier]
        assert matching and matching[0].status == "refused"
        assert "allowlist" in matching[0].reason
    assert not controller.killed


@pytest.mark.parametrize("pid", sorted(ALWAYS_PROTECTED_PIDS))
def test_protected_system_pids_are_refused(tmp_path: Path, pid: int) -> None:
    ladder, controller, _ = _ladder(tmp_path)
    records = ladder.handle(
        Verdict(ts=0.0, severity="critical", pid=pid), human_confirmation=True
    )
    assert [r for r in records if r.tier == 4][0].status == "refused"
    assert not controller.killed


def test_kill_switch_refuses_everything_above_tier_zero(tmp_path: Path) -> None:
    switch = tmp_path / "STOP"
    switch.write_text("halt\n", encoding="utf-8")
    ladder, controller, _ = _ladder(tmp_path, kill_switch_path=switch)
    assert ladder.kill_switch_active is True
    records = ladder.handle(
        Verdict(ts=0.0, severity="critical", pid=4242, process_name="evil.exe"),
        human_confirmation=True,
    )
    assert records[0].tier == 0 and records[0].status == "executed"
    assert all(r.status == "refused" for r in records if r.tier > 0)
    assert not controller.suspended and not controller.killed


def test_tier1_snapshots_implicated_paths(tmp_path: Path) -> None:
    ladder, _, root = _ladder(tmp_path)
    victim = root / "notes.txt"
    records = ladder.handle(
        Verdict(ts=0.0, severity="medium", paths=(str(victim),))
    )
    tier1 = [r for r in records if r.tier == 1]
    assert tier1 and tier1[0].status == "executed"
    assert ladder.journal is not None
    assert ladder.journal.is_protected(victim)


def test_tier3_only_quarantines_created_paths(tmp_path: Path) -> None:
    """Pre-existing user files must never be moved aside."""
    ladder, _, root = _ladder(tmp_path)
    victim = root / "notes.txt"
    dropped = root / "HOW_TO_DECRYPT.txt"

    records = ladder.handle(
        Verdict(
            ts=0.0,
            severity="high",
            paths=(str(victim),),
            evidence={"created_paths": [str(dropped)]},
        )
    )
    tier3 = [r for r in records if r.tier == 3]
    assert tier3 and tier3[0].status == "executed"
    assert not dropped.exists()
    assert victim.exists()


def test_tier3_without_created_paths_is_refused(tmp_path: Path) -> None:
    ladder, _, root = _ladder(tmp_path)
    records = ladder.handle(
        Verdict(ts=0.0, severity="high", paths=(str(root / "notes.txt"),))
    )
    tier3 = [r for r in records if r.tier == 3]
    assert tier3 and tier3[0].status == "refused"


def test_tier3_refuses_paths_outside_the_root(tmp_path: Path) -> None:
    ladder, _, _ = _ladder(tmp_path)
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("not yours\n", encoding="utf-8")
    records = ladder.handle(
        Verdict(ts=0.0, severity="high", evidence={"created_paths": [str(outside)]})
    )
    tier3 = [r for r in records if r.tier == 3]
    assert tier3 and tier3[0].status == "refused"
    assert outside.exists()


def test_ttl_expiry_resumes_process_and_restores_quarantine(tmp_path: Path) -> None:
    """A false positive should heal itself without operator action."""
    ladder, controller, root = _ladder(tmp_path)
    dropped = root / "HOW_TO_DECRYPT.txt"
    ladder.handle(
        Verdict(
            ts=0.0,
            severity="critical",
            pid=4242,
            process_name="evil.exe",
            evidence={"created_paths": [str(dropped)]},
        )
    )
    assert 4242 in controller.suspended
    assert not dropped.exists()
    assert len(ladder.pending_reversals) == 2

    reversals = ladder.revert_expired(now=1e12)
    assert all(r.status == "executed" for r in reversals)
    assert 4242 not in controller.suspended
    assert dropped.exists()
    assert ladder.pending_reversals == ()


def test_ttl_not_yet_expired_keeps_actions_in_force(tmp_path: Path) -> None:
    ladder, controller, _ = _ladder(tmp_path, ttl_seconds=10_000.0)
    ladder.handle(Verdict(ts=0.0, severity="critical", pid=4242, process_name="evil.exe"))
    assert ladder.revert_expired(now=0.0) == []
    assert 4242 in controller.suspended


def test_kill_is_never_scheduled_for_reversal(tmp_path: Path) -> None:
    ladder, _, _ = _ladder(tmp_path)
    ladder.handle(
        Verdict(ts=0.0, severity="critical", pid=4242, process_name="evil.exe"),
        human_confirmation=True,
    )
    assert all(item.tier != 4 for item in ladder.pending_reversals)


def test_controller_failure_is_reported_not_raised(tmp_path: Path) -> None:
    root, _, _ = _arena(tmp_path)
    ladder = ResponseLadder(
        ResponsePolicy(dry_run=False, max_tier=2),
        root=root,
        controller=FakeController(names={9: "evil.exe"}, fail=frozenset({9})),
    )
    records = ladder.handle(Verdict(ts=0.0, severity="critical", pid=9))
    assert [r for r in records if r.tier == 2][0].status == "failed"


def test_tier0_always_runs_even_when_all_else_is_refused(tmp_path: Path) -> None:
    switch = tmp_path / "STOP"
    switch.write_text("x", encoding="utf-8")
    seen: list[str] = []
    root, _, _ = _arena(tmp_path)
    ladder = ResponseLadder(
        ResponsePolicy(dry_run=False, max_tier=4, kill_switch_path=switch),
        root=root,
        alert=seen.append,
    )
    records = ladder.handle(Verdict(ts=0.0, severity="critical", pvalue=0.001))
    assert records[0].status == "executed"
    assert len(seen) == 1 and "CRITICAL" in seen[0]


def test_alert_exception_does_not_break_response(tmp_path: Path) -> None:
    root, _, _ = _arena(tmp_path)

    def boom(_message: str) -> None:
        raise RuntimeError("pager is down")

    ladder = ResponseLadder(ResponsePolicy(), root=root, alert=boom)
    records = ladder.handle(Verdict(ts=0.0, severity="medium"))
    assert records[0].status == "failed"
    assert "pager is down" in records[0].reason


def test_audit_chain_verifies_and_detects_tampering(tmp_path: Path) -> None:
    ladder, _, root = _ladder(tmp_path)
    ladder.handle(
        Verdict(ts=0.0, severity="medium", paths=(str(root / "notes.txt"),))
    )
    check = ladder.verify_audit_chain()
    assert check.ok is True and check.lines >= 2

    log = Path(str(ladder.policy.audit_log_path))
    lines = log.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["record"]["reason"] = "nothing to see here"
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tampered = ladder.verify_audit_chain()
    assert tampered.ok is False
    assert tampered.broken_line == 1


def test_invalid_severity_is_rejected() -> None:
    with pytest.raises(ValueError, match="severity"):
        Verdict(ts=0.0, severity="catastrophic")


def test_invalid_pvalue_is_rejected() -> None:
    with pytest.raises(ValueError, match="pvalue"):
        Verdict(ts=0.0, severity="high", pvalue=1.5)

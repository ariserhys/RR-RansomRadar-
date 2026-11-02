"""Viva proof: rollback + byte-for-byte SHA-256 verification.

Drives the REAL v2 research layer -- rr.respond.shadow (copy-on-write
journal, first-write-wins) and rr.sim.manifest (ground-truth hashing) -- the
same code exercised by tests/test_respond_shadow.py. This capability is
source-present but intentionally NOT integrated into the legacy dashboard;
run this script on demand to demonstrate the recovery story:

    python verify_rollback.py

Flow: baseline manifest -> protect every file in the shadow journal ->
simulated fast attack (detector live, alerts fire) -> damage report ->
rollback -> quarantine attacker-added files -> final SHA-256 verification.
"""
import time

import collector
import db
import simulator
from config import SANDBOX
from rr.respond.shadow import ShadowJournal
from rr.sim.manifest import build_manifest_data, verify_manifest
from rr.sim.safety import SENTINEL_NAME, create_arena

JOURNAL_DIRNAME = ".ransomradar"


def data_files(root):
    """Regular files in the arena, excluding journal internals + sentinel."""
    return [p for p in root.rglob("*")
            if p.is_file()
            and JOURNAL_DIRNAME not in p.parts
            and p.name != SENTINEL_NAME]


def main():
    print("=" * 64)
    print(" RansomRadar rollback proof (SHA-256, byte-for-byte)")
    print(" Safety: sandbox arena only, simulator-created files, passive")
    print(" detection; rollback via the v2 shadow journal (rr/respond).")
    print("=" * 64)

    db.init_db(reset=True)
    simulator.reset_sandbox()
    create_arena(SANDBOX)

    # [1] Protect originals BEFORE anything touches them (first-write-wins).
    journal = ShadowJournal(SANDBOX, dry_run=False)
    outcomes = journal.snapshot_many(data_files(SANDBOX))
    n_protected = sum(1 for o in outcomes if o.ok)
    before = {rel: h for rel, h in build_manifest_data(SANDBOX).hashes.items()
              if not rel.startswith(JOURNAL_DIRNAME) and rel != SENTINEL_NAME}
    print(f"\n[1/5] Baseline: {n_protected} files protected in the shadow "
          f"journal, {len(before)} files hashed (SHA-256)")

    # [2] Simulated attack with the detector live (collector started here so
    # journal-internal writes never pollute the event stream).
    observer, stop_ev = collector.start(with_detector=True)
    try:
        collector.set_session("demo_attack")
        print("\n[2/5] Simulated attack: canary touch + mass rewrite + "
              "ransom note + .locked renames")
        simulator.run_attack_fast()
        time.sleep(8)
    finally:
        collector.stop(observer, stop_ev)
    alerts = db.all_alerts()
    crit = sum(1 for a in alerts if a["severity"] == "CRITICAL")
    print(f"      detector raised {len(alerts)} alert(s) ({crit} CRITICAL)")

    # [3] Measure the damage against the baseline manifest.
    damage = verify_manifest(SANDBOX, before)
    added_data = [r for r in damage.added if not r.startswith(JOURNAL_DIRNAME)]
    print(f"\n[3/5] Damage report: {len(damage.modified)} modified, "
          f"{len(damage.missing)} missing, {len(added_data)} attacker-added")

    # [4] Rollback, then quarantine whatever the attacker added.
    print("\n[4/5] Rollback: restoring pristine bytes from the journal…")
    result = journal.restore_all(verify=True)
    print(f"      restored {result.n_restored} file(s), "
          f"verified_identical={result.verified_identical}")
    if result.failed:
        for rel, why in result.failed.items():
            print(f"      RESTORE FAILED {rel}: {why}")

    removed = 0
    for p in data_files(SANDBOX):
        rel = p.relative_to(SANDBOX).as_posix()
        if rel not in before:
            p.unlink()
            removed += 1
    print(f"      quarantined {removed} attacker-added file(s)")

    # [5] Prove it: re-hash everything and compare to the baseline.
    final = verify_manifest(SANDBOX, before)
    data_intact = [r for r in final.intact_paths if not r.startswith(JOURNAL_DIRNAME)]
    print(f"\n[5/5] Proof: {len(final.intact_paths)}/{len(before)} files "
          f"byte-for-byte identical to baseline (SHA-256) "
          f"— all_recovered={final.all_recovered}")
    print("      Sample restored files (original digest == restored digest):")
    for rel in data_intact[:4]:
        print(f"        {rel}")
        print(f"          sha256 {before[rel]}")
    if not final.all_recovered or result.failed:
        print("\nRESULT: VERIFICATION FAILED — do not present this as a pass.")
        raise SystemExit(1)
    print("\nRESULT: rollback verified — every original file came back "
          "byte-for-byte identical.")


if __name__ == "__main__":
    main()

"""Activity simulator: benign work + a benign imitation of ransomware behaviour.

SAFETY CONTRACT (state this in your report):
  * every file touched lives in RR(RansomRadar)/sandbox and was created by
    this script;
  * "attack" mode never encrypts anything - it overwrites sandbox files with
    dummy text, renames them, and deletes them (all reversible: --reset);
  * nothing outside the sandbox folder is ever touched;
  * detection is passive - it only writes rows to a local SQLite database.

Tip: pause the collector (Ctrl+C) while running --reset so the sandbox
rebuild itself does not pollute the event stream.
"""
import argparse
import os
import random
import shutil
import time
from pathlib import Path

from config import CANARY_DIR, CANARY_FILES, SANDBOX

random.seed(42)

SEED_DIRS = ["", "docs", "projects", "reports"]
SEED_EXTS = [".txt", ".csv", ".docx", ".xlsx", ".md", ".log", ".pdf"]

# Seed files carry valid magic headers so the file-type-divergence check has
# something real to verify: a benign write keeps the header, an "encrypted"
# write breaks it.
FILE_HEADERS = {".docx": b"PK\x03\x04", ".xlsx": b"PK\x03\x04",
                ".pptx": b"PK\x03\x04", ".pdf": b"%PDF-"}


def reset_sandbox(n_files=30):
    """Wipe and recreate the sandbox with dummy files."""
    if SANDBOX.exists():
        for attempt in range(3):            # Windows: observer may hold a handle briefly
            try:
                shutil.rmtree(SANDBOX)
                break
            except PermissionError:
                time.sleep(1)
    for d in SEED_DIRS:
        (SANDBOX / d).mkdir(parents=True, exist_ok=True)
    (SANDBOX / CANARY_DIR).mkdir(exist_ok=True)
    for i in range(n_files):
        d = SEED_DIRS[i % len(SEED_DIRS)]
        f = SANDBOX / d / f"file_{i:02d}{random.choice(SEED_EXTS)}"
        head = FILE_HEADERS.get(f.suffix, b"")
        body = (f"Dummy document {i} for the RansomRadar sandbox.\n"
                f"Quarterly figures and meeting notes follow.\n" * 8)
        f.write_bytes(head + body.encode("utf-8"))
    print(f"[simulator] sandbox reset: {n_files} dummy files at {SANDBOX}")


def _seed_files():
    # Dot-prefixed paths are control data (shadow journal, arena sentinel),
    # never attack targets - mirrors config.IGNORED_PREFIXES.
    return [str(p) for p in SANDBOX.rglob("*")
            if p.is_file() and CANARY_DIR not in p.parts
            and not any(part.startswith(".") for part in p.relative_to(SANDBOX).parts)]


# --- benign activity -----------------------------------------------------------

def run_benign(duration=40):
    """Slow, human-like edits: ~15-30 events/min, far below every threshold."""
    files = _seed_files()
    if not files:
        raise SystemExit("[simulator] sandbox is empty - run: python simulator.py --reset")
    t0, i = time.time(), 0
    while time.time() - t0 < duration:
        f = Path(random.choice(files))
        roll = random.random()
        if roll < 0.75 and f.exists():                 # edit a document a little
            with f.open("a", encoding="utf-8") as fh:
                fh.write(f"edit line {i}\n")
        elif roll < 0.9:                               # jot a quick note
            (f.parent / f"note_{int(time.time())}_{i}.txt").write_text("quick note\n")
        elif f.exists():                               # tidy rename, same extension
            nf = f.with_name(f"{f.stem}_v{i}{f.suffix}")
            f.rename(nf)
            files[files.index(str(f))] = str(nf)
        i += 1
        time.sleep(random.uniform(1.2, 2.6))
    print(f"[simulator] benign phase done: {i} operations in {duration:.0f}s")


def run_bulk(n_files=100, pause=0.06):
    """Legitimate bulk upload (e.g. a backup script) - fast but innocent.

    Filenames are unique per run so a second run only CREATES new files and
    never rewrites files from a previous run (that would look like a stealth
    attack to the feature extractor)."""
    bdir = SANDBOX / "bulk_upload"
    bdir.mkdir(exist_ok=True)
    run_id = int(time.time())
    tpl = "Quarterly backup record line\n" * 20
    for i in range(n_files):
        (bdir / f"backup_{run_id}_{i:03d}.csv").write_text(tpl)
        time.sleep(pause)
    print(f"[simulator] legitimate bulk upload: {n_files} files copied")


# --- simulated attacks (benign content, real behaviour patterns) -----------------

def run_attack_fast(duration=15):
    """Phase mimicry: canary touch -> mass "encryption" (random-byte rewrites)
    -> ransom note -> mass .locked rename -> deletions. Should trip rules,
    content forensics AND ML."""
    files = _seed_files()
    if not files:
        raise SystemExit("[simulator] sandbox is empty - run: python simulator.py --reset")
    deadline = time.time() + duration

    for cname in CANARY_FILES:                         # phase 1: touch the canaries
        c = SANDBOX / CANARY_DIR / cname
        if c.exists():
            c.write_bytes(os.urandom(2048))            # canary "encrypted" too
            time.sleep(0.1)
    print("[simulator] ATTACK: canary files touched")

    for f in files:                                    # phase 2: mass rewrite
        if time.time() > deadline - 3:
            break
        p = Path(f)
        if p.exists():
            p.write_bytes(os.urandom(3072 + random.randrange(1024)))
        time.sleep(0.1)

    (SANDBOX / "READ_ME_TO_RECOVER.txt").write_text(   # phase 2.5: ransom note
        "All your files have been encrypted!\n"
        "To decrypt them, send 0.5 bitcoin to the address below.\n"
        "After payment you will receive your personal decryption key.\n")
    print("[simulator] ATTACK: ransom note dropped")

    locked = []
    for f in files[: int(len(files) * 0.6)]:           # phase 3: mass .locked rename
        if time.time() > deadline:
            break
        p = Path(f)
        if p.exists():
            np = p.with_suffix(p.suffix + ".locked")
            p.rename(np)
            locked.append(np)
        time.sleep(0.08)

    for p in locked[:8]:                               # phase 4: delete some
        if p.exists():
            p.unlink()
        time.sleep(0.08)
    print("[simulator] ATTACK finished (rewrite + note + rename + delete, sandbox only)")


def run_attack_stealth(duration=25):
    """Low-and-slow full rewrites (~90-110 events/min): safely under every
    static rule threshold, no canary, no renames, no deletions - only the
    learned ML profile notices."""
    files = _seed_files()
    if not files:
        raise SystemExit("[simulator] sandbox is empty - run: python simulator.py --reset")
    t0, n = time.time(), 0
    while time.time() - t0 < duration:
        p = Path(random.choice(files))
        if p.exists():
            p.write_bytes(os.urandom(3072 + random.randrange(1024)))
            n += 1
        time.sleep(random.uniform(0.45, 0.8))
    print(f"[simulator] STEALTH attack done: {n} quiet rewrites "
          "(no canary / renames / deletions)")


def run_attack_partial(duration=25):
    """Header-corruption rewrite: only the first bytes of each file are
    replaced with random data, so content entropy stays at document levels
    (defeating entropy-only detection) while file-type signatures break -
    the evasion modern ransomware actually uses (arXiv 2510.15133)."""
    files = _seed_files()
    if not files:
        raise SystemExit("[simulator] sandbox is empty - run: python simulator.py --reset")
    t0, n = time.time(), 0
    while time.time() - t0 < duration:
        p = Path(random.choice(files))
        if p.exists():
            try:
                data = p.read_bytes()
            except OSError:
                data = b""
            if len(data) > 64:
                p.write_bytes(os.urandom(32) + data[32:])
                n += 1
        time.sleep(random.uniform(0.45, 0.8))
    print(f"[simulator] PARTIAL attack done: {n} header-only corruptions "
          "(low entropy, broken file types)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RansomRadar activity simulator")
    ap.add_argument("--mode", choices=["benign", "bulk", "attack", "stealth", "partial"],
                    default="benign")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds (benign default 40, stealth/partial default 25)")
    ap.add_argument("--reset", action="store_true",
                    help="wipe the sandbox and recreate dummy files first "
                         "(pause the collector while resetting)")
    args = ap.parse_args()
    if args.reset:
        reset_sandbox()
    if args.mode == "benign":
        run_benign(args.duration or 40)
    elif args.mode == "bulk":
        run_bulk()
    elif args.mode == "attack":
        run_attack_fast()
    elif args.mode == "stealth":
        run_attack_stealth(args.duration or 25)
    else:
        run_attack_partial(args.duration or 25)

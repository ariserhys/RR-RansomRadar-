"""Watchdog collector: sandbox file events -> SQLite, plus live detection.

Run me in one terminal:  python collector.py
Everything is read-only from the system's point of view - RansomRadar only
logs what happens inside the sandbox folder.
"""
import os
import threading
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import db
import forensics
import rules
from config import (CANARY_DIR, CANARY_FILES, ENTROPY_RESAMPLE_SECS,
                    IGNORED_EXACT, IGNORED_PREFIXES, IGNORED_SUFFIXES, SANDBOX)

CURRENT_SESSION = {"id": None}

# Decoy contents look like real documents; a decoy that announces itself is a
# wasted tripwire against a careful attacker (see Huntress canary guidance).
CANARY_FILE_CONTENT = {
    "Q4_Payroll_Report.docx": (
        "Q4 PAYROLL REPORT - DRAFT v3\n"
        "============================\n"
        "Prepared by: HR Operations\n\n"
        "Headcount (Q4): 142 (12 new hires, 3 exits)\n"
        "Payroll spend: 6,184,200\n"
        "Pending reimbursements: 412,750\n\n"
        "Figures pending finance sign-off - do not circulate.\n"),
    "FY27_Budget_Allocation.xlsx": (
        "FY27 BUDGET ALLOCATION - working sheet\n\n"
        "Infrastructure: 4,200,000\n"
        "Licenses: 1,850,000\n"
        "Contractors: 2,600,000\n"
        "Contingency: 600,000\n"),
    "server_admin_passwords.txt": (
        "INTERNAL - DO NOT SHARE\n\n"
        "prod-db-admin: (see vault entry #214)\n"
        "staging-deploy: (see vault entry #215)\n"
        "Rotated monthly per policy 7.2.\n"),
}


def set_session(session_id):
    CURRENT_SESSION["id"] = session_id


def _ignored(path):
    name = os.path.basename(path or "")
    return (name.startswith(IGNORED_PREFIXES)
            or name.endswith(IGNORED_SUFFIXES)
            or name in IGNORED_EXACT)


class RadarHandler(FileSystemEventHandler):
    # Windows/OneDrive can emit duplicate modified events for a single write;
    # collapse repeats of the same event within this many seconds.
    DUPLICATE_WINDOW = 0.1

    def __init__(self):
        self._last_seen = {}
        self._last_sample = {}

    def _sample(self, path):
        """Throttled content forensics for modified files. Returns
        (entropy, divergent, note_like) or (None, None, None)."""
        now = time.time()
        if now - self._last_sample.get(path, 0.0) < ENTROPY_RESAMPLE_SECS:
            return None, None, None
        self._last_sample[path] = now
        if len(self._last_sample) > 5000:
            self._last_sample.clear()
        return forensics.analyze_file(path)

    def _log(self, event_type, src, dest=None, entropy=None, divergent=None,
             note_like=None):
        if _ignored(src) or (dest and _ignored(dest)):
            return
        key = (event_type, src, dest)
        now = time.time()
        if now - self._last_seen.get(key, 0.0) < self.DUPLICATE_WINDOW:
            return
        self._last_seen[key] = now
        if len(self._last_seen) > 5000:
            self._last_seen.clear()
        db.insert_event(now, event_type, src, dest, CURRENT_SESSION["id"],
                        entropy, divergent, note_like)
        try:
            rel = os.path.relpath(src, SANDBOX)
        except ValueError:
            rel = src
        line = (f"[collector] {time.strftime('%H:%M:%S')} {event_type:<8} {rel}")
        if dest:
            try:
                line += f" -> {os.path.relpath(dest, SANDBOX)}"
            except ValueError:
                line += f" -> {dest}"
        if entropy is not None:
            line += f" (H={entropy:.1f}/8" + (", TYPE MISMATCH" if divergent else "") + ")"
        print(line, flush=True)

    def on_created(self, event):
        if not event.is_directory:
            self._log("created", event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            return
        entropy, divergent, note_like = self._sample(event.src_path)
        self._log("modified", event.src_path, None, entropy, divergent, note_like)

    def on_deleted(self, event):
        if not event.is_directory:
            self._log("deleted", event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._log("moved", event.src_path, event.dest_path)


def ensure_canaries():
    """Create the decoy canary files BEFORE the observer starts so their own
    creation events never pollute the event stream."""
    cdir = SANDBOX / CANARY_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    for name in CANARY_FILES:
        f = cdir / name
        if not f.exists():
            f.write_text(CANARY_FILE_CONTENT.get(name,
                          "Confidential internal document.\n"))


def start(with_detector=True):
    """Start watching the sandbox. Returns (observer, stop_event)."""
    SANDBOX.mkdir(parents=True, exist_ok=True)
    ensure_canaries()
    observer = Observer()
    observer.schedule(RadarHandler(), str(SANDBOX), recursive=True)
    observer.start()
    time.sleep(0.5)     # let the watch establish before callers act on it
    stop = threading.Event()
    if with_detector:
        threading.Thread(target=rules.LiveDetector().loop, args=(stop,),
                         daemon=True).start()
    return observer, stop


def stop(observer, stop_event):
    stop_event.set()
    observer.stop()
    observer.join()


def main():
    db.init_db()
    print("=" * 62)
    print(" RansomRadar collector")
    print(" Watching: {}".format(SANDBOX))
    print(" Safety: passive logging only - nothing is ever blocked.")
    print(" Stop with Ctrl+C.")
    print("=" * 62)
    observer, stop_event = start(with_detector=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop(observer, stop_event)
        print("Collector stopped.")


if __name__ == "__main__":
    main()

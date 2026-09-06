"""One-command viva demo.

python demo.py   ->  learns normal behaviour, trains the model, then runs a
fast attack, a stealth attack and a partial-encryption attack (all benign
simulations - toasts pop as alerts fire), prints the alerts, then launches
the Streamlit dashboard for you to show.
"""
import subprocess
import sys
import time

import collector
import db
import features
import model
import simulator
from config import MODEL_PATH, ROOT

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass


def main():
    print("=" * 64)
    print(" RansomRadar scripted demo")
    print(" Safety: the simulator only touches self-created files inside")
    print(" sandbox/. No real encryption. Detection is passive logging.")
    print("=" * 64)

    db.init_db(reset=True)
    simulator.reset_sandbox()
    observer, stop_ev = collector.start(with_detector=True)

    try:
        collector.set_session("demo_benign")
        print("\n[1/5] Learning normal behaviour (35 s) - no alerts expected…")
        simulator.run_benign(35)

        print("\n[2/5] Training the learned profile + Isolation Forest…")
        wins, _ = features.windows_from_events(db.all_events())
        model.train(wins)
        print(f"      model saved -> {MODEL_PATH}")

        collector.set_session("demo_attack")
        print("\n[3/5] FAST attack: canary touch + mass 'encryption' + ransom"
              " note + .locked renames.\n      Expect an instant CRITICAL canary"
              " alert (and a desktop toast).")
        simulator.run_attack_fast()
        time.sleep(8)

        collector.set_session("demo_stealth")
        print("\n[4/5] STEALTH attack: quiet full rewrites under every rule"
              " threshold.\n      Expect ML-only alerts (entropy evidence).")
        simulator.run_attack_stealth(20)
        time.sleep(14)

        collector.set_session("demo_partial")
        print("\n[5/5] PARTIAL-encryption attack: only file headers are"
              " corrupted.\n      Entropy stays low (entropy-only tools miss"
              " it) but file-type\n      divergence betrays it - watch the ML"
              " evidence line.")
        simulator.run_attack_partial(20)
        time.sleep(14)
    finally:
        collector.stop(observer, stop_ev)

    alerts = db.all_alerts()
    print(f"\n{len(alerts)} alerts recorded. Highlights:")
    for a in alerts[:6]:
        print(f"  [{a['severity']:<8}] ({a['source']}) {a['explanation'][:130]}")

    print("\nLaunching dashboard…")
    subprocess.Popen([sys.executable, "-m", "streamlit", "run",
                      str(ROOT / "app.py")], cwd=str(ROOT))
    print("Dashboard: http://localhost:8501  (Ctrl+C here to end the demo)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Demo stopped.")


if __name__ == "__main__":
    main()

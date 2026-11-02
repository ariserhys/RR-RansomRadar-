# RansomRadar — Installation Guide

**Supported platforms:** Windows 10/11 · macOS 12 Monterey or later (Intel & Apple Silicon) · Linux (Ubuntu 20.04+)

---

## What you need to install manually (before anything else)

### 1 — Python 3.11 or newer (Required on all platforms)

All packages in `requirements.txt` install via `pip`; Python itself is the only manual prerequisite.

| Platform | What to do |
|---|---|
| **Windows** | Download the installer from [python.org/downloads](https://www.python.org/downloads/). During setup tick **"Add Python to PATH"** and **"pip"**. |
| **macOS** | The system Python (`/usr/bin/python3`) is old. Install a fresh one via **Homebrew** (see Section A below) or from [python.org/downloads](https://www.python.org/downloads/). |
| **Linux** | `sudo apt install python3 python3-pip python3-venv` (Debian/Ubuntu). |

Verify after installing:
```bash
python3 --version   # must print 3.11 or higher
pip3 --version
```

---

## macOS — Section A: One-time system setup

> Skip this section if you are on Windows or Linux.

### A1 — Install Homebrew (the macOS package manager)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. When it finishes, run the two `eval` lines it prints (they add Homebrew to your PATH).

### A2 — Install Python via Homebrew

```bash
brew install python@3.13
```

After this, `python3` and `pip3` will point to the Homebrew-managed Python.

### A3 — Allow Notification Center alerts

RansomRadar uses `osascript` to fire desktop alerts — it's built into macOS, no download needed. The first time the alert fires, macOS may ask whether to allow notifications from *Script Editor / Terminal*. Click **Allow**.

If alerts are silently blocked later:
- **System Settings → Notifications → Terminal** (or *Script Editor*) → turn on **Allow Notifications**.

---

## Windows — Section B: One-time system setup

> Skip this section if you are on macOS or Linux.

### B1 — Enable Focus Assist exclusions (optional but recommended)

Desktop toasts from RansomRadar may be hidden by Windows Focus Assist.

**Settings → System → Focus Assist → Automatic rules** — turn off all auto-rules, or add a priority-app exception.

### B2 — PowerShell execution policy (only if you hit an error)

If PowerShell refuses to run scripts, open **PowerShell as Administrator** and run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Installing RansomRadar (all platforms)

### Step 1 — Get the project files

If you received a ZIP, extract it anywhere (e.g. `~/Projects/RansomRadar`).

```bash
cd path/to/RR(RansomRadar)
```

### Step 2 — Create a virtual environment

Using a virtual environment keeps all packages isolated from your system Python.

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt will change to show `(.venv)` — that means it worked.

### Step 3 — Install all Python dependencies

Everything listed in `requirements.txt` installs with one command:

```bash
pip install -r requirements.txt
```

This downloads and installs:

| Package | What it does |
|---|---|
| `streamlit >= 1.36` | The browser dashboard (`app.py`) |
| `watchdog >= 4.0` | Watches the sandbox folder for file changes |
| `scikit-learn >= 1.5` | Isolation Forest anomaly detection |
| `scipy >= 1.14` | Statistical helpers used by the ML layer |
| `pyod >= 2.0` | ECOD one-class detector (v2 layer) |
| `pandas >= 2.0` | Feature windowing and data frames |
| `numpy >= 1.26` | Numerical arrays, entropy math |
| `joblib >= 1.3` | Saving and loading the trained ML model |
| `matplotlib >= 3.8` | Evaluation plots (`experiment.py`) |
| `pytest >= 8.0` | Test suite (`python -m pytest -q`) |

> **Apple Silicon (M1/M2/M3/M4) note:** all packages ship native ARM64 wheels; `pip install` picks them automatically. No Rosetta needed.

### Step 4 — Verify the installation

```bash
python -m pytest -q
```

Expected output: **72 passed, 1 skipped** in ~40 s. The skipped test is a Windows-only long-path edge case; it's harmless.

---

## Running RansomRadar

All commands run from inside `RR(RansomRadar)/` with the virtual environment active.

### Option A — One-command scripted demo (recommended)

```bash
python demo.py
```

This runs five phases automatically — learns normal behaviour, trains the ML model, fires three simulated attacks, then opens the dashboard at **http://localhost:8501**.

Desktop alert toasts will fire as alerts are detected:
- **macOS** → Notification Center banner (top-right corner)
- **Windows** → Balloon tip in the system tray

### Option B — Manual, three terminals

```bash
# Terminal 1 — reset the sandbox, then start collecting
python simulator.py --reset
python collector.py

# Terminal 2 — run a benign phase, train the model, then attack
python simulator.py --mode benign --duration 60
python model.py train
python simulator.py --mode attack

# Terminal 3 — dashboard
streamlit run app.py
```

Dashboard URL: **http://localhost:8501**

### Option C — Rollback proof

```bash
python verify_rollback.py
```

Demonstrates the SHA-256 copy-on-write rollback: takes a baseline snapshot, runs a simulated attack, then restores every file byte-for-byte.

### Option D — Full experiment (4 min)

```bash
python experiment.py
# or for a shorter ~2 min version:
python experiment.py --quick
```

Outputs the four-detector comparison table (rules / ML / combined / entropy-only ablation) and saves results to `data/experiment_results.csv`.

---

## Simulator modes

| Mode | Command | What it simulates |
|---|---|---|
| Benign | `python simulator.py --mode benign` | Slow human-paced edits — no alerts |
| Bulk | `python simulator.py --mode bulk` | Fast batch copy (like a backup script) |
| Attack | `python simulator.py --mode attack` | Canary touch + mass rewrites + `.locked` renames |
| Stealth | `python simulator.py --mode stealth` | Quiet full rewrites under every rule threshold |
| Partial | `python simulator.py --mode partial` | Header-only corruption (evades entropy-only tools) |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'watchdog'` (or any other package)
The virtual environment is not active. Run the activate command from Step 2 again.

### `streamlit: command not found`
Same cause — activate the virtual environment first, then retry.

### macOS: No notification appears
1. Check **System Settings → Notifications → Terminal** (or *Script Editor*) → Allow.
2. Make sure Do Not Disturb / Focus is off.
3. The webhook fallback still works — alerts are always visible in the dashboard and printed to the terminal.

### Windows: Toast balloon not visible
Check **Settings → System → Focus Assist**. Ensure it is set to **Off** or that Terminal is in the priority app list.

### `PermissionError` on sandbox reset (macOS/Linux)
This happens if the sandbox directory is already open in Finder or another process. Close any open folder windows and retry.

### Apple Silicon: `pip install` fails on a package
Make sure you installed Python via Homebrew (`brew install python@3.13`) and not the python.org `.pkg` installer (which may default to an x86_64 build). Run `python3 -c "import platform; print(platform.machine())"` — it should print `arm64`.

---

## Optional: webhook alerts (Discord / Slack / n8n)

Open `config.py` and set:

```python
WEBHOOK_URL = "https://discord.com/api/webhooks/YOUR_WEBHOOK_HERE"
```

Only the alert title and explanation text are sent — never file names or contents.

---

## Quick reference card

```
# One-time setup
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt

# Every session — activate first
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate.bat     # Windows CMD

# Then run
python demo.py                 # scripted demo (easiest)
python -m pytest -q            # verify everything works
streamlit run app.py           # dashboard only
```

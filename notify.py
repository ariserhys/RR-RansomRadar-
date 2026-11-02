"""Alert delivery: cross-platform desktop toast + optional webhook (opt-in).

Toast:
  Windows — spawns PowerShell with a NotifyIcon balloon tip (built into
            Windows PowerShell 5.1). Rate-limited and fire-and-forget.
  macOS   — uses the built-in `osascript` command to fire a native
            Notification Center banner. No extra dependencies required.
  Linux   — uses `notify-send` (libnotify); silently skips if unavailable.

Webhook: POSTs a small JSON summary to config.WEBHOOK_URL (Discord/Slack/n8n
style). Off by default - no data ever leaves the machine unless the user
configures a URL, and only alert titles/explanations are sent, never file
contents.
"""
import json
import platform
import subprocess
import threading
import time
import urllib.request

from config import TOAST_ENABLED, TOAST_MIN_SECS, WEBHOOK_URL

_lock = threading.Lock()
_last_toast = 0.0
_webhook_failures = 0

_POWERSHELL_TOAST = """
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Warning
$n.Visible = $true
$n.BalloonTipTitle = {title}
$n.BalloonTipText  = {body}
$n.ShowBalloonTip(8000) | Out-Null
Start-Sleep -Seconds 9
$n.Dispose()
"""

_OS = platform.system()   # "Windows" | "Darwin" | "Linux"


def _ps_quote(s: str) -> str:
    """Escape a string for safe embedding in a PowerShell single-quoted literal."""
    return "'" + s.replace("'", "''") + "'"


def _osa_quote(s: str) -> str:
    """Escape a string for safe embedding in an AppleScript double-quoted literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toast(title: str, body: str):
    """Fire a native desktop notification; silently absorbs all OS errors."""
    if _OS == "Darwin":
        # macOS — Notification Center banner via osascript (built-in, no deps).
        script = (
            f"display notification {_osa_quote(body)} "
            f"with title {_osa_quote(title)}"
        )
        try:
            subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            print(f"[notify] macOS toast failed: {exc!r}", flush=True)

    elif _OS == "Windows":
        script = _POWERSHELL_TOAST.format(
            title=_ps_quote(title), body=_ps_quote(body)
        )
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle",
                 "Hidden", "-Command", script],
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            print(f"[notify] Windows toast failed: {exc!r}", flush=True)

    else:
        # Linux — notify-send (libnotify). Silently skips if not installed.
        try:
            subprocess.Popen(
                ["notify-send", "--urgency=critical", "--expire-time=8000",
                 title, body],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass   # notify-send not available; nothing to do


def _webhook(title: str, body: str):
    global _webhook_failures
    if not WEBHOOK_URL:
        return
    payload = json.dumps({"title": title, "body": body,
                          "text": f"{title}\n{body}"}).encode("utf-8")
    req = urllib.request.Request(WEBHOOK_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=3).close()
        _webhook_failures = 0
    except OSError:
        if _webhook_failures < 1:      # warn once, then stay silent
            _webhook_failures += 1
            print("[notify] webhook unreachable (continuing silently)", flush=True)


def alert(severity: str, title: str, body: str):
    """Deliver an alert out-of-band. Toasts: CRITICAL/HIGH, rate-limited.
    Webhook: every alert, only if configured."""
    global _last_toast
    body = " ".join((body or "").split())[:280]

    if WEBHOOK_URL:
        threading.Thread(target=_webhook, args=(title, body), daemon=True).start()

    if not (TOAST_ENABLED and severity in ("CRITICAL", "HIGH")):
        return
    with _lock:
        now = time.time()
        if now - _last_toast < TOAST_MIN_SECS:
            return
        _last_toast = now
    threading.Thread(target=_toast,
                     args=(f"{severity} - {title}", body), daemon=True).start()

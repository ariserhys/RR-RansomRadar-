"""Alert delivery. Currently ntfy phone push, built to fail silently and safely.

:mod:`rr.notify.push` never raises, defaults to dry-run, and suppresses
duplicate alerts, so wiring it into the detector cannot crash detection or spam
a phone during a sustained attack.
"""

from __future__ import annotations

from rr.notify.push import (
    DEFAULT_SERVER,
    Notifier,
    NotifyResult,
    Transport,
    UrllibTransport,
)

__all__ = [
    "DEFAULT_SERVER",
    "Notifier",
    "NotifyResult",
    "Transport",
    "UrllibTransport",
]

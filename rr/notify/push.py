"""Phone push notifications via ntfy, built so alerting can never crash detection.

Design rules:

* **Never raises.** Every failure -- bad topic, no network, HTTP 500, timeout --
  is returned as a :class:`NotifyResult` with ``ok=False``. An alerting bug must
  not take down the detector that produced the alert.
* **Dry-run by default.** ``dry_run=True`` means no packet leaves the machine,
  so tests and demos stay offline. This also satisfies the project's "no network
  in tests" rule.
* **Duplicate suppression.** Identical messages inside ``min_interval_seconds``
  are suppressed. During an attack the detector may fire every window; 200
  identical phone buzzes is a worse user experience than one.
* **Injectable transport.** :class:`Transport` is a protocol, so tests assert on
  a fake and the real :class:`UrllibTransport` uses only the standard library.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "DEFAULT_SERVER",
    "NotifyResult",
    "Notifier",
    "Transport",
    "UrllibTransport",
]

DEFAULT_SERVER = "https://ntfy.sh"
_USER_AGENT = "RansomRadar/2.0"
_VALID_PRIORITIES = frozenset({"min", "low", "default", "high", "urgent"})


@dataclass(frozen=True)
class NotifyResult:
    """Outcome of one send attempt.

    Attributes:
        ok: Whether the notification was delivered (or simulated cleanly).
        status: ``sent``, ``simulated``, ``suppressed``, ``refused`` or ``failed``.
        detail: Human-readable explanation, especially on failure.
        topic: The ntfy topic targeted.
        dry_run: Whether this was a simulation.
        http_status: HTTP status code, when a request actually happened.
        attempts: Number of transport attempts made.
    """

    ok: bool
    status: str
    detail: str = ""
    topic: str = ""
    dry_run: bool = True
    http_status: int | None = None
    attempts: int = 0


@runtime_checkable
class Transport(Protocol):
    """Minimal HTTP POST surface used by the notifier."""

    def post(
        self, url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> int:
        """POST ``body`` to ``url`` and return the HTTP status code."""
        ...


class UrllibTransport:
    """Standard-library HTTP transport. No third-party dependency required."""

    def post(
        self, url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> int:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)


@dataclass
class Notifier:
    """Sends RansomRadar alerts to a phone via an ntfy topic.

    Args:
        topic: ntfy topic name. Treat it as a secret: anyone who knows it can
            read your alerts.
        server: Base URL of the ntfy server.
        dry_run: When True (default) nothing is sent; results are ``simulated``.
        timeout: Per-attempt socket timeout in seconds.
        retries: Extra attempts after a failure.
        min_interval_seconds: Identical messages inside this window are
            suppressed, so a firing detector cannot spam the phone.
        transport: Injectable HTTP transport; defaults to urllib.
        clock: Injectable time source, so suppression is testable.
    """

    topic: str = ""
    server: str = DEFAULT_SERVER
    dry_run: bool = True
    timeout: float = 5.0
    retries: int = 1
    min_interval_seconds: float = 60.0
    transport: Transport | None = None
    clock: object = None
    _last_sent: dict[str, float] = field(default_factory=dict, repr=False)

    def _now(self) -> float:
        """Current time from the injected clock, or the wall clock."""
        if callable(self.clock):
            return float(self.clock())
        return time.time()

    @property
    def url(self) -> str:
        """Full endpoint for the configured topic."""
        return f"{self.server.rstrip('/')}/{self.topic}"

    def _headers(
        self,
        title: str,
        priority: str,
        tags: tuple[str, ...],
    ) -> dict[str, str]:
        """Build ntfy headers. Title/Priority/Tags are ntfy's documented controls."""
        headers = {
            "Title": title,
            "Content-Type": "text/plain; charset=utf-8",
            "User-Agent": _USER_AGENT,
            "Priority": priority,
        }
        if tags:
            headers["Tags"] = ",".join(tags)
        return headers

    def send(
        self,
        message: str,
        *,
        title: str = "RansomRadar alert",
        priority: str = "high",
        tags: tuple[str, ...] = ("rotating_light",),
        force: bool = False,
    ) -> NotifyResult:
        """Send one notification, returning the outcome instead of raising.

        Args:
            message: Body text.
            title: Notification title.
            priority: One of min/low/default/high/urgent.
            tags: ntfy tag names (rendered as emoji on the phone).
            force: Bypass duplicate suppression.

        Returns:
            A :class:`NotifyResult`. ``ok`` is False for every failure mode;
            this method never propagates an exception.
        """
        if not self.topic:
            return NotifyResult(
                False, "refused", "no ntfy topic configured", "", self.dry_run
            )
        if priority not in _VALID_PRIORITIES:
            return NotifyResult(
                False,
                "refused",
                f"priority must be one of {sorted(_VALID_PRIORITIES)}",
                self.topic,
                self.dry_run,
            )

        key = f"{title}\n{message}"
        now = self._now()
        previous = self._last_sent.get(key)
        if not force and previous is not None:
            elapsed = now - previous
            if elapsed < self.min_interval_seconds:
                return NotifyResult(
                    True,
                    "suppressed",
                    f"identical alert sent {elapsed:.1f}s ago "
                    f"(min_interval={self.min_interval_seconds}s)",
                    self.topic,
                    self.dry_run,
                )

        if self.dry_run:
            self._last_sent[key] = now
            return NotifyResult(
                True, "simulated", "dry_run=True; nothing sent", self.topic, True
            )

        transport: Transport = self.transport or UrllibTransport()
        headers = self._headers(title, priority, tags)
        body = message.encode("utf-8")
        attempts = 0
        detail = ""
        status_code: int | None = None
        for _ in range(max(1, self.retries + 1)):
            attempts += 1
            try:
                status_code = int(
                    transport.post(self.url, body, headers, self.timeout)
                )
            except urllib.error.HTTPError as exc:
                status_code = int(exc.code)
                detail = f"HTTP {exc.code} from ntfy"
            except Exception as exc:  # noqa: BLE001 - alerting must never crash
                detail = f"{type(exc).__name__}: {exc}"
            else:
                if 200 <= status_code < 300:
                    self._last_sent[key] = now
                    return NotifyResult(
                        True, "sent", "", self.topic, False, status_code, attempts
                    )
                detail = f"HTTP {status_code} from ntfy"
        return NotifyResult(
            False, "failed", detail or "unknown error", self.topic, False,
            status_code, attempts,
        )


if __name__ == "__main__":  # pragma: no cover - manual smoke demo
    class _FakeTransport:
        """Records requests instead of sending them."""

        def __init__(self, code: int = 200) -> None:
            self.code = code
            self.calls: list[str] = []

        def post(self, url, body, headers, timeout):
            self.calls.append(url)
            return self.code

    clock = [1000.0]
    fake = _FakeTransport()
    notifier = Notifier(
        topic="ransomradar-demo",
        dry_run=False,
        transport=fake,
        clock=lambda: clock[0],
    )
    print("first :", notifier.send("6 canary files touched"))
    print("repeat:", notifier.send("6 canary files touched").status)
    clock[0] += 120.0
    print("later :", notifier.send("6 canary files touched").status)
    print("no topic:", Notifier().send("x").status)
    print("bad prio:", notifier.send("x", priority="loud").status)

    broken = Notifier(topic="t", dry_run=False, transport=_FakeTransport(500), retries=2)
    print("http 500:", broken.send("x"))
    print("transport calls:", len(fake.calls))

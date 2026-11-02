"""Tests for ntfy push notifications.

The contract under test: never raise, never send in dry-run, never spam. All
tests use a fake transport, so the suite makes no network calls.
"""

from __future__ import annotations

import urllib.error

from rr.notify.push import DEFAULT_SERVER, Notifier, NotifyResult


class _FakeTransport:
    """Records POSTs instead of performing them."""

    def __init__(self, code: int = 200, raises: BaseException | None = None) -> None:
        self.code = code
        self.raises = raises
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    def post(self, url, body, headers, timeout):
        self.calls.append((url, body, dict(headers)))
        if self.raises is not None:
            raise self.raises
        return self.code


def _notifier(transport: _FakeTransport, **kwargs: object) -> Notifier:
    clock = kwargs.pop("clock", lambda: 1000.0)
    return Notifier(
        topic="ransomradar-test",
        dry_run=False,
        transport=transport,
        clock=clock,
        **kwargs,  # type: ignore[arg-type]
    )


def test_dry_run_is_default_and_sends_nothing() -> None:
    transport = _FakeTransport()
    notifier = Notifier(topic="t", transport=transport)
    assert notifier.dry_run is True
    result = notifier.send("canary touched")
    assert result.status == "simulated"
    assert result.ok is True
    assert transport.calls == []


def test_successful_send_reports_http_status() -> None:
    transport = _FakeTransport(200)
    result = _notifier(transport).send("6 canary files touched")
    assert isinstance(result, NotifyResult)
    assert result.status == "sent"
    assert result.ok is True
    assert result.http_status == 200
    assert result.attempts == 1
    assert len(transport.calls) == 1


def test_url_and_headers_follow_ntfy_conventions() -> None:
    transport = _FakeTransport()
    notifier = _notifier(transport)
    notifier.send("body text", title="Attack", priority="urgent", tags=("warning",))
    url, body, headers = transport.calls[0]
    assert url == f"{DEFAULT_SERVER}/ransomradar-test"
    assert body == b"body text"
    assert headers["Title"] == "Attack"
    assert headers["Priority"] == "urgent"
    assert headers["Tags"] == "warning"
    assert "RansomRadar" in headers["User-Agent"]


def test_missing_topic_is_refused_not_raised() -> None:
    result = Notifier(dry_run=False).send("anything")
    assert result.status == "refused"
    assert result.ok is False


def test_invalid_priority_is_refused() -> None:
    transport = _FakeTransport()
    result = _notifier(transport).send("x", priority="loud")
    assert result.status == "refused"
    assert transport.calls == []


def test_duplicate_alerts_are_suppressed_inside_the_window() -> None:
    """A firing detector must not buzz the phone every window."""
    transport = _FakeTransport()
    notifier = _notifier(transport, min_interval_seconds=60.0)
    assert notifier.send("same message").status == "sent"
    second = notifier.send("same message")
    assert second.status == "suppressed"
    assert second.ok is True
    assert len(transport.calls) == 1


def test_suppression_expires_after_the_interval() -> None:
    transport = _FakeTransport()
    now = [1000.0]
    notifier = _notifier(transport, min_interval_seconds=60.0, clock=lambda: now[0])
    notifier.send("same message")
    now[0] += 120.0
    assert notifier.send("same message").status == "sent"
    assert len(transport.calls) == 2


def test_force_bypasses_suppression() -> None:
    transport = _FakeTransport()
    notifier = _notifier(transport, min_interval_seconds=600.0)
    notifier.send("same message")
    assert notifier.send("same message", force=True).status == "sent"
    assert len(transport.calls) == 2


def test_distinct_messages_are_not_suppressed() -> None:
    transport = _FakeTransport()
    notifier = _notifier(transport, min_interval_seconds=600.0)
    notifier.send("first")
    assert notifier.send("second").status == "sent"
    assert len(transport.calls) == 2


def test_server_error_retries_then_reports_failure() -> None:
    transport = _FakeTransport(500)
    result = _notifier(transport, retries=2).send("x")
    assert result.status == "failed"
    assert result.ok is False
    assert result.http_status == 500
    assert result.attempts == 3
    assert len(transport.calls) == 3


def test_transport_exception_never_propagates() -> None:
    transport = _FakeTransport(raises=OSError("network unreachable"))
    result = _notifier(transport).send("x")
    assert result.status == "failed"
    assert "network unreachable" in result.detail


def test_http_error_is_captured_with_its_code() -> None:
    error = urllib.error.HTTPError("url", 429, "Too Many Requests", {}, None)  # type: ignore[arg-type]
    transport = _FakeTransport(raises=error)
    result = _notifier(transport, retries=0).send("x")
    assert result.status == "failed"
    assert result.http_status == 429


def test_failed_send_is_not_recorded_for_suppression() -> None:
    """A failure must not suppress the retry that follows it."""
    transport = _FakeTransport(500)
    notifier = _notifier(transport, retries=0, min_interval_seconds=600.0)
    notifier.send("x")
    assert notifier.send("x").status == "failed"

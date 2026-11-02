"""Shared event/window types for RansomRadar.

This module is the single source of truth for the data that flows between the
collector, the windower, the featuriser and every downstream detector. Both
dataclasses are frozen so that windows can share event objects without any
aliasing hazard, and so events are hashable and safely reusable across the
overlapping windows produced by :func:`rr.core.windows.sliding_windows`.

No I/O, no OS-specific behaviour, no network access occurs here or at import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "ENTROPY_MAX_BITS",
    "VALID_KINDS",
    "NormalisedEvent",
    "Window",
    "extension_of",
    "from_legacy_row",
]

#: Canonical event kinds. Anything else is coerced (see :func:`from_legacy_row`).
VALID_KINDS: frozenset[str] = frozenset({"created", "modified", "deleted", "moved"})

#: Shannon entropy of a byte stream is bounded by 8 bits/byte.
ENTROPY_MAX_BITS: float = 8.0

#: Tolerated spellings emitted by older collectors / other platforms.
_KIND_ALIASES: dict[str, str] = {
    "created": "created",
    "create": "created",
    "added": "created",
    "modified": "modified",
    "modify": "modified",
    "write": "modified",
    "written": "modified",
    "changed": "modified",
    "deleted": "deleted",
    "delete": "deleted",
    "removed": "deleted",
    "moved": "moved",
    "move": "moved",
    "rename": "moved",
    "renamed": "moved",
}

#: Fallback when the legacy row carries no usable ``event_type``.
_DEFAULT_KIND = "modified"


def extension_of(path: str | None) -> str:
    """Return the lowercase extension of ``path`` including the leading dot.

    Returns ``""`` when ``path`` is falsy or has no extension. Dotfiles such as
    ``".bashrc"`` are treated as having no extension, matching
    :func:`os.path.splitext`.
    """
    if not path:
        return ""
    ext = os.path.splitext(str(path))[1]
    return ext.lower() if ext else ""


def _as_float_or_none(value: Any) -> float | None:
    """Coerce ``value`` to ``float``, returning ``None`` when not convertible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / +-inf
        return None
    return out


def _as_int_or_none(value: Any) -> int | None:
    """Coerce ``value`` to ``int``, returning ``None`` when not convertible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    """Coerce legacy ``0``/``1``/``None``/``"0"``/``"true"`` flags to ``bool``."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "none", "null"}:
        return False
    return True


def _as_str_or_none(value: Any) -> str | None:
    """Coerce ``value`` to a non-empty ``str`` or ``None``."""
    if value is None:
        return None
    text = str(value)
    return text if text != "" else None


def _clamp_entropy(value: float | None) -> float | None:
    """Clamp entropy into ``[0, 8]`` bits/byte, preserving ``None``."""
    if value is None:
        return None
    return min(max(value, 0.0), ENTROPY_MAX_BITS)


@dataclass(frozen=True)
class NormalisedEvent:
    """A single filesystem event, normalised across collectors and platforms.

    Attributes:
        ts: Unix timestamp in seconds (float).
        kind: One of ``"created"``, ``"modified"``, ``"deleted"``, ``"moved"``.
        path: Source path of the event.
        dest_path: Destination path for ``"moved"`` events, else ``None``.
        ext: Lowercase extension of ``path`` including the dot, ``""`` if none.
        dest_ext: Lowercase extension of ``dest_path``, or ``None``.
        size_delta: Byte-size change if known, else ``None``.
        entropy: Shannon entropy of the sampled bytes in bits/byte (0..8).
        chi2: Chi-square uniformity statistic of the sampled bytes.
        divergent: True when file magic bytes disagree with the extension.
        note_like: True when the file looks like a ransom note.
        pid: Owning process id when attributable.
        process_guid: Stable process identifier when attributable.
        session_id: Capture-session identifier for grouping / evaluation splits.
    """

    ts: float
    kind: str
    path: str
    dest_path: str | None
    ext: str
    dest_ext: str | None
    size_delta: int | None
    entropy: float | None
    chi2: float | None
    divergent: bool
    note_like: bool
    pid: int | None
    process_guid: str | None
    session_id: str | None


@dataclass(frozen=True)
class Window:
    """A half-open ``[start, end)`` slice of the event stream.

    Attributes:
        start: Inclusive window start (Unix seconds).
        end: Exclusive window end (Unix seconds).
        scale: Nominal window length in seconds (``end - start`` by construction).
        events: Events with ``start <= ts < end``, ordered by ``ts``.
    """

    start: float
    end: float
    scale: float
    events: tuple[NormalisedEvent, ...]


def from_legacy_row(row: dict) -> NormalisedEvent:
    """Build a :class:`NormalisedEvent` from a legacy SQLite ``events`` row.

    The legacy schema is::

        id, ts, session_id, event_type, src_path, dest_path,
        entropy, divergent, note_like

    Mapping rules:

    * ``event_type`` -> ``kind`` (lowercased, aliases tolerated). A missing or
      unrecognised value degrades to ``"modified"`` rather than raising, because
      a single malformed row must not abort a whole replay.
    * ``divergent`` / ``note_like`` are ``0``/``1``/``None`` ints -> ``bool``.
    * ``ext`` is derived from ``src_path``; ``dest_ext`` from ``dest_path`` and
      is ``None`` when there is no destination.
    * ``chi2``, ``pid``, ``process_guid`` and ``size_delta`` are ``None``: the
      legacy collector never recorded them.
    """
    data: Mapping[str, Any] = row if isinstance(row, Mapping) else {}

    ts = _as_float_or_none(data.get("ts"))
    raw_kind = data.get("event_type")
    kind = (
        _KIND_ALIASES.get(str(raw_kind).strip().lower(), _DEFAULT_KIND)
        if raw_kind is not None
        else _DEFAULT_KIND
    )

    path = _as_str_or_none(data.get("src_path")) or ""
    dest_path = _as_str_or_none(data.get("dest_path"))

    return NormalisedEvent(
        ts=0.0 if ts is None else ts,
        kind=kind,
        path=path,
        dest_path=dest_path,
        ext=extension_of(path),
        dest_ext=extension_of(dest_path) if dest_path is not None else None,
        size_delta=None,
        entropy=_clamp_entropy(_as_float_or_none(data.get("entropy"))),
        chi2=None,
        divergent=_as_bool(data.get("divergent")),
        note_like=_as_bool(data.get("note_like")),
        pid=None,
        process_guid=None,
        session_id=_as_str_or_none(data.get("session_id")),
    )

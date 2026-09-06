"""Content forensics: entropy + file-type divergence + ransom-note checks.

Sampled by the collector at event time (a 4KB read per modified file,
throttled per path), because by window-close the file may have been renamed
or deleted. Grounded in the CryptoDrop feature set (Scaife et al., ICDCS 2016):
a sharp entropy jump and file-type divergence are the classic in-place
encryption signals. Note: modern ransomware evades entropy-only detection via
partial encryption (arXiv 2510.15133) - which is exactly why the header check
exists independently.
"""
import math
import os
from collections import Counter

from config import (CONTENT_SAMPLE_BYTES, NOTE_CONTENT_KEYWORDS,
                    NOTE_EXTENSIONS, NOTE_NAME_KEYWORDS)

# Magic-number signatures for common document formats (a real deployment would
# use libmagic; this covers the formats our sandbox and the demo produce).
FILE_SIGNATURES = {
    ".docx": b"PK\x03\x04", ".xlsx": b"PK\x03\x04", ".pptx": b"PK\x03\x04",
    ".pdf": b"%PDF-", ".png": b"\x89PNG\r\n\x1a\n", ".jpg": b"\xff\xd8\xff",
    ".jpeg": b"\xff\xd8\xff", ".zip": b"PK\x03\x04", ".gz": b"\x1f\x8b",
    ".exe": b"MZ", ".dll": b"MZ",
}
TEXTUAL_EXTS = {".txt", ".csv", ".md", ".log", ".json", ".xml", ".html", ".ini"}


def shannon_entropy(data: bytes) -> float:
    """Entropy in bits/byte (0-8). Human text ~2-5; encrypted data ~7.5-8."""
    if not data:
        return 0.0
    n = len(data)
    ent = 0.0
    for c in Counter(data).values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def is_textual(sample: bytes) -> bool:
    """>=85% printable ASCII / whitespace counts as text. Encrypted content
    (uniform random bytes) is ~37% printable and always fails this."""
    if not sample:
        return True
    printable = sum(1 for b in sample if 0x20 <= b <= 0x7E or b in (9, 10, 13))
    return printable / len(sample) >= 0.85


def note_name(path) -> bool:
    """Filename matches the ransom-note pattern (Elastic-style keywords)."""
    name = os.path.basename(str(path or "")).lower()
    if not name or not name.endswith(NOTE_EXTENSIONS):
        return False
    return any(k in name for k in NOTE_NAME_KEYWORDS)


def analyze_sample(path, sample: bytes):
    """Classify one content sample -> (divergent, note_like)."""
    ext = os.path.splitext(str(path or ""))[1].lower()

    note = note_name(path)
    if not note and is_textual(sample):
        text = sample.decode("latin-1", errors="ignore").lower()
        if sum(k in text for k in NOTE_CONTENT_KEYWORDS) >= 2:
            note = True

    if ext in FILE_SIGNATURES:
        divergent = not sample.startswith(FILE_SIGNATURES[ext])
    elif ext in TEXTUAL_EXTS or ext == "":
        divergent = not is_textual(sample)
    else:
        divergent = False            # unknown binary format - cannot judge
    return divergent, note


def analyze_file(path):
    """Read a small sample and return (entropy, divergent, note_like).

    Returns (None, None, None) when the file cannot be read (e.g. it was
    deleted between the event and the sample - normal during attacks).
    """
    try:
        with open(path, "rb") as fh:
            sample = fh.read(CONTENT_SAMPLE_BYTES)
    except OSError:
        return None, None, None
    if not sample:
        return 0.0, False, note_name(path)
    divergent, note = analyze_sample(path, sample)
    return round(shannon_entropy(sample), 3), divergent, note

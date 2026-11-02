"""Streaming core: event schema, forensic statistics, windowing, features.

`rr.core` is the lowest layer of RansomRadar and, by contract, imports nothing
from `rr.detect`, `rr.eval`, `rr.sim` or `rr.respond`. Importing this module has
no side effects (no file, socket or subprocess access).
"""

from rr.core.featurize import FEATURE_ORDER, featurize, featurize_window
from rr.core.forensics import chi_square_uniformity
from rr.core.schema import NormalisedEvent, Window, from_legacy_row
from rr.core.windows import sliding_windows

__all__ = [
    "FEATURE_ORDER",
    "NormalisedEvent",
    "Window",
    "chi_square_uniformity",
    "featurize",
    "featurize_window",
    "from_legacy_row",
    "sliding_windows",
]

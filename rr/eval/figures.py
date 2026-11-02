"""rr.eval.figures — the detection/response trade-off, drawn.

The project's central claim is a *trade-off*: you can sit anywhere on the
curve between "twitchy detector, few files lost" and "quiet detector, many
files lost". A Pareto plot makes the frontier explicit: every point is a
detector configuration, the frontier is the set of configurations no other
point dominates, and anything off-frontier is a configuration you should not
ship.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")                       # headless save; no display needed
import matplotlib.pyplot as plt

__all__ = ["pareto_curve"]


def _as_points(points) -> list[tuple[float, float, str]]:
    """Accept (x, y) tuples, (x, y, label) triples, or dicts with x/y/label."""
    out: list[tuple[float, float, str]] = []
    for i, p in enumerate(points):
        if isinstance(p, dict):
            out.append((float(p["x"]), float(p["y"]), str(p.get("label", f"cfg-{i}"))))
        else:
            seq = tuple(p)
            if len(seq) == 2:
                out.append((float(seq[0]), float(seq[1]), f"cfg-{i}"))
            elif len(seq) == 3:
                out.append((float(seq[0]), float(seq[1]), str(seq[2])))
            else:
                raise ValueError(f"point {i} must be (x, y) or (x, y, label).")
    return out


def pareto_frontier(pts: Sequence[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """The non-dominated subset: no other point has both lower x and lower y.

    Both axes are costs (false alarms per hour, files lost), so a point is
    dominated when some other point is <= on both and < on at least one.
    """
    frontier = []
    for x, y, _ in pts:
        dominated = any(ox <= x and oy <= y and (ox < x or oy < y)
                        for ox, oy, _ in pts)
        if not dominated:
            frontier.append((x, y, _))
    return sorted(frontier)


def pareto_curve(points, out_png: str | Path) -> list[tuple[float, float, str]]:
    """Plot the files-lost vs false-alarms trade-off and save it.

    ``points`` is any iterable of ``(false_alarms_per_hour, files_lost)`` or
    ``(x, y, label)`` triples (dicts with ``x``/``y``/``label`` also work).
    Returns the Pareto frontier as sorted ``(x, y, label)`` triples.
    """
    pts = _as_points(points)
    if not pts:
        raise ValueError("no points to plot.")
    frontier = pareto_frontier(pts)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts],
               color="#94A3B8", s=42, label="configurations", zorder=2)
    if frontier:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.step(fx, fy, where="post", color="#0066CC", linewidth=2,
                label="Pareto frontier", zorder=3)
        for x, y, label in frontier:
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="#0F172A")
    ax.set_xlabel("False alarms per benign hour")
    ax.set_ylabel("Files lost before detection")
    ax.set_title("Detection cost frontier — lower-left is better")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return frontier


if __name__ == "__main__":  # pragma: no cover - smoke demo
    import tempfile

    demo = [
        (0.5, 40.0, "quiet"),
        (2.0, 6.0, "balanced"),
        (8.0, 2.0, "twitchy"),
        (3.0, 9.0, "dominated"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pareto.png"
        front = pareto_curve(demo, out)
        print(f"frontier: {[(p[2], p[0], p[1]) for p in front]}")
        print(f"saved: {out.stat().st_size} bytes")

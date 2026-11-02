"""rr.eval.ablation — which features earn their place?

``leave_one_feature_out`` re-scores the detector once per removed feature and
reports the metric drop. The guide's claim — "we can tell you which features
actually earn their place rather than asserting they all do" — is this file.

The function is deliberately generic: the caller supplies a ``score_fn`` that
maps a feature-subset name list to scores over all rows, so the same ablation
works for the legacy envelope, the v2 ensemble, or any future scorer.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from rr.eval.metrics import pr_auc

__all__ = ["leave_one_feature_out"]


def leave_one_feature_out(
    df: pd.DataFrame,
    *,
    label_col: str,
    score_fn: Callable[[Sequence[str]], np.ndarray],
    features: Sequence[str] | None = None,
    metric: Callable[[Sequence[int], np.ndarray], float] = pr_auc,
) -> pd.DataFrame:
    """Re-run scoring once per removed feature; report the metric delta.

    Parameters
    ----------
    df:
        Feature frame (one row per window) including ``label_col``.
    label_col:
        Column with the ground-truth label (1 = attack window).
    score_fn:
        ``score_fn(feature_subset) -> scores`` over **all** rows of ``df``, in
        row order. Closing over ``df`` is the caller's job; this keeps the
        function agnostic to how scores are produced.
    features:
        Candidate features; defaults to every column except ``label_col`` and
        the standard window metadata columns.
    metric:
        ``metric(labels, scores) -> float``; PR-AUC by default.

    Returns
    -------
    pd.DataFrame
        Columns ``feature, metric_without, delta_vs_full`` sorted by delta
        ascending (most damaging removals first). A large negative delta means
        the feature was carrying real signal; a delta near zero means it is a
        passenger.
    """
    meta = {"window_start", "window_end", "scale", "session_id", label_col}
    cols = list(features) if features is not None \
        else [c for c in df.columns if c not in meta]
    if not cols:
        raise ValueError("no candidate features to ablate.")

    labels = df[label_col].to_numpy(dtype=int)
    full = float(metric(labels, np.asarray(score_fn(cols), dtype=float)))

    rows = []
    for feature in cols:
        subset = [c for c in cols if c != feature]
        try:
            score = np.asarray(score_fn(subset), dtype=float)
            value = float(metric(labels, score))
        except Exception as exc:            # a subset a scorer refuses is a result too
            value = float("nan")
            print(f"[ablation] scorer refused subset without {feature!r}: {exc!r}")
        rows.append({"feature": feature,
                     "metric_without": value,
                     "delta_vs_full": value - full})
    return (pd.DataFrame(rows, columns=["feature", "metric_without",
                                        "delta_vs_full"])
            .sort_values("delta_vs_full")
            .reset_index(drop=True))


if __name__ == "__main__":  # pragma: no cover - smoke demo
    rng = np.random.default_rng(0)
    n = 200
    useful = rng.normal(0, 1, n) + rng.choice([0, 4], n, p=[0.8, 0.2])
    noise = rng.normal(0, 1, n)
    passenger = rng.normal(0, 1, n)
    frame = pd.DataFrame({"useful": useful, "noise": noise,
                          "passenger": passenger, "label":
                          (rng.choice([0, 4], n, p=[0.8, 0.2]) > 0).astype(int)})

    def score(feature_subset):
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=64, random_state=0).fit(
            frame[list(feature_subset)])
        return -model.score_samples(frame[list(feature_subset)])

    result = leave_one_feature_out(frame, label_col="label", score_fn=score)
    print(result.to_string(index=False,
                           formatters={"metric_without": "{:.3f}".format,
                                       "delta_vs_full": "{:+.3f}".format}))

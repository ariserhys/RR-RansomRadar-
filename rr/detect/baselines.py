"""Reference systems for honest comparison against the calibrated detector.

These exist to be beaten, and to be beaten *transparently*. In particular
:class:`EnvelopeBaseline` is a faithful reimplementation of the legacy
RansomRadar decision logic so that the evaluation can report it as
"Baseline-1 (adaptive threshold)" rather than quietly deleting it.

All three follow the same tiny interface: ``fit(X)``, ``score(X)``,
``predict(X)``. Scores are oriented so that **higher = more anomalous**.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from rr.detect.detector import _as_feature_matrix, _resolve_feature_order

__all__ = ["EnvelopeBaseline", "WritesThresholdBaseline", "RandomScorer"]


class EnvelopeBaseline:
    """Baseline-1 (adaptive threshold) -- the legacy RansomRadar rule.

    For every feature ``f`` the training pass records the maximum value seen
    and sets an allowed envelope::

        allowed_f = max(floor_f, factor * max_seen_in_training_f)

    with ``factor = 1.4`` by default. A window alerts if **any** feature
    exceeds its allowed value.

    Why this is not a detector
    --------------------------
    * **No false-alarm control whatsoever.** There is no probability
      statement attached to the threshold; ``1.4`` is a guess. You cannot
      state, predict, or tune the alert rate except by trial and error on
      whatever traffic you happen to observe.
    * **It breaks permanently on one spiky training window.** The envelope is
      a running maximum, so a single benign burst -- a backup, a ``git
      checkout``, an OS update touching 40k files -- inflates ``max_seen``
      forever. The envelope never shrinks, so real ransomware afterwards
      passes underneath it. The failure is silent and permanent.
    * **Univariate and axis-aligned.** It cannot express "many renames *and*
      high entropy *together* are suspicious even though neither is extreme
      alone", which is exactly the signature of encryption-in-place.
    * **Scale-blind.** The same envelope is applied to a 10 s and a 60 s
      window unless fitted separately.

    Parameters
    ----------
    factor:
        Multiplier applied to the training maximum. Default ``1.4``, matching
        the legacy implementation.
    floors:
        Optional per-feature absolute minimum for the envelope, keyed by
        feature name. Defaults to an empty mapping (no floors). Unknown keys
        raise, so a typo cannot silently disable a floor.
    feature_order:
        Explicit feature list; ``None`` reads ``rr.core.featurize.FEATURE_ORDER``
        at fit time.
    """

    def __init__(
        self,
        factor: float = 1.4,
        floors: dict[str, float] | None = None,
        *,
        feature_order: Sequence[str] | None = None,
    ) -> None:
        if factor <= 0:
            raise ValueError("factor must be positive.")
        self.factor = float(factor)
        self.floors: dict[str, float] = dict(floors or {})
        self._feature_order_arg = (
            None if feature_order is None else list(feature_order)
        )
        self.feature_order_: list[str] | None = None
        self.max_seen_: np.ndarray | None = None
        self.allowed_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "EnvelopeBaseline":
        """Record per-feature training maxima and derive the envelope."""
        order = _resolve_feature_order(self._feature_order_arg)
        unknown = [f for f in self.floors if f not in order]
        if unknown:
            raise ValueError(
                f"floors refers to unknown features: {unknown}. "
                f"Known features: {order}"
            )
        Xa = _as_feature_matrix(X, order)
        self.feature_order_ = order
        self.max_seen_ = Xa.max(axis=0)
        floor_vec = np.array(
            [float(self.floors.get(f, 0.0)) for f in order], dtype=float
        )
        self.allowed_ = np.maximum(floor_vec, self.factor * self.max_seen_)
        return self

    def _require_fitted(self) -> tuple[list[str], np.ndarray]:
        if self.feature_order_ is None or self.allowed_ is None:
            raise RuntimeError("EnvelopeBaseline is not fitted.")
        return self.feature_order_, self.allowed_

    def _ratios(self, Xa: np.ndarray) -> np.ndarray:
        """Per-feature ``value / allowed`` with a sane zero-envelope rule."""
        _, allowed = self._require_fitted()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                allowed[None, :] > 0.0,
                Xa / np.where(allowed[None, :] > 0.0, allowed[None, :], 1.0),
                np.where(Xa > 0.0, np.inf, 0.0),
            )
        return ratios

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Worst per-feature envelope ratio; ``> 1`` means a violation."""
        order, _ = self._require_fitted()
        Xa = _as_feature_matrix(X, order)
        return self._ratios(Xa).max(axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Boolean alerts: ``True`` iff any feature exceeds its envelope."""
        return self.score(X) > 1.0

    def worst_violation(self, x: Any) -> tuple[str, float, float]:
        """Return ``(feature, value, allowed)`` for the worst feature of one row.

        Accepts a :class:`pandas.Series`, a 1-row DataFrame, a mapping, or a
        1-D array ordered like the fitted feature order. The returned tuple is
        reported verbatim in evaluation tables so the baseline's alerts are
        auditable.
        """
        order, allowed = self._require_fitted()
        if isinstance(x, pd.DataFrame):
            if len(x) != 1:
                raise ValueError("worst_violation expects exactly one row.")
            row = _as_feature_matrix(x, order)[0]
        elif isinstance(x, dict):
            row = _as_feature_matrix(pd.Series(x), order)[0]
        else:
            row = _as_feature_matrix(x, order)[0]
        ratios = self._ratios(row.reshape(1, -1))[0]
        j = int(np.argmax(ratios))
        return order[j], float(row[j]), float(allowed[j])

    def envelope(self) -> pd.Series:
        """The fitted allowed maximum per feature, as a named Series."""
        order, allowed = self._require_fitted()
        return pd.Series(allowed, index=list(order), name="allowed")


class WritesThresholdBaseline:
    """Trivial single-feature rule: alert when ``existing_writes > k``.

    The point of this baseline is to expose how much (or how little) the rest
    of the pipeline adds over "a lot of files were overwritten". If the full
    detector cannot beat this on both latency and false alarms, the extra
    machinery is not earning its keep.

    The feature name is a constructor parameter rather than a hardcoded
    constant so the rule survives a rename of the upstream featuriser; the
    default matches the current schema and is validated against the input
    columns at fit time.
    """

    def __init__(self, k: float = 50.0, *, feature: str = "existing_writes") -> None:
        self.k = float(k)
        self.feature = str(feature)
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> "WritesThresholdBaseline":
        """Validate that the configured feature exists. Nothing is learned."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WritesThresholdBaseline.fit expects a DataFrame.")
        if self.feature not in X.columns:
            raise ValueError(
                f"Column {self.feature!r} not found; available columns "
                f"include {list(X.columns)[:12]}..."
            )
        self.fitted_ = True
        return self

    def _values(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("WritesThresholdBaseline is not fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a DataFrame.")
        if self.feature not in X.columns:
            raise ValueError(f"Column {self.feature!r} not found in input.")
        values = X[self.feature].to_numpy(dtype=float, copy=True)
        return np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Raw feature value; monotone in the alert decision."""
        return self._values(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Boolean alerts: ``value > k``."""
        return self._values(X) > self.k


class RandomScorer:
    """Seeded uniform random scores -- the floor any real detector must clear.

    Included because an ROC/PR curve without a random reference is hard to
    read, and because a detector that is indistinguishable from this on some
    slice of the data should be reported as such.
    """

    def __init__(self, seed: int = 0, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0, 1].")
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> "RandomScorer":
        """No-op, present for interface symmetry."""
        self.fitted_ = True
        return self

    @staticmethod
    def _n_rows(X: Any) -> int:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return int(len(X))
        arr = np.asarray(X)
        return int(arr.shape[0]) if arr.ndim > 1 else 1

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Deterministic ``Uniform(0, 1)`` draws given the seed and row count."""
        if not self.fitted_:
            raise RuntimeError("RandomScorer is not fitted.")
        rng = np.random.default_rng(self.seed)
        return rng.random(self._n_rows(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Boolean alerts: ``score > threshold``."""
        return self.score(X) > self.threshold

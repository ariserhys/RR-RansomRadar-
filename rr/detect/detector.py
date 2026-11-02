"""Calibrated one-class anomaly detector for behavioural ransomware detection.

Design summary
--------------
1. **Ensemble (one-class).** Fitted on *benign* windows only. Two members:

   * **ECOD** (Empirical-Cumulative-distribution-based Outlier Detection).
     Parameter-free: it has no bandwidth, no ``k``, no contamination-driven
     geometry -- it only estimates, per dimension, the empirical left/right
     tail probability of a value and aggregates ``-log`` of those tails.
     Because the aggregation is a plain sum over dimensions, the score is
     *interpretable by construction*: the per-dimension term
     ``-log(tail_prob_j)`` is exactly that feature's additive contribution.
   * **IsolationForest**, which is *not* decorative here: it is z- or
     rank-normalised onto a common scale and enters the combined score with
     a real, non-zero weight, so it participates in every p-value and every
     alert decision.

2. **Split-conformal calibration.** Raw anomaly scores are not probabilities
   and are not comparable across deployments. :meth:`Detector.calibrate`
   stores scores of *held-out benign* windows, and :meth:`Detector.pvalue`
   converts a new score into a conformal p-value with a finite-sample,
   distribution-free validity guarantee that requires **no attack labels**.

3. **Sequential decision.** A per-window bound of ``alpha`` is not an
   operational false-alarm rate: at a 1 s hop, ``alpha = 0.01`` permits
   ~864 alerts/day, and even at 10 s non-overlapping windows ~8.6/day.
   :meth:`Detector.decide` therefore fires only on *sustained* evidence
   across ``k`` consecutive windows.

The module never hardcodes feature names or counts: ``FEATURE_ORDER`` is read
from :mod:`rr.core.featurize` at fit time (or supplied explicitly, which is
what the test-suite fixtures do), captured in the fitted state, and persisted.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

__all__ = ["Detector", "SCHEMA_VERSION"]

LOGGER = logging.getLogger(__name__)

#: Bumped whenever the persisted payload layout changes.
SCHEMA_VERSION = 3

#: Finite substitutes for non-finite feature values (ratio features can divide
#: by zero upstream; EWMA features can be NaN during warm-up).
_POSINF_SUBSTITUTE = 1e12
_NEGINF_SUBSTITUTE = -1e12

try:  # pragma: no cover - exercised indirectly via the monkeypatched tests
    from pyod.models.ecod import ECOD as _PyODECOD

    _HAS_PYOD = True
except Exception:  # pragma: no cover - pyod is an optional dependency
    _PyODECOD = None  # type: ignore[assignment]
    _HAS_PYOD = False


CombineMethod = Literal["z", "rank"]
SequentialMethod = Literal["hmp", "fisher"]


# --------------------------------------------------------------------------- #
# Feature-order handling
# --------------------------------------------------------------------------- #
def _resolve_feature_order(explicit: Sequence[str] | None) -> list[str]:
    """Return the feature order to use, reading ``rr.core`` at runtime.

    Parameters
    ----------
    explicit:
        Optional feature list. When ``None`` the canonical
        ``rr.core.featurize.FEATURE_ORDER`` is imported lazily. The import is
        deliberately lazy so this module has no import-time side effects and
        can be unit-tested without the rest of the package.

    Raises
    ------
    RuntimeError
        If no explicit order is given and ``rr.core.featurize`` cannot be
        imported.
    """
    if explicit is not None:
        order = list(explicit)
        if not order:
            raise ValueError("feature_order must be a non-empty list of names.")
        if len(set(order)) != len(order):
            raise ValueError("feature_order contains duplicate feature names.")
        return order
    try:
        from rr.core.featurize import FEATURE_ORDER  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on AGENT-A
        raise RuntimeError(
            "Could not import rr.core.featurize.FEATURE_ORDER; pass "
            "feature_order=[...] explicitly to the Detector."
        ) from exc
    return list(FEATURE_ORDER)


def _as_feature_matrix(X: Any, feature_order: Sequence[str]) -> np.ndarray:
    """Select ``feature_order`` from ``X`` and return a sanitised float matrix.

    ``X`` may be a :class:`pandas.DataFrame` whose columns are a *superset* of
    ``feature_order`` (the extra ``window_start``/``window_end``/``scale``/
    ``session_id`` columns produced upstream are ignored), a 2-D array whose
    column count matches, or a single 1-D row.

    Non-finite values are replaced by finite substitutes (NaN -> 0.0,
    +/-inf -> +/-1e12) so that a single divide-by-zero ratio feature cannot
    poison an entire fit.
    """
    order = list(feature_order)
    if isinstance(X, pd.DataFrame):
        missing = [f for f in order if f not in X.columns]
        if missing:
            raise ValueError(
                f"Input is missing required feature columns: {missing}. "
                f"Expected a superset of {order}."
            )
        arr = X.loc[:, order].to_numpy(dtype=float, copy=True)
    elif isinstance(X, pd.Series):
        missing = [f for f in order if f not in X.index]
        if missing:
            raise ValueError(f"Input Series is missing features: {missing}.")
        arr = X.loc[order].to_numpy(dtype=float, copy=True).reshape(1, -1)
    else:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != len(order):
            raise ValueError(
                f"Array input must have {len(order)} columns matching "
                f"{order}; got shape {arr.shape}."
            )
        arr = arr.copy()
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=_POSINF_SUBSTITUTE,
            neginf=_NEGINF_SUBSTITUTE,
        )
    if arr.shape[0] == 0:
        raise ValueError("Input contains zero rows.")
    return arr


# --------------------------------------------------------------------------- #
# ECOD: self-implemented tail-probability model
# --------------------------------------------------------------------------- #
@dataclass
class _TailProbECOD:
    """Self-implemented ECOD (empirical per-dimension tail probabilities).

    For every dimension ``j`` the training sample defines an empirical CDF.
    For a query value ``x_j`` we form the two *smoothed* tail probabilities

    ``left_j  = (1 + #{train_j <= x_j}) / (n + 1)``
    ``right_j = (1 + #{train_j >= x_j}) / (n + 1)``

    The ``+1 / n+1`` smoothing keeps the probabilities strictly inside
    ``(0, 1]`` so ``-log`` is always finite, and makes the estimate the usual
    conservative plug-in for an unseen sample.

    Three aggregate scores are formed, exactly as in the ECOD paper:

    * ``O_left  = sum_j -log(left_j)``     (detects unusually *small* values)
    * ``O_right = sum_j -log(right_j)``    (detects unusually *large* values)
    * ``O_skew  = sum_j -log(t_j)`` where ``t_j`` is the left tail if the
      training skewness of dimension ``j`` is negative and the right tail
      otherwise -- i.e. the tail in which that dimension's mass is thin.

    The final score is ``max(O_left, O_right, O_skew)``. Higher is more
    anomalous. There are **no hyperparameters**.

    This class is used unconditionally to produce per-feature contributions
    (see :meth:`Detector.per_feature_contributions`) and is also used as the
    scoring member when :mod:`pyod` is unavailable.
    """

    train_sorted_: np.ndarray | None = None
    skew_: np.ndarray | None = None
    n_: int = 0

    def fit(self, Xa: np.ndarray) -> "_TailProbECOD":
        """Store the sorted training columns and their skewness."""
        if Xa.ndim != 2:
            raise ValueError("Expected a 2-D training matrix.")
        self.train_sorted_ = np.sort(Xa, axis=0)
        self.n_ = int(Xa.shape[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            skew = stats.skew(Xa, axis=0, bias=False)
        self.skew_ = np.nan_to_num(np.asarray(skew, dtype=float), nan=0.0)
        return self

    def _check_fitted(self) -> np.ndarray:
        if self.train_sorted_ is None or self.skew_ is None:
            raise RuntimeError("_TailProbECOD is not fitted.")
        return self.train_sorted_

    def tail_probs(self, Xa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(left_prob, right_prob)`` matrices shaped like ``Xa``."""
        train = self._check_fitted()
        n = self.n_
        left = np.empty_like(Xa, dtype=float)
        right = np.empty_like(Xa, dtype=float)
        for j in range(Xa.shape[1]):
            col = train[:, j]
            le = np.searchsorted(col, Xa[:, j], side="right")
            ge = n - np.searchsorted(col, Xa[:, j], side="left")
            left[:, j] = (1.0 + le) / (n + 1.0)
            right[:, j] = (1.0 + ge) / (n + 1.0)
        return left, right

    def contributions(self, Xa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(per_dim_contributions, score)``.

        The contribution matrix holds ``-log(tail_prob)`` for the tail
        orientation that produced the winning aggregate for that row, so the
        row of contributions sums exactly to that row's score.
        """
        left, right = self.tail_probs(Xa)
        assert self.skew_ is not None
        o_left = -np.log(left)
        o_right = -np.log(right)
        use_right = self.skew_ >= 0.0
        o_skew = np.where(use_right[None, :], o_right, o_left)

        stacked = np.stack([o_left, o_right, o_skew], axis=0)  # (3, n, d)
        sums = stacked.sum(axis=2)  # (3, n)
        winner = np.argmax(sums, axis=0)  # (n,)
        rows = np.arange(Xa.shape[0])
        contrib = stacked[winner, rows, :]
        score = sums[winner, rows]
        return contrib, score

    def score(self, Xa: np.ndarray) -> np.ndarray:
        """Anomaly score, higher = more anomalous."""
        return self.contributions(Xa)[1]


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class Detector:
    """One-class ensemble detector with conformal calibration.

    Parameters
    ----------
    feature_order:
        Explicit feature list. When ``None`` (production default) the list is
        read from ``rr.core.featurize.FEATURE_ORDER`` at :meth:`fit` time and
        frozen into the fitted state.
    combine:
        ``"z"`` (default) or ``"rank"`` -- how member scores are put on a
        common scale before averaging. See :meth:`score` for the rationale.
    weights:
        ``(w_ecod, w_iforest)``; normalised to sum to 1. Default ``(0.5, 0.5)``.
        ECOD is the primary member in the sense that it supplies the
        interpretable per-feature decomposition, but IsolationForest carries a
        genuine half of the decision weight.
    n_estimators, max_samples:
        IsolationForest sizing.
    sliding_calibration:
        If set, :meth:`calibrate` retains only the most recent ``n`` benign
        scores. Filesystem activity is non-stationary -- nightly backups,
        package installs, indexing sweeps -- which violates exchangeability
        with respect to calibration data collected weeks earlier. A sliding
        window restores *approximate* local exchangeability at the cost of a
        noisier p-value grid (resolution is ``1 / (n_cal + 1)``).
    random_state:
        Seed for IsolationForest; the rest of the pipeline is deterministic.
    """

    def __init__(
        self,
        feature_order: Sequence[str] | None = None,
        *,
        combine: CombineMethod = "z",
        weights: tuple[float, float] = (0.5, 0.5),
        n_estimators: int = 200,
        max_samples: int | float | str = "auto",
        sliding_calibration: int | None = None,
        random_state: int = 0,
    ) -> None:
        if combine not in ("z", "rank"):
            raise ValueError("combine must be 'z' or 'rank'.")
        if len(weights) != 2 or any(w < 0 for w in weights):
            raise ValueError("weights must be two non-negative numbers.")
        total = float(weights[0] + weights[1])
        if total <= 0:
            raise ValueError("weights must not both be zero.")
        if sliding_calibration is not None and sliding_calibration < 2:
            raise ValueError("sliding_calibration must be >= 2 when set.")

        self._feature_order_arg = (
            None if feature_order is None else list(feature_order)
        )
        self.combine: CombineMethod = combine
        self.weights = (weights[0] / total, weights[1] / total)
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.sliding_calibration = sliding_calibration
        self.random_state = int(random_state)

        # Fitted state.
        self.feature_order_: list[str] | None = None
        self.ecod_backend_: str | None = None
        self._ecod: Any = None
        self._contrib_model: _TailProbECOD | None = None
        self._iforest: IsolationForest | None = None
        self._norm: dict[str, Any] = {}
        self._cal_scores_recent: np.ndarray | None = None
        self._cal_scores_sorted: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self.feature_order_ is not None

    @property
    def is_calibrated(self) -> bool:
        """Whether :meth:`calibrate` has stored benign scores."""
        return self._cal_scores_sorted is not None

    @property
    def n_calibration(self) -> int:
        """Number of retained calibration scores."""
        return 0 if self._cal_scores_sorted is None else int(
            self._cal_scores_sorted.size
        )

    def _require_fitted(self) -> list[str]:
        if self.feature_order_ is None:
            raise RuntimeError("Detector is not fitted; call fit(X) first.")
        return self.feature_order_

    def fit(self, X: pd.DataFrame) -> "Detector":
        """Fit both ensemble members on **benign windows only**.

        Parameters
        ----------
        X:
            DataFrame whose columns are a superset of ``FEATURE_ORDER``.
            Extra columns (``window_start``, ``window_end``, ``scale``,
            ``session_id``) are ignored. Rows must be benign: this is a
            one-class model and contamination degrades the empirical tails
            that both the ECOD score and the conformal calibration rely on.

        Notes
        -----
        Windows overlap (1 s hop), so rows are strongly autocorrelated. That
        does not bias the *fit* (the empirical marginals are still consistent)
        but it does matter for calibration -- see :meth:`calibrate`.
        """
        order = _resolve_feature_order(self._feature_order_arg)
        Xa = _as_feature_matrix(X, order)
        if Xa.shape[0] < 8:
            raise ValueError(
                f"Need at least 8 benign windows to fit; got {Xa.shape[0]}."
            )

        # Contribution model is always our own, so per-feature attribution is
        # available regardless of which pyod version (if any) is installed.
        self._contrib_model = _TailProbECOD().fit(Xa)

        if _HAS_PYOD and _PyODECOD is not None:
            ecod = _PyODECOD()
            ecod.fit(Xa)
            self._ecod = ecod
            self.ecod_backend_ = "pyod"
        else:
            self._ecod = _TailProbECOD().fit(Xa)
            self.ecod_backend_ = "fallback"
        LOGGER.info(
            "Detector.fit: ECOD backend=%s, n_windows=%d, n_features=%d",
            self.ecod_backend_,
            Xa.shape[0],
            Xa.shape[1],
        )

        self._iforest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            bootstrap=False,
        ).fit(Xa)

        self.feature_order_ = order
        raw = self._raw_member_scores(Xa)
        self._norm = self._fit_normaliser(raw)
        # Refitting invalidates any previous calibration.
        self._cal_scores_recent = None
        self._cal_scores_sorted = None
        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _raw_member_scores(self, Xa: np.ndarray) -> dict[str, np.ndarray]:
        """Raw, un-normalised member scores (higher = more anomalous)."""
        if self._ecod is None or self._iforest is None:
            raise RuntimeError("Detector is not fitted.")
        if self.ecod_backend_ == "pyod":
            ecod_raw = np.asarray(
                self._ecod.decision_function(Xa), dtype=float
            ).ravel()
        else:
            ecod_raw = np.asarray(self._ecod.score(Xa), dtype=float).ravel()
        # score_samples is higher for inliers; negate so higher = anomalous.
        iforest_raw = -np.asarray(
            self._iforest.score_samples(Xa), dtype=float
        ).ravel()
        return {"ecod": ecod_raw, "iforest": iforest_raw}

    def _fit_normaliser(self, raw: dict[str, np.ndarray]) -> dict[str, Any]:
        norm: dict[str, Any] = {"method": self.combine}
        for name, values in raw.items():
            if self.combine == "z":
                mu = float(np.mean(values))
                sd = float(np.std(values))
                norm[name] = {"mu": mu, "sd": sd if sd > 1e-12 else 1.0}
            else:
                norm[name] = {"train_sorted": np.sort(values)}
        return norm

    def _normalise(self, name: str, values: np.ndarray) -> np.ndarray:
        stats_ = self._norm[name]
        if self._norm["method"] == "z":
            return (values - stats_["mu"]) / stats_["sd"]
        train_sorted = stats_["train_sorted"]
        n = train_sorted.size
        ranks = np.searchsorted(train_sorted, values, side="right")
        return ranks / float(n)

    def member_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return raw and normalised scores for each ensemble member.

        Useful for diagnostics and for demonstrating that IsolationForest
        genuinely moves the combined score.
        """
        order = self._require_fitted()
        Xa = _as_feature_matrix(X, order)
        raw = self._raw_member_scores(Xa)
        data = {}
        for name, values in raw.items():
            data[f"{name}_raw"] = values
            data[f"{name}_norm"] = self._normalise(name, values)
        frame = pd.DataFrame(data)
        if isinstance(X, pd.DataFrame):
            frame.index = X.index
        return frame

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Combined ensemble anomaly score; **higher = more anomalous**.

        Combination
        -----------
        Member scores live on incomparable scales: ECOD returns a sum of
        ``-log`` tail probabilities (unbounded above, heavy right tail) while
        IsolationForest returns a path-length statistic in a narrow band
        around 0.5. Averaging them directly would let ECOD dominate by units
        alone, which is precisely the "decorative second model" failure mode
        we are removing.

        * ``combine="z"`` (default): each member is standardised by the mean
          and standard deviation of its scores on the **training benign**
          windows, then averaged with ``weights``. Standardisation is affine
          and therefore *preserves resolution in the tail*, which matters: the
          conformal p-value grid is only as fine as the score's ability to
          distinguish extreme windows.
        * ``combine="rank"``: each member is mapped to its empirical quantile
          in the training benign distribution, then averaged. This is
          invariant to any monotone re-scaling of a member and is robust to
          heavy tails, but it **saturates**: every window beyond the training
          maximum maps to 1.0, so all severe attacks receive an identical
          score and their p-values collapse to the floor ``1/(n_cal+1)``.
          Choose it only when member score distributions are so skewed that
          a standard deviation is meaningless.
        """
        order = self._require_fitted()
        Xa = _as_feature_matrix(X, order)
        raw = self._raw_member_scores(Xa)
        w_ecod, w_iforest = self.weights
        return (
            w_ecod * self._normalise("ecod", raw["ecod"])
            + w_iforest * self._normalise("iforest", raw["iforest"])
        )

    def per_feature_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature ECOD contributions, one column per feature.

        How these are computed
        ----------------------
        We always use the internally fitted :class:`_TailProbECOD` rather than
        reaching into pyod's private ``O``/``O_`` attributes, whose presence
        and shape vary across pyod releases. For each row the entry for
        feature ``j`` is ``-log(tail_prob_j)`` using the tail orientation
        (left / right / skewness-selected) that produced that row's winning
        ECOD aggregate. Consequently each row sums *exactly* to that row's raw
        ECOD score, which is returned in the extra column ``ecod_score``.

        These are ECOD contributions only -- they explain the ECOD member, not
        the combined ensemble score. For an attribution of the *combined*
        score use :func:`rr.detect.shapley.exact_shapley` with
        :meth:`score` as ``score_fn``.
        """
        order = self._require_fitted()
        if self._contrib_model is None:
            raise RuntimeError("Detector is not fitted.")
        Xa = _as_feature_matrix(X, order)
        contrib, score = self._contrib_model.contributions(Xa)
        frame = pd.DataFrame(contrib, columns=list(order))
        frame["ecod_score"] = score
        if isinstance(X, pd.DataFrame):
            frame.index = X.index
        return frame

    # ------------------------------------------------------------------ #
    # Conformal calibration
    # ------------------------------------------------------------------ #
    def calibrate(
        self, X_cal: pd.DataFrame, *, append: bool = False
    ) -> "Detector":
        """Store scores of **held-out benign** windows for conformal inference.

        Parameters
        ----------
        X_cal:
            Benign windows **not used for** :meth:`fit`. Reusing training
            windows destroys the guarantee: training scores are optimistically
            low, so the calibration distribution would be shifted and the
            realised false-alarm rate would exceed ``alpha``.
        append:
            When ``True`` the new scores are appended to the existing
            calibration set (respecting ``sliding_calibration``) instead of
            replacing it. This is how an online deployment refreshes its
            reference distribution.
        """
        self._require_fitted()
        new_scores = np.asarray(self.score(X_cal), dtype=float).ravel()
        if append and self._cal_scores_recent is not None:
            recent = np.concatenate([self._cal_scores_recent, new_scores])
        else:
            recent = new_scores
        if self.sliding_calibration is not None:
            recent = recent[-self.sliding_calibration :]
        if recent.size < 2:
            raise ValueError(
                "Calibration needs at least 2 benign windows; "
                f"got {recent.size}."
            )
        self._cal_scores_recent = recent
        self._cal_scores_sorted = np.sort(recent)
        LOGGER.info(
            "Detector.calibrate: n_cal=%d (sliding=%s), p-value resolution=%.3g",
            recent.size,
            self.sliding_calibration,
            1.0 / (recent.size + 1.0),
        )
        return self

    def pvalue(self, X: pd.DataFrame) -> np.ndarray:
        """Split-conformal p-values for each window. Lower = more anomalous.

        Definition
        ----------
        With ``n_cal`` calibration scores ``s_1..s_{n_cal}`` from held-out
        benign windows and a new score ``s``::

            p = (1 + #{i : s_i >= s}) / (n_cal + 1)

        computed with a sorted calibration array and :func:`numpy.searchsorted`
        (``O(log n_cal)`` per query).

        Guarantee
        ---------
        If the new window is *exchangeable* with the calibration windows --
        i.e. it is drawn from the same benign distribution -- then ``p`` is
        **super-uniform**: ``P(p <= alpha) <= alpha`` for every
        ``alpha in (0, 1)``. Hence thresholding at ``alpha`` bounds the
        **per-window** false-alarm probability by ``alpha``. The bound is
        *distribution-free* (no Gaussianity, no parametric tail model),
        *finite-sample* (not asymptotic), valid for **any** underlying score
        function, and requires **no attack labels** -- only benign data.

        Assumption (and how it breaks)
        ------------------------------
        Exchangeability of the calibration windows with the test window. This
        is the load-bearing assumption and it is *not* free:

        * Filesystem activity is non-stationary. A nightly backup, a package
          install, or a source-tree checkout produces benign windows that are
          not exchangeable with an idle-desktop calibration set, and the
          realised false-alarm rate will exceed ``alpha`` during those
          episodes. Mitigation: ``sliding_calibration`` plus periodic
          ``calibrate(..., append=True)``.
        * Windows overlap with a 1 s hop, so consecutive windows are strongly
          dependent. Exchangeability concerns the *joint* distribution, and
          marginal validity of each individual p-value still holds; what
          dependence breaks is the *independence* assumed by multi-window
          combiners (see :meth:`decide`) and by naive "expected alerts per
          day" arithmetic.
        * The p-value is discrete with atoms at multiples of
          ``1/(n_cal+1)``; ``alpha`` below that floor can never be attained.

        Use :meth:`pvalue_uniformity` to verify the guarantee empirically
        rather than trusting this docstring.
        """
        if self._cal_scores_sorted is None:
            raise RuntimeError(
                "Detector is not calibrated; call calibrate(X_cal) with "
                "held-out benign windows first."
            )
        scores = np.asarray(self.score(X), dtype=float).ravel()
        cal = self._cal_scores_sorted
        n_cal = cal.size
        # #{cal_i >= s} = n_cal - (index of first element >= s)
        ge_count = n_cal - np.searchsorted(cal, scores, side="left")
        return (1.0 + ge_count) / (n_cal + 1.0)

    def pvalue_uniformity(
        self, X_benign: pd.DataFrame, *, bins: int = 10
    ) -> dict[str, Any]:
        """Empirically check that benign p-values are (super-)uniform.

        Returns a dict with the one-sample Kolmogorov-Smirnov statistic and
        p-value against ``Uniform(0, 1)``, histogram counts, and the realised
        exceedance rates at conventional ``alpha`` levels.

        Interpretation
        --------------
        * ``ks_pvalue`` small (< 0.01) with a *right*-shifted histogram is
          benign-conservative; with a *left*-shifted histogram it means the
          guarantee is being violated -- almost always because the evaluation
          windows are not exchangeable with the calibration windows.
        * Conformal p-values are discrete (atoms at ``j/(n_cal+1)``), so the
          continuous KS test is mildly conservative here; treat ``ks_pvalue``
          as a diagnostic, not a hypothesis test to be reported as-is.
        * Because windows overlap, the effective sample size is far below the
          row count, so the KS test is *optimistic* about significance. The
          realised ``rate_at_alpha`` values are the more honest summary.
        """
        p = self.pvalue(X_benign)
        ks = stats.kstest(p, "uniform")
        counts, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
        rates = {
            f"rate_at_{a}": float(np.mean(p <= a))
            for a in (0.001, 0.01, 0.05, 0.1)
        }
        return {
            "n": int(p.size),
            "n_calibration": self.n_calibration,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "mean_pvalue": float(np.mean(p)),
            "min_pvalue": float(np.min(p)),
            "histogram_counts": counts.astype(int).tolist(),
            "histogram_edges": edges.astype(float).tolist(),
            **rates,
        }

    # ------------------------------------------------------------------ #
    # Sequential decision rule
    # ------------------------------------------------------------------ #
    @staticmethod
    def decide(
        pvalues: Iterable[float],
        alpha: float = 0.01,
        k: int = 3,
        *,
        method: SequentialMethod = "hmp",
    ) -> np.ndarray:
        """Fire on **sustained** evidence across ``k`` consecutive windows.

        Parameters
        ----------
        pvalues:
            Conformal p-values **in time order** for a single session/scale.
            Do not interleave sessions or window scales: the combiner assumes
            consecutive elements are consecutive windows of one stream.
        alpha:
            Threshold applied to the *combined* p-value of the trailing
            ``k``-window block.
        k:
            Block length. ``k = 1`` reduces to per-window thresholding.
        method:
            ``"hmp"`` -- Wilson's harmonic mean p-value,
            ``HMP = k / sum_i (1 / p_i)``, fired when ``HMP <= alpha``.
            ``"fisher"`` -- ``-2 sum_i log p_i`` compared to
            ``chi2`` with ``2k`` degrees of freedom.

        Returns
        -------
        numpy.ndarray of bool
            Same length as the input. Element ``t`` is ``True`` iff the block
            ``[t-k+1, t]`` triggers.

        Why this is an *early decision* rule, not a classifier
        ------------------------------------------------------
        A per-window bound of ``alpha`` is not an operational false-alarm
        rate. With a 1 s hop, ``alpha = 0.01`` permits ~864 alerts per day
        *by design*, and ~8.6/day even at non-overlapping 10 s windows. That
        is unusable. Requiring ``k`` consecutive windows of evidence converts
        the detector into a rule that answers "how long until we are
        sufficiently sure?", trading **detection latency** for **alert
        volume**. Sweeping ``alpha`` (and ``k``) therefore traces a
        latency-versus-false-alarm curve, which is the correct evaluation
        object -- AGENT-C should report that curve rather than a single
        precision/recall pair at an arbitrary operating point.

        Warm-up: the first ``k-1`` windows
        ----------------------------------
        They are returned as ``False``, unconditionally and explicitly. The
        alternatives were considered and rejected: padding with ``p = 1``
        silently weakens the first real blocks (and is not even neutral -- it
        adds ``1`` to the HMP denominator sum and ``0`` to Fisher's statistic
        while still inflating its degrees of freedom), while shrinking ``k``
        near the boundary changes the null distribution exactly where the
        stream is least understood. Refusing to decide costs at most ``k-1``
        windows of latency once per session, and every session starts benign.

        Calibration caveats (read before quoting a false-alarm rate)
        ------------------------------------------------------------
        * **Fisher's method is exact only under independence.** Overlapping
          windows (1 s hop over 10 s/60 s scales) are strongly positively
          dependent, so the chi-square reference distribution is
          **anti-conservative**: realised block-level false alarms will exceed
          ``alpha``. Use it when windows are effectively non-overlapping, or
          treat ``alpha`` as a tuning knob calibrated on benign data.
        * **HMP is robust to dependence but only asymptotically calibrated.**
          Wilson shows the HMP is approximately valid under arbitrary
          dependence for small ``alpha``, but it is mildly anti-conservative
          for small ``k`` and moderate ``alpha``; Wilson recommends comparing
          against a slightly shrunken threshold. HMP is the default because
          dependence, not exactness, is the dominant error source here.
        * **Both combiners are dominated by their smallest p-value.** A single
          overwhelmingly extreme window (``p`` at the ``1/(n_cal+1)`` floor)
          can fire a block on its own. That is intended -- a burst of 5000
          encrypting writes in one second *should* alert -- but it means ``k``
          aggregates evidence rather than imposing a hard persistence filter.
        * The honest procedure is to pick ``(alpha, k)`` on a benign
          hold-out so the *empirical* alert rate matches the operations
          budget, then report attack detection latency at that setting.
        """
        p = np.asarray(list(pvalues), dtype=float).ravel()
        if k < 1:
            raise ValueError("k must be >= 1.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1).")
        if method not in ("hmp", "fisher"):
            raise ValueError("method must be 'hmp' or 'fisher'.")
        out = np.zeros(p.size, dtype=bool)
        if p.size == 0:
            return out
        if np.any(p < 0.0) or np.any(p > 1.0):
            raise ValueError("p-values must lie in [0, 1].")
        if p.size < k:
            return out

        safe = np.clip(p, np.finfo(float).tiny, 1.0)
        blocks = np.lib.stride_tricks.sliding_window_view(safe, k)
        if method == "hmp":
            combined = k / np.sum(1.0 / blocks, axis=1)
        else:
            statistic = -2.0 * np.sum(np.log(blocks), axis=1)
            combined = stats.chi2.sf(statistic, df=2 * k)
        out[k - 1 :] = combined <= alpha
        return out

    def combined_pvalues(
        self,
        pvalues: Iterable[float],
        k: int = 3,
        *,
        method: SequentialMethod = "hmp",
    ) -> np.ndarray:
        """Block-combined p-values (``NaN`` for the first ``k-1`` warm-up windows).

        Exposed so AGENT-C can sweep ``alpha`` without recomputing the
        combination for every threshold.
        """
        p = np.asarray(list(pvalues), dtype=float).ravel()
        if k < 1:
            raise ValueError("k must be >= 1.")
        out = np.full(p.size, np.nan, dtype=float)
        if p.size < k:
            return out
        safe = np.clip(p, np.finfo(float).tiny, 1.0)
        blocks = np.lib.stride_tricks.sliding_window_view(safe, k)
        if method == "hmp":
            out[k - 1 :] = k / np.sum(1.0 / blocks, axis=1)
        elif method == "fisher":
            statistic = -2.0 * np.sum(np.log(blocks), axis=1)
            out[k - 1 :] = stats.chi2.sf(statistic, df=2 * k)
        else:
            raise ValueError("method must be 'hmp' or 'fisher'.")
        return out

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> str:
        """Persist the fitted detector with :mod:`joblib`.

        Stores both fitted members, the normalisation statistics, the
        calibration array, the ``FEATURE_ORDER`` captured **at fit time**, and
        a schema version.
        """
        order = self._require_fitted()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "feature_order": list(order),
            "combine": self.combine,
            "weights": self.weights,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "sliding_calibration": self.sliding_calibration,
            "random_state": self.random_state,
            "ecod_backend": self.ecod_backend_,
            "ecod": self._ecod,
            "contrib_model": self._contrib_model,
            "iforest": self._iforest,
            "norm": self._norm,
            "cal_scores_recent": self._cal_scores_recent,
        }
        joblib.dump(payload, path)
        LOGGER.info("Detector.save: wrote schema v%d to %s", SCHEMA_VERSION, path)
        return path

    @classmethod
    def load(
        cls, path: str, *, feature_order: Sequence[str] | None = None
    ) -> "Detector":
        """Load a detector, refusing to run against a changed feature schema.

        Parameters
        ----------
        feature_order:
            Expected feature list. When ``None`` the current
            ``rr.core.featurize.FEATURE_ORDER`` is used if importable; if it
            is not importable the check is skipped and a warning is logged.

        Raises
        ------
        ValueError
            If the schema version differs, or if the persisted feature list
            no longer matches the expected one. A silent mismatch would feed
            the model mislabelled columns and produce confidently wrong
            p-values, so this is a hard error.
        """
        payload = joblib.load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain a Detector payload.")
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible detector schema: file has v{version}, code "
                f"expects v{SCHEMA_VERSION}. Refit the detector."
            )
        stored_order = list(payload["feature_order"])

        expected: list[str] | None
        if feature_order is not None:
            expected = list(feature_order)
        else:
            try:
                expected = _resolve_feature_order(None)
            except RuntimeError:
                expected = None
                LOGGER.warning(
                    "Detector.load: rr.core.featurize unavailable; skipping "
                    "feature-order verification."
                )
        if expected is not None and expected != stored_order:
            only_stored = [f for f in stored_order if f not in expected]
            only_expected = [f for f in expected if f not in stored_order]
            raise ValueError(
                "Feature schema mismatch: the persisted detector was fitted "
                f"on {len(stored_order)} features and the current "
                f"FEATURE_ORDER has {len(expected)}. "
                f"Removed/unknown: {only_stored}; new/missing: {only_expected}"
                + (
                    "; same members but different order."
                    if not only_stored and not only_expected
                    else ""
                )
                + " Refit the detector against the current featuriser."
            )

        obj = cls(
            feature_order=stored_order,
            combine=payload["combine"],
            weights=tuple(payload["weights"]),  # type: ignore[arg-type]
            n_estimators=payload["n_estimators"],
            max_samples=payload["max_samples"],
            sliding_calibration=payload["sliding_calibration"],
            random_state=payload["random_state"],
        )
        obj.feature_order_ = stored_order
        obj.ecod_backend_ = payload["ecod_backend"]
        obj._ecod = payload["ecod"]
        obj._contrib_model = payload["contrib_model"]
        obj._iforest = payload["iforest"]
        obj._norm = payload["norm"]
        recent = payload.get("cal_scores_recent")
        if recent is not None:
            obj._cal_scores_recent = np.asarray(recent, dtype=float)
            obj._cal_scores_sorted = np.sort(obj._cal_scores_recent)
        return obj

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self.is_fitted else "unfitted"
        return (
            f"Detector({state}, backend={self.ecod_backend_}, "
            f"combine={self.combine}, weights={self.weights}, "
            f"n_cal={self.n_calibration})"
        )

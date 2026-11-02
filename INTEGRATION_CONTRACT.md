# RansomRadar v2 — Integration Contract (frozen, v1)

All build agents code against this. Nobody changes it unilaterally.

## Ground rules
- Python 3.11+. Allowed deps ONLY: numpy, pandas, scipy, scikit-learn, pyod, matplotlib, joblib, streamlit, watchdog, pytest.
- New code lives under `rr/`. The existing top-level modules (config.py, db.py, collector.py,
  features.py, model.py, rules.py, forensics.py, app.py, experiment.py, simulator.py, demo.py,
  notify.py) are FROZEN. Do not edit them. Legacy interop goes through `rr/compat.py` (integrator-owned).
- No I/O or network at import time. No network anywhere in tests.
- Must import and pass tests on Linux. Windows-only code sits behind `platform.system() == "Windows"`
  with a documented no-op fallback.
- Deterministic: every stochastic API takes `seed`.
- Each module ships a `if __name__ == "__main__":` smoke demo and a pytest file under `tests/`.

## Shared types — `rr/core/schema.py` (owner: AGENT-A, imported by everyone)
```python
@dataclass(frozen=True)
class NormalisedEvent:
    ts: float                      # unix seconds
    kind: str                      # "created" | "modified" | "deleted" | "moved"
    path: str
    dest_path: str | None
    ext: str                       # lowercase with dot, "" if none
    dest_ext: str | None
    size_delta: int | None
    entropy: float | None          # bits/byte, 0..8
    chi2: float | None             # chi-square uniformity stat of the 4KB sample
    divergent: bool
    note_like: bool
    pid: int | None
    process_guid: str | None
    session_id: str | None

@dataclass(frozen=True)
class Window:
    start: float
    end: float
    scale: float
    events: tuple[NormalisedEvent, ...]
```

## Exports by owner

### AGENT-A — streaming core
- `rr.core.schema`: `NormalisedEvent`, `Window`, `from_legacy_row(row: dict) -> NormalisedEvent`
- `rr.core.forensics`: `chi_square_uniformity(sample: bytes) -> float`
- `rr.core.windows`: `sliding_windows(events, size: float, hop: float, fill_idle: bool = True) -> Iterator[Window]`
- `rr.core.featurize`: `FEATURE_ORDER: list[str]`, `featurize_window(w: Window) -> dict[str, float]`,
  `featurize(events, scales=(10.0, 60.0), hop=1.0) -> pd.DataFrame`
- featurize DataFrame columns == `FEATURE_ORDER + ["window_start", "window_end", "scale", "session_id"]`

### AGENT-B — detection
- `rr.detect.detector.Detector`: `fit(X)`, `calibrate(X_cal)`, `score(X) -> np.ndarray`,
  `pvalue(X) -> np.ndarray`, `decide(pvalues, alpha=0.01, k=3) -> np.ndarray[bool]`,
  `save(path)`, `load(path)` (classmethod)
- `rr.detect.baselines`: `EnvelopeBaseline`, `WritesThresholdBaseline`, `RandomScorer`
  — each with `fit(X)`, `score(X)`, `predict(X)`
- `rr.detect.shapley`: `exact_shapley(score_fn, x, background) -> np.ndarray`

### AGENT-C — evaluation
- `rr.eval.protocol`: `temporal_split(df, frac)`, `session_group_kfold(df, n_splits)`,
  `bootstrap_ci(values, stat, n=1000, seed=0)`
- `rr.eval.metrics`: `files_lost_before_detection(...)`, `time_to_detect(...)`,
  `false_alarms_per_hour(...)`, `pr_auc(y_true, scores)`
- `rr.eval.figures`: `pareto_curve(points, out_png)`
- `rr.eval.ablation`: `leave_one_feature_out(...)`

### AGENT-D — adversary, workloads, ground truth
- `rr.sim.manifest`: `build_manifest(root) -> dict[str, str]`, `verify_manifest(root, manifest) -> VerifyResult`
- `rr.sim.evader`: `EvaderConfig(rate_hz, sleep_jitter, encrypt_fraction, in_place, rename_policy,
  note_timing, decoy_avoidance, n_processes, seed)`, `run_evader(root, cfg) -> AttackTrace`
- `rr.sim.benign_workloads`: `WORKLOADS: dict[str, Callable[[Path], None]]`

### AGENT-E — response, rollback, notification
- `rr.respond.shadow.ShadowJournal(root, cap_bytes)`: `snapshot(path)`, `restore_all() -> RestoreResult`, `usage_bytes()`
- `rr.respond.ladder.ResponseLadder(policy)`: `handle(verdict) -> list[ActionRecord]`, tiers T0..T4,
  `dry_run=True` default, never-kill allowlist, TTL auto-reversal, append-only JSONL audit log
- `rr.notify.push`: `send(topic, title, message, dry_run=True)`

## Direction of dependency (no cycles)
`core` <- `detect` <- `eval`;  `core` <- `sim`;  `core` <- `respond`.
`core` imports nothing from the others. Nobody imports `rr.dashboard`.

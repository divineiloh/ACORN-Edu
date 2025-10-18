# acorn/scheduler_priority.py
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

@dataclass
class Candidate:
    size_kb: float
    deadline_ts: float
    reuse_score: float

def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)

def compute_acorn_scores(
    candidates: Sequence[Candidate],
    *,
    now_ts: float,
    next_window_sec: float,
    predicted_throughput_kBps: float,   # MUST be KB/s
    # NEW default weights: (alpha, beta, gamma, delta)
    weights: Tuple[float, float, float, float] = (0.30, 0.30, 0.30, 0.10),
    ablation: Optional[str] = None,
):
    """
    Scores higher = schedule earlier. All component terms are in [0,1], higher-is-better.
    alpha: deadline/slack, beta: reuse, gamma: inverse size, delta: connectivity/finishability
    """
    n = len(candidates)
    if n == 0:
        return np.array([]), np.array([])

    size_kb   = np.array([c.size_kb for c in candidates], dtype=float)
    reuse_raw = np.array([c.reuse_score for c in candidates], dtype=float)
    deadlines = np.array([c.deadline_ts for c in candidates], dtype=float)

    # ---- Robustness: enforce KB/s and sane floor ----
    th = float(predicted_throughput_kBps)
    if th <= 0:
        th = 1e-6  # guard, not expected
    # Predicted download time for this window's link
    dl_time_sec = size_kb / th

    # ---- Slack with CAP + DEAD-ZONE (prevents over-rewarding razor-thin slack) ----
    # slack_sec = time left after finishing download; negative => impossible
    slack_sec = (deadlines - now_ts) - dl_time_sec
    slack_nonneg = np.clip(slack_sec, 0.0, None)
    # Cap slack to the window length so "very early" items don't crowd out urgent-but-feasible
    S_capped = np.clip(slack_nonneg, 0.0, next_window_sec)
    # Dead-zone: treat slack beyond 25% of the window as equivalent (saturates the benefit)
    deadzone = 0.25 * next_window_sec
    S_eff = np.clip(S_capped, 0.0, deadzone)
    slack_term = 1.0 - _norm01(S_eff)  # smaller (but nonnegative) slack ⇒ higher priority

    # ---- Finishability-oriented connectivity term (bounded, monotone, per-item) ----
    # Fraction of the upcoming window needed to fetch this item at predicted throughput.
    # 0 => trivial to finish; 1 => exactly fills the window; >1 will be gated as infeasible anyway.
    window_capacity_kb = th * next_window_sec
    frac_of_window = np.clip(size_kb / (window_capacity_kb + 1e-9), 0.0, 1.0)
    conn_term = 1.0 - _norm01(frac_of_window)  # higher means easier to finish within window

    # ---- Other terms ----
    size_term  = 1.0 - _norm01(size_kb)         # smaller is better
    reuse_term = _norm01(reuse_raw)             # higher reuse is better

    # ---- Hard feasibility gate (kept) ----
    feasible = (slack_sec >= 0.0) & (dl_time_sec <= next_window_sec)

    alpha, beta, gamma, delta = weights
    if ablation == "alpha0": alpha = 0.0
    if ablation == "beta0":  beta  = 0.0
    if ablation == "gamma0": gamma = 0.0
    if ablation == "delta0": delta = 0.0

    # Weighted sum (interpretable, low-compute)
    scores = alpha*slack_term + beta*reuse_term + gamma*size_term + delta*conn_term
    scores = np.where(feasible, scores, 0.0)

    # Stable tie-breakers: higher reuse, then smaller size
    sort_idx = np.lexsort((size_kb, -reuse_term, scores))
    return scores, sort_idx

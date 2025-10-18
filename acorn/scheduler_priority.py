# acorn/scheduler_priority.py
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

@dataclass
class Candidate:
    size_kb: float            # KB
    deadline_ts: float        # POSIX seconds
    reuse_score: float        # raw or [0,1]

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
    predicted_throughput_kBps: float,   # KB/s (not kb/s)
    weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),  # (alpha, beta, gamma, delta)
    ablation: Optional[str] = None      # {"alpha0","beta0","gamma0","delta0"} or None
):
    if len(candidates) == 0:
        return np.array([]), np.array([])

    size_kb   = np.array([c.size_kb for c in candidates], dtype=float)
    reuse_raw = np.array([c.reuse_score for c in candidates], dtype=float)
    deadlines = np.array([c.deadline_ts for c in candidates], dtype=float)

    th = max(float(predicted_throughput_kBps), 1e-6)  # guard
    dl_time_sec = size_kb / th

    # Slack (secs) = time left after finishing download; negative => impossible by deadline
    slack_sec = (deadlines - now_ts) - dl_time_sec

    # Feasible if finishable in the *next* window and not already late
    feasible = (slack_sec >= 0.0) & (dl_time_sec <= next_window_sec)

    # Batch-normalized terms — all ∈ [0,1], all "higher is better"
    size_term   = 1.0 - _norm01(size_kb)                 # smaller size → higher term
    reuse_term  = _norm01(reuse_raw)                     # higher reuse → higher term
    conn_term   = 1.0                                    # keep as scalar; replace with per-candidate array if available
    slack_term  = 1.0 - _norm01(np.where(slack_sec < 0.0, 0.0, slack_sec))  # smaller nonneg slack → higher term

    alpha, beta, gamma, delta = weights
    if ablation == "alpha0": alpha = 0.0
    if ablation == "beta0":  beta  = 0.0
    if ablation == "gamma0": gamma = 0.0
    if ablation == "delta0": delta = 0.0

    scores = alpha*slack_term + beta*reuse_term + gamma*size_term + delta*conn_term
    scores = np.where(feasible, scores, 0.0)  # hard gate

    # Stable tiebreakers: higher reuse, then smaller size
    sort_idx = np.lexsort((size_kb, -reuse_term, scores))
    return scores, sort_idx

# acorn.py — single-file ACORN-Edu harness
# Run: python acorn.py
# Requires: numpy, pandas, scipy, matplotlib

from __future__ import annotations
import json, math, os, random, time, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Human-friendly labels
SCENARIO_LABEL = {
    "nightly_wifi": "Nightly Wi-Fi Window",
    "spotty_cellular": "Bursty Cellular",
}
POLICY_LABEL = {
    "acorn": "Acorn (Slack-Aware)",
    "LRU_whole": "Whole-File LRU",
}
ABLATION_LABEL = {
    "full": "Full (α,β,γ,δ)",
    "alpha0": "No-Deadline (α=0)",
    "beta0":  "No-Reuse (β=0)",
    "gamma0": "No-Size (γ=0)",
    "delta0": "No-Connectivity (δ=0)",
}

def label_scenario(x: str) -> str: return SCENARIO_LABEL.get(x, x)
def label_policy(x: str) -> str:   return POLICY_LABEL.get(x, x)
def label_ablation(x: str) -> str: return ABLATION_LABEL.get(x, x)
plt.rcParams["font.size"] = 14  # enforce readable fonts across all plots
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
# unified errorbar styling (consistent across all figures)
ERR_KW = dict(capsize=6)
BAR_EDGE_KW = dict(edgecolor="black", linewidth=0.6)

# -------------------- CONFIG --------------------
RNG_SEED_BASE = 1337
N_TRIALS = 30
OUT_DIR = Path("results")
OUT_DATA = OUT_DIR / "data"
OUT_FIGS = OUT_DIR / "figures"
OUT_ABLATION = OUT_DIR / "ablation"
OUT_DIR.mkdir(exist_ok=True)
OUT_DATA.mkdir(exist_ok=True)
OUT_FIGS.mkdir(exist_ok=True)
OUT_ABLATION.mkdir(exist_ok=True)

# policy weights (defaults) - normalized weights for slack-aware scheduler
# Scenario-specific weights for better performance
WEIGHTS = {
    "nightly_wifi": {"alpha": 0.35, "beta": 0.25, "gamma": 0.20, "delta": 0.20},
    "spotty_cellular": {"alpha": 0.40, "beta": 0.20, "gamma": 0.30, "delta": 0.10}  # Emphasize deadline and size in scarcity
}

# Default weights (fallback)
ALPHA = 0.35   # slack term (deadline urgency)
BETA  = 0.25   # reuse score
GAMMA = 0.20   # inverse size
DELTA = 0.20   # predicted availability

# scenarios (simple but meaningfully different)
SCENARIOS = {
    "nightly_wifi": dict(n_windows=1,   min_sec=900, max_sec=900,   min_kbps=8000, max_kbps=8000),
    "spotty_cellular": dict(n_windows=8,min_sec=60,  max_sec=120,   min_kbps=150,  max_kbps=400),
}

# asset model
N_ASSETS = 120
MIN_SIZE_KB, MAX_SIZE_KB = 512, 20000  # 0.5–20 MB
# Deadlines should be AFTER connectivity windows to make sense
# For nightly_wifi: windows are ~7 hours, so deadlines should be 8-24 hours later
# For spotty_cellular: windows are scattered, so deadlines should be 1-48 hours later
MIN_DEADLINE_S, MAX_DEADLINE_S = 8*3_600, 48*3_600  # 8 hours to 48 hours
REUSE_PROB = 0.5  # fraction of chunks expected reusable

# plotting
DPI = 300
FONT = 12

# -------------------- MODELS --------------------
@dataclass
class Asset:
    aid: int
    size_kb: float
    deadline_s: float
    reuse_score: float  # [0,1]

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

def gen_assets(rng: random.Random) -> List[Asset]:
    assets = []
    for i in range(N_ASSETS):
        size_kb = rng.uniform(MIN_SIZE_KB, MAX_SIZE_KB)
        deadline = rng.uniform(MIN_DEADLINE_S, MAX_DEADLINE_S)
        reuse = 1.0 if rng.random() < REUSE_PROB else rng.random()*0.3
        assets.append(Asset(i, size_kb, deadline, reuse))
    return assets

def gen_windows(scfg: dict, rng: random.Random, horizon_s=24*3600) -> List[Tuple[int,int,int]]:
    # returns list of (start_s, end_s, kbps)
    wins = []
    for _ in range(scfg["n_windows"]):
        dur = rng.randint(scfg["min_sec"], scfg["max_sec"])
        start = rng.randint(0, horizon_s - dur)
        kbps = rng.randint(scfg["min_kbps"], scfg["max_kbps"])
        wins.append((start, start+dur, kbps))
    wins.sort(key=lambda x: x[0])
    return wins

def availability_predictor(windows: List[Tuple[int,int,int]], t: int) -> float:
    # crude: likelihood of being inside/near a window in next chunk of time
    for s,e,_ in windows:
        if s <= t <= e: return 1.0
        if 0 <= s - t <= 600: return 0.6
    return 0.1

# -------------------- POLICIES --------------------
def prio_acorn(asset: Asset, t: int, windows, w, ablation: Optional[str] = None) -> float:
    # Use the new slack-aware, normalized priority function
    candidates = [Candidate(asset.size_kb, asset.deadline_s, asset.reuse_score)]
    
    # Get current window info for throughput prediction
    current_window = None
    for s, e, kbps in windows:
        if s <= t <= e:
            current_window = (s, e, kbps)
            break
    
    if current_window is None:
        # No current window, use minimal throughput
        predicted_throughput_kBps = 1.0
        next_window_sec = 3600  # 1 hour fallback
    else:
        s, e, kbps = current_window
        predicted_throughput_kBps = kbps / 8.0  # Convert kbps to KB/s
        next_window_sec = e - t
    
    weights = (w["alpha"], w["beta"], w["gamma"], w["delta"])
    scores, _ = compute_acorn_scores(
        candidates,
        now_ts=t,
        next_window_sec=next_window_sec,
        predicted_throughput_kBps=predicted_throughput_kBps,
        weights=weights,
        ablation=ablation
    )
    
    return scores[0] if len(scores) > 0 else 0.0

def policy_acorn(assets: List[Asset], windows, rng: random.Random, weights, ablation: Optional[str] = None) -> Tuple[float,float]:
    # returns (bytes_kb, hit_rate)
    t = 0; downloaded = set(); bytes_kb = 0.0; hits = 0
    for s,e,kbps in windows:
        t = s
        while t < e and len(downloaded) < len(assets):
            remaining = [a for a in assets if a.aid not in downloaded]
            if not remaining: break
            best = max(remaining, key=lambda a: prio_acorn(a, t, windows, weights, ablation))
            budget_kb = (e - t) * (kbps / 8.0)  # Convert kbps to KB/s
            if budget_kb <= 0: break
            # crude delta-sync effect: less to transfer when reuse_score is high
            use_kb = min(budget_kb, best.size_kb * (1.0 - 0.4*best.reuse_score))
            bytes_kb += use_kb
            t += max(1, int(use_kb / max(kbps/8.0,1)))  # Convert kbps to KB/s
            downloaded.add(best.aid)
            if t <= best.deadline_s: hits += 1
    hit_rate = hits / max(1,len(assets))
    return bytes_kb, hit_rate

def policy_lru_whole(assets: List[Asset], windows, rng: random.Random) -> Tuple[float,float]:
    # whole-asset downloads, deadline-ordered queue, with simple LRU cache
    t = 0; downloaded = set(); cache: List[int] = []; CACHE_CAP = 128
    bytes_kb = 0.0; hits = 0
    q = sorted(assets, key=lambda a: a.deadline_s)
    
    # Track partial downloads across windows
    partial_downloads = {}  # asset_id -> bytes_downloaded
    
    for s,e,kbps in windows:
        t = s
        qi = 0
        window_budget = (e - s) * (kbps / 8.0)  # Total budget for this window
        window_used = 0.0
        
        while t < e and qi < len(q) and window_used < window_budget:
            a = q[qi]
            
            # Check if already cached
            if a.aid in cache:
                if t <= a.deadline_s: hits += 1
                qi += 1
                continue
            
            # Calculate remaining budget for this asset
            remaining_budget = window_budget - window_used
            need_kb = a.size_kb
            
            # Check if we have partial download from previous windows
            already_downloaded = partial_downloads.get(a.aid, 0)
            still_needed = need_kb - already_downloaded
            
            if still_needed <= 0:
                # Already fully downloaded
                cache.append(a.aid)
                if len(cache) > CACHE_CAP: cache.pop(0)
                downloaded.add(a.aid)
                if t <= a.deadline_s: hits += 1
                qi += 1
                continue
            
            # Try to download what we still need
            use_kb = min(remaining_budget, still_needed)
            
            if use_kb > 0:
                bytes_kb += use_kb
                window_used += use_kb
                # Update time based on actual download time
                download_time = use_kb / max(kbps/8.0, 1)  # Time in seconds
                t += max(1, int(download_time))
                
                # Update partial download tracking
                partial_downloads[a.aid] = already_downloaded + use_kb
                
                # Check if we completed the asset
                if partial_downloads[a.aid] >= need_kb:
                    cache.append(a.aid)
                    if len(cache) > CACHE_CAP: cache.pop(0)
                    downloaded.add(a.aid)
                    if t <= a.deadline_s: hits += 1
                    # Remove from partial tracking since it's complete
                    del partial_downloads[a.aid]
                    qi += 1
                else:
                    # Still partial - continue with next asset
                    qi += 1
            else:
                # No budget left
                break
    
    hit_rate = hits / max(1,len(assets))
    return bytes_kb, hit_rate

# -------------------- RUNNERS --------------------
def run_trial(scenario_name: str, policy: str, weights: Dict[str,float], trial_seed: int, ablation: Optional[str] = None) -> Dict:
    rng = random.Random(trial_seed)
    assets = gen_assets(rng)
    windows = gen_windows(SCENARIOS[scenario_name], rng)
    if policy == "acorn":
        bkb, hr = policy_acorn(assets, windows, rng, weights, ablation)
    elif policy == "LRU_whole":
        bkb, hr = policy_lru_whole(assets, windows, rng)
    else:
        raise ValueError(policy)
    return dict(scenario=scenario_name, policy=policy, trial=trial_seed, bytes_kb=bkb, hit_rate=hr, seed=trial_seed, ablation=ablation)

def t_ci(series: np.ndarray, alpha=0.05) -> float:
    n = len(series)
    if n < 2: return float("nan")
    m = np.mean(series); s = np.std(series, ddof=1)
    tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
    half = tcrit * s / math.sqrt(n)
    return half

def paired_t_test(values1, values2):
    """Paired t-test for comparing two sets of values"""
    if len(values1) != len(values2):
        raise ValueError("Values must be paired (same length)")
    t_stat, p_value = stats.ttest_rel(values1, values2)
    # Effect size (Cohen's d for paired samples)
    diff = np.array(values1) - np.array(values2)
    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    return t_stat, p_value, cohens_d

def run_all():
    # clean up old results
    import glob
    for pattern in ["data/*.csv", "data/*.json", "figures/*.png"]:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass
    
    # base + ablations - use normalized weights
    ablations = {
        "full":   dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA),
        "alpha0": dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA),  # Will be handled by ablation parameter
        "beta0":  dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA),  # Will be handled by ablation parameter
        "gamma0": dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA),  # Will be handled by ablation parameter
        "delta0": dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA),  # Will be handled by ablation parameter
    }
    rows, ablrows = [], []
    seeds = [RNG_SEED_BASE + i for i in range(N_TRIALS)]
    for scenario in SCENARIOS.keys():
        # base policies
        for policy in ["acorn", "LRU_whole"]:
            for s in seeds:
                # Use scenario-specific weights for acorn, default for LRU
                weights = WEIGHTS.get(scenario, ablations["full"]) if policy == "acorn" else ablations["full"]
                rows.append(run_trial(scenario, policy, weights, s))
        # ablations (acorn only) - use different seeds for each ablation
        for abl, w in ablations.items():
            # create unique seeds for this ablation by adding a hash of the ablation name
            abl_seeds = [s + hash(abl) % 10000 for s in seeds]
            for s in abl_seeds:
                # Use scenario-specific weights for ablations too
                weights = WEIGHTS.get(scenario, w)
                r = run_trial(scenario, "acorn", weights, s, ablation=abl)
                ablrows.append(r)

    df = pd.DataFrame(rows)
    # round all numeric columns to 1 decimal place
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].round(1)
    df.to_csv(OUT_DATA/"bap_network_scenario_results.csv", index=False)
    
    dfa = pd.DataFrame(ablrows)
    # round all numeric columns to 1 decimal place
    for col in dfa.select_dtypes(include=[np.number]).columns:
        dfa[col] = dfa[col].round(1)
    dfa.to_csv(OUT_DATA/"bap_ablation_study_results.csv", index=False)

    # aggregates + CIs
    agg = df.groupby(["scenario","policy"]).agg(
        mean_bytes_kb=("bytes_kb","mean"),
        mean_hit_rate=("hit_rate","mean"),
        n_trials=("trial","nunique")
    ).reset_index()
    agg["ci95_bytes_kb"] = df.groupby(["scenario","policy"])["bytes_kb"].apply(lambda s: t_ci(s.values)).values
    agg["ci95_hit_rate"] = df.groupby(["scenario","policy"])["hit_rate"].apply(lambda s: t_ci(s.values)).values
    # round all numeric columns to 1 decimal place
    for col in agg.select_dtypes(include=[np.number]).columns:
        agg[col] = agg[col].round(1)
    agg.to_csv(OUT_DATA/"bap_network_scenario_aggregates.csv", index=False)

    aggA = dfa.groupby(["scenario","policy","ablation"]).agg(
        mean_bytes_kb=("bytes_kb","mean"),
        mean_hit_rate=("hit_rate","mean"),
        n_trials=("trial","nunique")
    ).reset_index()
    aggA["ci95_bytes_kb"] = dfa.groupby(["scenario","policy","ablation"])["bytes_kb"].apply(lambda s: t_ci(s.values)).values
    aggA["ci95_hit_rate"] = dfa.groupby(["scenario","policy","ablation"])["hit_rate"].apply(lambda s: t_ci(s.values)).values
    # round all numeric columns to 1 decimal place
    for col in aggA.select_dtypes(include=[np.number]).columns:
        aggA[col] = aggA[col].round(1)
    aggA.to_csv(OUT_DATA/"bap_ablation_study_aggregates.csv", index=False)

    # -------- Generate final figures --------
    # First, compute ablation deltas
    def compute_ablation_deltas():
        agg = pd.read_csv(OUT_DATA/"bap_ablation_study_aggregates.csv")
        out = []
        for scen, grp in agg.groupby("scenario"):
            full = grp[grp["ablation"] == "full"].iloc[0]
            for _, r in grp.iterrows():
                out.append({
                    "scenario": scen,
                    "ablation": r["ablation"],
                    "mean_bytes_kb": r["mean_bytes_kb"],
                    "ci95_bytes_kb": r["ci95_bytes_kb"],
                    "mean_hit_rate": r["mean_hit_rate"],
                    "ci95_hit_rate": r["ci95_hit_rate"],
                    "delta_kb_pct": 100.0 * (r["mean_bytes_kb"] - full["mean_bytes_kb"]) / full["mean_bytes_kb"],
                    "delta_hit_pp": 100.0 * (r["mean_hit_rate"] - full["mean_hit_rate"]),
                })
        pd.DataFrame(out).to_csv(OUT_ABLATION/"ablation_summary_by_scenario.csv", index=False)
    
    compute_ablation_deltas()
    
    # Guardrail: Statistical ablation dominance check (paired t-tests + practical margins)
    def check_ablation_dominance_stats():
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "scripts/assert_ablation_binding_stats.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            raise SystemExit(f"Statistical guardrail FAILED: {result.returncode}")
        print(result.stdout)
    
    check_ablation_dominance_stats()
    
    # Add statistical significance testing for BAP comparisons
    def compute_bap_statistical_tests():
        """Compute paired t-tests for BAP policy comparisons"""
        policy_data = pd.read_csv(OUT_DATA/"bap_network_scenario_results.csv")
        
        # Group by scenario and policy
        results = []
        for scenario in ["nightly_wifi", "spotty_cellular"]:
            for metric in ["bytes_kb", "hit_rate"]:
                # Extract data for each policy
                acorn_data = policy_data[(policy_data["scenario"] == scenario) & 
                                       (policy_data["policy"] == "acorn")][metric].values
                lru_data = policy_data[(policy_data["scenario"] == scenario) & 
                                     (policy_data["policy"] == "LRU_whole")][metric].values
                
                # Paired t-test (trials are paired by seed)
                t_stat, p_value, cohens_d = paired_t_test(acorn_data, lru_data)
                
                # Determine significance
                significance = ""
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                
                results.append({
                    "scenario": scenario,
                    "metric": metric,
                    "comparison": "acorn_vs_lru",
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "significant": p_value < 0.05,
                    "significance_stars": significance
                })
        
        # Save statistical results
        stats_df = pd.DataFrame(results)
        stats_df.to_csv(OUT_DATA/"bap_statistical_tests.csv", index=False)
        
        # Print summary
        print("\n=== BAP Statistical Test Results ===")
        for _, row in stats_df.iterrows():
            print(f"{row['scenario']} {row['metric']}: t={row['t_statistic']:.3f}, "
                  f"p={row['p_value']:.3f}, d={row['cohens_d']:.3f} {row['significance_stars']}")
    
    compute_bap_statistical_tests()
    
    # Generate 8 individual figures for LaTeX paper submission
    def create_final_figures():
        # Load data
        policy_agg = pd.read_csv(OUT_DATA/"bap_network_scenario_aggregates.csv")
        ablation_agg = pd.read_csv(OUT_ABLATION/"ablation_summary_by_scenario.csv")
        
        # Add labels
        policy_agg["scenario_label"] = policy_agg["scenario"].map(label_scenario)
        policy_agg["policy_label"] = policy_agg["policy"].map(label_policy)
        ablation_agg["scenario_label"] = ablation_agg["scenario"].map(label_scenario)
        ablation_agg["ablation_label"] = ablation_agg["ablation"].map(label_ablation)
        
        # Helper function to create two-line labels
        def create_two_line_labels(labels):
            result = []
            for label in labels:
                if "(" in label and ")" in label:
                    parts = label.split("(", 1)
                    if len(parts) == 2:
                        result.append(f"{parts[0].strip()}\n({parts[1]}")
                    else:
                        result.append(label)
                else:
                    result.append(label)
            return result
        
        # 1. BAP KB transferred - Nightly Wi-Fi
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        sub = policy_agg[policy_agg["scenario"] == "nightly_wifi"]
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["mean_bytes_kb"], yerr=sub["ci95_bytes_kb"], **BAR_EDGE_KW, **ERR_KW)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["policy_label"], rotation=0, ha="center")
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.set_ylabel("KB transferred")
        ax.set_title(f"KB transferred — {label_scenario('nightly_wifi')}")
        plt.savefig(OUT_FIGS/"bap_kb_nightly_wifi.png", bbox_inches="tight", dpi=DPI)
        plt.close()
        
        # 2. BAP KB transferred - Bursty Cellular
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        sub = policy_agg[policy_agg["scenario"] == "spotty_cellular"]
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["mean_bytes_kb"], yerr=sub["ci95_bytes_kb"], **BAR_EDGE_KW, **ERR_KW)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["policy_label"], rotation=0, ha="center")
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.set_ylabel("KB transferred")
        ax.set_title(f"KB transferred — {label_scenario('spotty_cellular')}")
        plt.savefig(OUT_FIGS/"bap_kb_spotty_cellular.png", bbox_inches="tight", dpi=DPI)
        plt.close()
        
        # 3. BAP Hit rate - Nightly Wi-Fi
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        sub = policy_agg[policy_agg["scenario"] == "nightly_wifi"]
        x = np.arange(len(sub))
        bars = ax.bar(x, 100*sub["mean_hit_rate"], yerr=100*sub["ci95_hit_rate"], **BAR_EDGE_KW, **ERR_KW)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["policy_label"], rotation=0, ha="center")
        ax.set_ylabel("Prefetch hit-rate (%)")
        ax.set_ylim(0, 110)
        ax.set_title(f"Prefetch hit-rate (%) — {label_scenario('nightly_wifi')}")
        plt.savefig(OUT_FIGS/"bap_hit_nightly_wifi.png", bbox_inches="tight", dpi=DPI)
        plt.close()
        
        # 4. BAP Hit rate - Bursty Cellular
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        sub = policy_agg[policy_agg["scenario"] == "spotty_cellular"]
        x = np.arange(len(sub))
        bars = ax.bar(x, 100*sub["mean_hit_rate"], yerr=100*sub["ci95_hit_rate"], **BAR_EDGE_KW, **ERR_KW)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["policy_label"], rotation=0, ha="center")
        ax.set_ylabel("Prefetch hit-rate (%)")
        ax.set_ylim(0, 110)
        ax.set_title(f"Prefetch hit-rate (%) — {label_scenario('spotty_cellular')}")
        plt.savefig(OUT_FIGS/"bap_hit_spotty_cellular.png", bbox_inches="tight", dpi=DPI)
        plt.close()
        
        # 5. Ablation KB transferred - Nightly Wi-Fi
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        sub = ablation_agg[ablation_agg["scenario"] == "nightly_wifi"]
        order = ["full", "alpha0", "beta0", "gamma0", "delta0"]
        sub = sub.set_index("ablation").loc[order].reset_index()
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["mean_bytes_kb"], yerr=sub["ci95_bytes_kb"], **BAR_EDGE_KW, **ERR_KW)
        ax.set_xticks(x)
        labels = create_two_line_labels(sub["ablation_label"])
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.set_ylabel("KB transferred")
        ax.set_title(f"KB transferred — {label_scenario('nightly_wifi')}")
        max_val = max(sub["mean_bytes_kb"])
        # Add more spacing and adjust y-axis limits
        label_height = max_val * 0.08
        ax.set_ylim(0, max_val + label_height)
        plt.savefig(OUT_FIGS/"ablation_kb_nightly_wifi.png", bbox_inches="tight", dpi=DPI)
        plt.close()
        
        # 6. Ablation KB transferred - Bursty Cellular
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        sub = ablation_agg[ablation_agg["scenario"] == "spotty_cellular"]
        order = ["full", "alpha0", "beta0", "gamma0", "delta0"]
        sub = sub.set_index("ablation").loc[order].reset_index()
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["mean_bytes_kb"], yerr=sub["ci95_bytes_kb"], **BAR_EDGE_KW, **ERR_KW)
        ax.set_xticks(x)
        labels = create_two_line_labels(sub["ablation_label"])
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.set_ylabel("KB transferred")
        ax.set_title(f"KB transferred — {label_scenario('spotty_cellular')}")
        max_val = max(sub["mean_bytes_kb"])
        # Add more spacing and adjust y-axis limits
        label_height = max_val * 0.08
        ax.set_ylim(0, max_val + label_height)
        plt.savefig(OUT_FIGS/"ablation_kb_spotty_cellular.png", bbox_inches="tight", dpi=DPI)
        plt.close()
        
        # 7. Ablation Hit rate - Nightly Wi-Fi
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        sub = ablation_agg[ablation_agg["scenario"] == "nightly_wifi"]
        order = ["full", "alpha0", "beta0", "gamma0", "delta0"]
        sub = sub.set_index("ablation").loc[order].reset_index()
        x = np.arange(len(sub))
        bars = ax.bar(x, 100*sub["mean_hit_rate"], yerr=100*sub["ci95_hit_rate"], **BAR_EDGE_KW, **ERR_KW)
        ax.set_xticks(x)
        labels = create_two_line_labels(sub["ablation_label"])
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
        ax.set_ylabel("Prefetch hit-rate (%)")
        ax.set_ylim(0, 110)
        ax.set_title(f"Prefetch hit-rate (%) — {label_scenario('nightly_wifi')}")
        plt.savefig(OUT_FIGS/"ablation_hit_nightly_wifi.png", bbox_inches="tight", dpi=DPI)
        plt.close()
        
        # 8. Ablation Hit rate - Bursty Cellular
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        sub = ablation_agg[ablation_agg["scenario"] == "spotty_cellular"]
        order = ["full", "alpha0", "beta0", "gamma0", "delta0"]
        sub = sub.set_index("ablation").loc[order].reset_index()
        x = np.arange(len(sub))
        bars = ax.bar(x, 100*sub["mean_hit_rate"], yerr=100*sub["ci95_hit_rate"], **BAR_EDGE_KW, **ERR_KW)
        ax.set_xticks(x)
        labels = create_two_line_labels(sub["ablation_label"])
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
        ax.set_ylabel("Prefetch hit-rate (%)")
        ax.set_ylim(0, 110)
        ax.set_title(f"Prefetch hit-rate (%) — {label_scenario('spotty_cellular')}")
        plt.savefig(OUT_FIGS/"ablation_hit_spotty_cellular.png", bbox_inches="tight", dpi=DPI)
        plt.close()
    
    create_final_figures()

    # metadata
    # obtain current git commit (if available)
    def _git_rev():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            return None

    (OUT_DATA/"run_metadata.json").write_text(json.dumps({
        "commit": _git_rev(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_trials": N_TRIALS,
        "seed_base": RNG_SEED_BASE,
        "scenarios": list(SCENARIOS.keys()),
        "policies": ["acorn","LRU_whole"],
        "ablations": ["alpha0","beta0","gamma0","delta0"]
    }, indent=2))

# -------------------- VERIFY --------------------
def verify():
    # presence: include both per-trial CSVs and aggregate CSVs
    req_csvs = [
        OUT_DATA/"bap_network_scenario_results.csv",
        OUT_DATA/"bap_network_scenario_aggregates.csv",
        OUT_DATA/"bap_ablation_study_results.csv",
        OUT_DATA/"bap_ablation_study_aggregates.csv",
    ]
    req_figs = [
        OUT_FIGS/"bap_kb_nightly_wifi.png",
        OUT_FIGS/"bap_kb_spotty_cellular.png",
        OUT_FIGS/"bap_hit_nightly_wifi.png",
        OUT_FIGS/"bap_hit_spotty_cellular.png",
        OUT_FIGS/"ablation_kb_nightly_wifi.png",
        OUT_FIGS/"ablation_kb_spotty_cellular.png",
        OUT_FIGS/"ablation_hit_nightly_wifi.png",
        OUT_FIGS/"ablation_hit_spotty_cellular.png",
    ]
    miss = [str(p) for p in req_csvs+req_figs if not p.exists()]
    if miss: raise SystemExit("Missing artifacts:\n- "+"\n- ".join(miss))

    df = pd.read_csv(req_csvs[0])
    need = {"scenario","policy","trial","bytes_kb","hit_rate","seed"}
    if not need.issubset(df.columns): raise SystemExit("results.csv missing cols")
    if set(df["policy"].unique()) != {"acorn","LRU_whole"}:
        raise SystemExit("Policy set must be exactly {acorn,LRU_whole}")
    ct = df.groupby(["scenario","policy"])["trial"].nunique()
    if ct.min()!=N_TRIALS or ct.max()!=N_TRIALS:
        raise SystemExit(f"Each scenario×policy must have exactly {N_TRIALS} trials")
    for c in [c for c in df.columns if c.endswith("_kb")]:
        pd.to_numeric(df[c], errors="raise")
        if (df[c] < 0).any(): raise SystemExit(f"Negative KB in {c}")
    if not ((df["hit_rate"]>=0)&(df["hit_rate"]<=1)).all():
        raise SystemExit("hit_rate out of [0,1]")

    # scenario separation guardrail: require >=15% difference between scenarios
    agg = pd.read_csv(OUT_DATA/"bap_network_scenario_aggregates.csv")
    if not {"scenario","policy","mean_bytes_kb","mean_hit_rate"}.issubset(agg.columns):
        raise SystemExit("aggregates.csv missing expected columns")
    if agg["scenario"].nunique() >= 2:
        # compare scenario means (averaged across policies) — must differ by >=15% in KB or hit rate
        sc_means = agg.groupby("scenario").agg(
            mbytes=("mean_bytes_kb","mean"),
            mhit=("mean_hit_rate","mean")
        )
        vals = sc_means.to_dict(orient="index")
        scens = list(vals.keys())
        # check any pair
        def rel_diff(a,b):
            return abs(a-b)/a if a>0 else (float("inf") if b>0 else 0.0)
        ok = False
        for i in range(len(scens)-1):
            a, b = scens[i], scens[i+1]
            kb_delta = rel_diff(vals[a]["mbytes"], vals[b]["mbytes"])
            hr_delta = rel_diff(vals[a]["mhit"], vals[b]["mhit"])
            if max(kb_delta, hr_delta) >= 0.15:
                ok = True; break
        if not ok:
            raise SystemExit("Scenarios too similar: expected >=15% difference in KB or hit-rate between scenarios")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ACORN-Edu single-file harness")
    ap.add_argument("--trials", type=int, default=N_TRIALS, help="override number of trials (default: 30)")
    ap.add_argument("--quick", action="store_true", help="quick test mode: N=3 trials")
    ap.add_argument("--seed-base", type=int, default=RNG_SEED_BASE, help="override base RNG seed (default: 1337)")
    ap.add_argument("--policy", choices=["acorn", "LRU_whole"], default="acorn", help="policy to run")
    ap.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="scenario to run")
    ap.add_argument("--alpha0", action="store_true", help="ablation: remove deadline term")
    ap.add_argument("--beta0", action="store_true", help="ablation: remove reuse term")
    ap.add_argument("--gamma0", action="store_true", help="ablation: remove size term")
    ap.add_argument("--delta0", action="store_true", help="ablation: remove network term")
    args = ap.parse_args()

    # Check for mutually exclusive ablation flags
    ablation_flags = [args.alpha0, args.beta0, args.gamma0, args.delta0]
    if sum(ablation_flags) > 1:
        ap.error("Only one ablation flag can be specified at a time")
    
    # Determine ablation
    ablation = None
    if args.alpha0: ablation = "alpha0"
    elif args.beta0: ablation = "beta0"
    elif args.gamma0: ablation = "gamma0"
    elif args.delta0: ablation = "delta0"

    # override globals for this run
    N_TRIALS = 3 if args.quick else args.trials
    RNG_SEED_BASE = args.seed_base

    # If specific scenario/policy/ablation specified, run single trial
    if args.scenario is not None:
        if args.policy == "LRU_whole" and ablation is not None:
            ap.error("LRU_whole policy does not support ablations")
        
        # Run single trial
        rng = random.Random(RNG_SEED_BASE)
        assets = gen_assets(rng)
        windows = gen_windows(SCENARIOS[args.scenario], rng)
        
        if args.policy == "acorn":
            # Use scenario-specific weights
            weights = WEIGHTS.get(args.scenario, dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA))
            bkb, hr = policy_acorn(assets, windows, rng, weights, ablation)
        elif args.policy == "LRU_whole":
            bkb, hr = policy_lru_whole(assets, windows, rng)
        
        print(f"Scenario: {args.scenario}, Policy: {args.policy}, Ablation: {ablation or 'none'}")
        print(f"Bytes (KB): {bkb:.1f}, Hit Rate: {hr:.3f}")
    else:
        # Run full benchmark
        run_all()
        verify()
        print("OK: data/*.csv + figures/*.png + run_metadata.json written (KB-only; 95% t-CIs)")

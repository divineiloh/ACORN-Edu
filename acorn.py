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

# Import the new slack-aware priority function
from acorn.scheduler_priority import compute_acorn_scores, Candidate
# Import human-friendly labels
from utils.labels import label_scenario, label_policy, label_ablation
plt.rcParams["font.size"] = 12  # enforce readable fonts across all plots
# unified errorbar styling (consistent across all figures)
ERR_KW = dict(capsize=6)
BAR_EDGE_KW = dict(edgecolor="black", linewidth=0.6)

# -------------------- CONFIG --------------------
RNG_SEED_BASE = 1337
N_TRIALS = 30
OUT_DATA = Path("data"); OUT_FIGS = Path("figures")
OUT_DATA.mkdir(exist_ok=True); OUT_FIGS.mkdir(exist_ok=True)

# policy weights (defaults) - normalized weights for slack-aware scheduler
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
MIN_DEADLINE_S, MAX_DEADLINE_S = 3_600, 48*3_600
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
    t = 0; downloaded = set(); cache: List[int] = []; CACHE_CAP = 64
    bytes_kb = 0.0; hits = 0
    q = sorted(assets, key=lambda a: a.deadline_s)
    for s,e,kbps in windows:
        t = s
        qi = 0
        while t < e and qi < len(q):
            a = q[qi]
            if a.aid in cache:
                if t <= a.deadline_s: hits += 1
                qi += 1
                continue
            budget_kb = (e - t) * (kbps / 8.0)  # Convert kbps to KB/s
            if budget_kb <= 0: break
            need_kb = a.size_kb
            use_kb = min(budget_kb, need_kb)
            bytes_kb += use_kb
            t += max(1, int(use_kb / max(kbps/8.0,1)))  # Convert kbps to KB/s
            if use_kb >= need_kb:
                cache.append(a.aid)
                if len(cache) > CACHE_CAP: cache.pop(0)
                downloaded.add(a.aid)
                if t <= a.deadline_s: hits += 1
                qi += 1
            else:
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
                rows.append(run_trial(scenario, policy, ablations["full"], s))
        # ablations (acorn only) - use different seeds for each ablation
        for abl, w in ablations.items():
            # create unique seeds for this ablation by adding a hash of the ablation name
            abl_seeds = [s + hash(abl) % 10000 for s in seeds]
            for s in abl_seeds:
                r = run_trial(scenario, "acorn", w, s, ablation=abl)
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
    from scripts.compute_ablation_deltas import compute_ablation_deltas
    compute_ablation_deltas(
        OUT_DATA/"bap_ablation_study_aggregates.csv",
        "results/ablation/ablation_summary_by_scenario.csv"
    )
    
    # Generate the 4 final figures
    from scripts.generate_final_figures import create_final_figures
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
        Path("results/figures/final_bap_kb.png"),
        Path("results/figures/final_bap_hit.png"),
        Path("results/figures/final_ablation_kb.png"),
        Path("results/figures/final_ablation_hit.png"),
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
    N_TRIALS = args.trials
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
            weights = dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA)
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

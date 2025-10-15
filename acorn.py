# acorn.py — single-file ACORN-Edu harness
# Run: python acorn.py
# Requires: numpy, pandas, scipy, matplotlib

from __future__ import annotations
import json, math, os, random, time, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
plt.rcParams["font.size"] = 12  # enforce readable fonts across all plots
# unified errorbar styling (consistent across all figures)
ERR_KW = dict(capsize=6)
BAR_EDGE_KW = dict(edgecolor="black", linewidth=0.6)

# -------------------- CONFIG --------------------
RNG_SEED_BASE = 1337
N_TRIALS = 30
OUT_DATA = Path("data"); OUT_FIGS = Path("figures")
OUT_DATA.mkdir(exist_ok=True); OUT_FIGS.mkdir(exist_ok=True)

# policy weights (defaults)
ALPHA = 1.0   # deadline urgency
BETA  = 1.0   # reuse score
GAMMA = 1.0   # inverse size
DELTA = 1.0   # predicted availability

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
def prio_acorn(asset: Asset, t: int, windows, w) -> float:
    eps = 1.0
    urgency = 1.0 / max(asset.deadline_s - t, eps)
    inv_size = 1.0 / max(asset.size_kb, 1.0)
    pavail = availability_predictor(windows, t)
    return w["alpha"]*urgency + w["beta"]*asset.reuse_score + w["gamma"]*inv_size + w["delta"]*pavail

def policy_acorn(assets: List[Asset], windows, rng: random.Random, weights) -> Tuple[float,float]:
    # returns (bytes_kb, hit_rate)
    t = 0; downloaded = set(); bytes_kb = 0.0; hits = 0
    for s,e,kbps in windows:
        t = s
        while t < e and len(downloaded) < len(assets):
            remaining = [a for a in assets if a.aid not in downloaded]
            if not remaining: break
            best = max(remaining, key=lambda a: prio_acorn(a, t, windows, weights))
            budget_kb = (e - t) * (kbps)  # kb/s * s = kb
            if budget_kb <= 0: break
            # crude delta-sync effect: less to transfer when reuse_score is high
            use_kb = min(budget_kb, best.size_kb * (1.0 - 0.4*best.reuse_score))
            bytes_kb += use_kb
            t += max(1, int(use_kb / max(kbps,1)))
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
            budget_kb = (e - t) * (kbps)
            if budget_kb <= 0: break
            need_kb = a.size_kb
            use_kb = min(budget_kb, need_kb)
            bytes_kb += use_kb
            t += max(1, int(use_kb / max(kbps,1)))
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
def run_trial(scenario_name: str, policy: str, weights: Dict[str,float], trial_seed: int) -> Dict:
    rng = random.Random(trial_seed)
    assets = gen_assets(rng)
    windows = gen_windows(SCENARIOS[scenario_name], rng)
    if policy == "acorn":
        bkb, hr = policy_acorn(assets, windows, rng, weights)
    elif policy == "LRU_whole":
        bkb, hr = policy_lru_whole(assets, windows, rng)
    else:
        raise ValueError(policy)
    return dict(scenario=scenario_name, policy=policy, trial=trial_seed, bytes_kb=bkb, hit_rate=hr, seed=trial_seed)

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
    
    # base + ablations
    ablations = {
        "full":   dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA),
        "alpha0": dict(alpha=0.0,   beta=BETA, gamma=GAMMA, delta=DELTA),
        "beta0":  dict(alpha=ALPHA, beta=0.0,  gamma=GAMMA, delta=DELTA),
        "gamma0": dict(alpha=ALPHA, beta=BETA, gamma=0.0,   delta=DELTA),
        "delta0": dict(alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=0.0),
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
                r = run_trial(scenario, "acorn", w, s)
                r.update(ablation=abl)
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

    # -------- figures (one metric per image) --------
    plt.figure(figsize=(6,4))
    for sc in SCENARIOS.keys():
        sub = agg[agg["scenario"]==sc]
        x = np.arange(len(sub))
        plt.bar(x, sub["mean_bytes_kb"], yerr=sub["ci95_bytes_kb"], **ERR_KW, **BAR_EDGE_KW)
        plt.xticks(x, sub["policy"], rotation=0)
        plt.ylabel("KB transferred")
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.title(f"Bytes (KB) – {sc}")
        plt.tight_layout()
        plt.savefig(OUT_FIGS/f"bap_bytes_comparison_{sc}.png", dpi=DPI)
        plt.close()

    # per-scenario hit-rate plots (%)
    plt.figure(figsize=(6,4))
    for sc in SCENARIOS.keys():
        sub = agg[agg["scenario"]==sc]
        x = np.arange(len(sub))
        plt.bar(x, 100*sub["mean_hit_rate"], yerr=100*sub["ci95_hit_rate"], **ERR_KW, **BAR_EDGE_KW)
        plt.xticks(x, sub["policy"], rotation=0)
        plt.ylabel("Hit rate (%)")
        plt.title(f"Hit rate (%) – {sc}")
        plt.tight_layout()
        plt.savefig(OUT_FIGS/f"bap_hit_rate_comparison_{sc}.png", dpi=DPI)
        plt.close()

    # default-named pair for quick checks (bigger canvas + two-line ticks)
    sub = agg.sort_values(["scenario","policy"]).reset_index(drop=True)
    x = np.arange(len(sub))
    xt = [f"{sc}\n{pol}" for sc, pol in zip(sub["scenario"], sub["policy"])]

    plt.figure(figsize=(10, 5))
    plt.bar(x, sub["mean_bytes_kb"], yerr=sub["ci95_bytes_kb"], **ERR_KW, **BAR_EDGE_KW)
    plt.xticks(x, xt, rotation=0, ha="center")
    plt.ylabel("KB transferred")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.gcf().subplots_adjust(bottom=0.25)  # prevent label overlap
    plt.tight_layout()
    plt.savefig(OUT_FIGS/"bap_bytes_comparison.png", dpi=DPI)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, 100*sub["mean_hit_rate"], yerr=100*sub["ci95_hit_rate"], **ERR_KW, **BAR_EDGE_KW)
    plt.xticks(x, xt, rotation=0, ha="center")
    plt.ylabel("Hit rate (%)")
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.savefig(OUT_FIGS/"bap_hit_rate_comparison.png", dpi=DPI)
    plt.close()

    # -------- Ablation figures (acorn only, from RAW trials for correct CI) --------
    # Recompute ablation means + CIs from per-trial data (dfa) across all scenarios.
    def _agg_with_ci(df_series):
        return pd.Series({
            "mean": df_series.mean(),
            "ci95": t_ci(df_series.to_numpy())
        })
    abl_bytes = dfa.groupby("ablation")["bytes_kb"].apply(_agg_with_ci).unstack()
    abl_hit   = dfa.groupby("ablation")["hit_rate"].apply(_agg_with_ci).unstack()
    # Pretty order + labels
    abl_order = ["full","alpha0","beta0","gamma0","delta0"]
    pretty = {
        "full":"Full AcornScheduler",
        "alpha0":"No Deadline Priority",
        "beta0":"No Reuse Priority",
        "gamma0":"No Size Priority",
        "delta0":"No Network Priority",
    }
    x = np.arange(len(abl_order))
    labels = [pretty[a] for a in abl_order]

    plt.figure(figsize=(12,5))
    plt.bar(x, 100*abl_hit.loc[abl_order,"mean"], yerr=100*abl_hit.loc[abl_order,"ci95"], **ERR_KW, **BAR_EDGE_KW, color="#9bd0ff")
    plt.xticks(x, labels, rotation=0, ha="center")
    plt.ylabel("Prefetch Hit Rate (%)")
    plt.title("ACORN-Edu Ablation: Hit Rate")
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.tight_layout()
    plt.savefig(OUT_FIGS/"ablation_hit_rate.png", dpi=DPI)
    plt.close()

    plt.figure(figsize=(12,5))
    plt.bar(x, abl_bytes.loc[abl_order,"mean"], yerr=abl_bytes.loc[abl_order,"ci95"], **ERR_KW, **BAR_EDGE_KW, color="#ffb3b3")
    plt.xticks(x, labels, rotation=0, ha="center")
    plt.ylabel("Bytes Transferred (KB)")
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.title("ACORN-Edu Ablation: Bandwidth")
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.tight_layout()
    plt.savefig(OUT_FIGS/"ablation_bytes.png", dpi=DPI)
    plt.close()

    # metadata
    # obtain current git commit (if available)
    def _git_rev():
        try:
            return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            return None

    (OUT_DATA/"run_metadata.json").write_text(json.dumps({
        "seed_base": RNG_SEED_BASE,
        "n_trials": N_TRIALS,
        "timestamp": int(time.time()),
        "git_rev": _git_rev(),
        "policies": ["acorn","LRU_whole"],
        "ablations": ["full","alpha0","beta0","gamma0","delta0"]
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
        OUT_FIGS/"bap_hit_rate_comparison.png",
        OUT_FIGS/"bap_bytes_comparison.png",
        OUT_FIGS/"ablation_hit_rate.png",
        OUT_FIGS/"ablation_bytes.png",
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
    ap.add_argument("--seed", type=int, default=RNG_SEED_BASE, help="override base RNG seed (default: 1337)")
    args = ap.parse_args()

    # override globals for this run
    N_TRIALS = args.trials
    RNG_SEED_BASE = args.seed

    run_all()
    verify()
    print("OK: data/*.csv + figures/*.png + run_metadata.json written (KB-only; 95% t-CIs)")

"""
η (CDC reuse factor) sensitivity sweep.
Runs ACORN-Edu's policy_acorn with η ∈ {0.2, 0.3, 0.4, 0.5, 0.6}
for both scenarios × 30 trials, reports mean KB transferred and hit-rate.
"""
import sys, os, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

# Import core pieces from acorn.py
from acorn import (
    gen_assets, gen_windows, SCENARIOS, WEIGHTS,
    N_ASSETS, RNG_SEED_BASE, N_TRIALS,
    Asset, Candidate, compute_acorn_scores, prio_acorn,
)

# ---------- Re-implement policy_acorn with configurable η ----------
def policy_acorn_eta(assets, windows, rng, weights, eta, ablation=None):
    """Identical to policy_acorn but with η as a parameter."""
    t = 0; downloaded = set(); bytes_kb = 0.0; hits = 0
    for s, e, kbps in windows:
        t = s
        while t < e and len(downloaded) < len(assets):
            remaining = [a for a in assets if a.aid not in downloaded]
            if not remaining:
                break
            best = max(remaining, key=lambda a: prio_acorn(a, t, windows, weights, ablation))
            budget_kb = (e - t) * (kbps / 8.0)
            if budget_kb <= 0:
                break
            use_kb = min(budget_kb, best.size_kb * (1.0 - eta * best.reuse_score))
            bytes_kb += use_kb
            t += max(1, int(use_kb / max(kbps / 8.0, 1)))
            downloaded.add(best.aid)
            if t <= best.deadline_s:
                hits += 1
    hit_rate = hits / max(1, len(assets))
    return bytes_kb, hit_rate


# ---------- Run sweep ----------
ETA_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]
rows = []

for scenario_name in ["nightly_wifi", "spotty_cellular"]:
    scfg = SCENARIOS[scenario_name]
    w = WEIGHTS[scenario_name]
    for eta in ETA_VALUES:
        for trial in range(N_TRIALS):
            seed = RNG_SEED_BASE + trial
            rng = random.Random(seed)
            assets = gen_assets(rng)
            windows = gen_windows(scfg, rng)
            bkb, hr = policy_acorn_eta(assets, windows, rng, w, eta)
            rows.append({
                "scenario": scenario_name,
                "eta": eta,
                "trial": trial,
                "bytes_kb": bkb,
                "hit_rate": hr,
            })
        print(f"  Done: {scenario_name} η={eta}")

df = pd.DataFrame(rows)

# ---------- Summary ----------
print("\n" + "=" * 70)
print("η SENSITIVITY SWEEP — ACORN-Edu (N=30 per cell)")
print("=" * 70)

for scenario_name in ["nightly_wifi", "spotty_cellular"]:
    sub = df[df["scenario"] == scenario_name]
    print(f"\n{'─' * 50}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'─' * 50}")
    print(f"  {'η':>5}  {'Mean KB':>12}  {'Std KB':>10}  {'Mean HR':>8}  {'Std HR':>8}")
    
    baseline_kb = sub[sub["eta"] == 0.4]["bytes_kb"].mean()
    
    for eta in ETA_VALUES:
        s = sub[sub["eta"] == eta]
        mean_kb = s["bytes_kb"].mean()
        std_kb = s["bytes_kb"].std()
        mean_hr = s["hit_rate"].mean()
        std_hr = s["hit_rate"].std()
        pct_diff = 100.0 * (mean_kb - baseline_kb) / baseline_kb
        marker = " ← baseline" if eta == 0.4 else f"  ({pct_diff:+.1f}% vs η=0.4)"
        print(f"  {eta:>5.1f}  {mean_kb:>12,.1f}  {std_kb:>10,.1f}  {mean_hr:>8.4f}  {std_hr:>8.4f}{marker}")
    
    # Check if hit-rate ranking (ACORN > baseline) holds for all η
    # (we only have ACORN here, so check if hit-rate is stable)
    hr_by_eta = sub.groupby("eta")["hit_rate"].mean()
    print(f"\n  Hit-rate range: {hr_by_eta.min():.4f} – {hr_by_eta.max():.4f}")
    print(f"  Hit-rate spread: {(hr_by_eta.max() - hr_by_eta.min()):.4f}")

# Save CSV
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "data", "eta_sensitivity.csv")
df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")

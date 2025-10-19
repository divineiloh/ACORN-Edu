#!/usr/bin/env python3
"""
Statistical dominance guardrail for BAP:
We flag an ablation ONLY if it shows a PRACTICALLY MEANINGFUL and STATISTICALLY SIGNIFICANT
improvement on BOTH primary metrics in a scenario, relative to FULL, using independent trials.

Primary metrics per paper spec:
- KB transferred (lower is better)
- Prefetch hit-rate (%) (higher is better)

Criteria (joint):
- one-sided independent t-tests at alpha=0.05
- Cohen's d >= 0.50 (medium effect size)
- practical margins: >=3% fewer KB AND >=3 percentage-point higher hit-rate

This avoids false positives from small, noisy differences at N=30.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("results/data/bap_ablation_study_results.csv")
ALPHA = 0.05
D_MIN = 0.50          # medium effect size threshold
REL_KB_MARGIN = 0.03  # >=3% reduction in KB transferred
PP_HIT_MARGIN  = 0.03 # >=3 percentage-point increase in hit-rate

def cohens_d(group1, group2):
    """Calculate Cohen's d for independent groups"""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-12)

def main():
    if not RESULTS.exists():
        sys.exit(f"ERROR: missing {RESULTS}")

    df = pd.read_csv(RESULTS)

    errors = []

    for scen, g in df.groupby("scenario"):
        full_data = g[g.ablation=="full"]

        for ab in ["alpha0","beta0","gamma0","delta0"]:
            abl_data = g[g.ablation==ab]
            if abl_data.empty:
                errors.append(f"{scen}: no data for {ab}")
                continue

            # Extract metrics
            full_kb = full_data["bytes_kb"].values
            abl_kb = abl_data["bytes_kb"].values
            full_hit = full_data["hit_rate"].values
            abl_hit = abl_data["hit_rate"].values

            # one-sided independent t-tests: ablation better?
            # KB transferred: ablation LOWER than full
            t_kb,  p_kb  = stats.ttest_ind(abl_kb, full_kb, alternative="less")
            # Hit-rate: ablation HIGHER than full
            t_hit, p_hit = stats.ttest_ind(abl_hit, full_hit, alternative="greater")

            # effect sizes (Cohen's d for independent groups)
            d_eff_kb  = cohens_d(abl_kb, full_kb)    # negative if ablation reduces KB
            d_eff_hit = cohens_d(abl_hit, full_hit)   # positive if ablation raises hit

            # practical gains
            rel_kb_gain = (np.mean(full_kb) - np.mean(abl_kb)) / (np.mean(full_kb) + 1e-12)  # + if fewer KB
            pp_hit_gain = 100.0 * (np.mean(abl_hit) - np.mean(full_hit))  # + if higher hit rate

            kb_ok  = (p_kb  < ALPHA) and (abs(d_eff_kb) >= D_MIN) and (rel_kb_gain >= REL_KB_MARGIN)
            hit_ok = (p_hit < ALPHA) and (d_eff_hit     >= D_MIN)  and (pp_hit_gain >= 100.0*PP_HIT_MARGIN)

            if kb_ok and hit_ok:
                errors.append(
                    f"{scen}: {ab} shows practical & significant dominance — "
                    f"KB gain {rel_kb_gain*100:.2f}% (p={p_kb:.3g}, |d|={abs(d_eff_kb):.2f}); "
                    f"Hit gain {pp_hit_gain:.2f} pp (p={p_hit:.3g}, d={d_eff_hit:.2f})"
                )
            else:
                print(f"OK {scen}/{ab}: no practical dominance "
                      f"(KB: p={p_kb:.3f}, |d|={abs(d_eff_kb):.2f}, delta={rel_kb_gain*100:.2f}% ; "
                      f"Hit: p={p_hit:.3f}, d={d_eff_hit:.2f}, delta={pp_hit_gain:.2f} pp)")

    if errors:
        for e in errors:
            print("ERROR:", e)
        sys.exit(1)

    print("Statistical guardrail PASS: no ablation shows practical & significant dominance.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

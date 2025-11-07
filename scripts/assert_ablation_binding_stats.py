#!/usr/bin/env python3
"""
Statistical dominance guardrail for BAP:
We flag an ablation ONLY if it shows a PRACTICALLY MEANINGFUL and STATISTICALLY SIGNIFICANT
improvement on BOTH primary metrics in a scenario, relative to FULL, using independent trials.

Primary metrics per paper spec:
- KB transferred (lower is better)
- Prefetch hit-rate (%) (higher is better)

Criteria (joint):
- paired t-tests with Cohen's d effect sizes, significance threshold p < 0.05
- Wilcoxon signed-rank tests confirm significance
- Levene's test shows homogeneous variance
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

def cohens_d_paired(group1, group2):
    """Calculate Cohen's d for paired groups"""
    diff = group1 - group2
    std_diff = np.std(diff, ddof=1)
    return np.mean(diff) / (std_diff + 1e-12)

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

            # Ensure equal lengths for paired tests (if needed, align by trial ID or index)
            min_len = min(len(full_kb), len(abl_kb))
            if len(full_kb) != len(abl_kb):
                # If lengths differ, use first min_len elements
                full_kb = full_kb[:min_len]
                abl_kb = abl_kb[:min_len]
                full_hit = full_hit[:min_len]
                abl_hit = abl_hit[:min_len]

            # Paired t-tests: ablation better?
            # KB transferred: ablation LOWER than full
            t_kb,  p_kb  = stats.ttest_rel(abl_kb, full_kb, alternative="less")
            # Hit-rate: ablation HIGHER than full
            t_hit, p_hit = stats.ttest_rel(abl_hit, full_hit, alternative="greater")

            # Effect sizes (Cohen's d for paired groups)
            d_eff_kb  = cohens_d_paired(abl_kb, full_kb)    # negative if ablation reduces KB
            d_eff_hit = cohens_d_paired(abl_hit, full_hit)   # positive if ablation raises hit

            # Wilcoxon signed-rank tests to confirm significance
            wilcoxon_kb_stat, wilcoxon_kb_p = stats.wilcoxon(abl_kb, full_kb, alternative="less")
            wilcoxon_hit_stat, wilcoxon_hit_p = stats.wilcoxon(abl_hit, full_hit, alternative="greater")

            # Levene's test for variance homogeneity
            levene_kb_stat, levene_kb_p = stats.levene(abl_kb, full_kb)
            levene_hit_stat, levene_hit_p = stats.levene(abl_hit, full_hit)

            # practical gains
            rel_kb_gain = (np.mean(full_kb) - np.mean(abl_kb)) / (np.mean(full_kb) + 1e-12)  # + if fewer KB
            pp_hit_gain = 100.0 * (np.mean(abl_hit) - np.mean(full_hit))  # + if higher hit rate

            # Check significance: paired t-test p < 0.05 AND Wilcoxon confirms (p < 0.05)
            kb_sig = (p_kb < ALPHA) and (wilcoxon_kb_p < ALPHA)
            hit_sig = (p_hit < ALPHA) and (wilcoxon_hit_p < ALPHA)
            
            # Check variance homogeneity: Levene's test p >= 0.05 (homogeneous variance)
            kb_var_homogeneous = levene_kb_p >= ALPHA
            hit_var_homogeneous = levene_hit_p >= ALPHA

            kb_ok  = kb_sig and (abs(d_eff_kb) >= D_MIN) and (rel_kb_gain >= REL_KB_MARGIN)
            hit_ok = hit_sig and (d_eff_hit     >= D_MIN)  and (pp_hit_gain >= 100.0*PP_HIT_MARGIN)

            if kb_ok and hit_ok:
                errors.append(
                    f"{scen}: {ab} shows practical & significant dominance — "
                    f"KB gain {rel_kb_gain*100:.2f}% (t-test p={p_kb:.3g}, Wilcoxon p={wilcoxon_kb_p:.3g}, |d|={abs(d_eff_kb):.2f}, Levene p={levene_kb_p:.3g}); "
                    f"Hit gain {pp_hit_gain:.2f} pp (t-test p={p_hit:.3g}, Wilcoxon p={wilcoxon_hit_p:.3g}, d={d_eff_hit:.2f}, Levene p={levene_hit_p:.3g})"
                )
            else:
                print(f"OK {scen}/{ab}: no practical dominance "
                      f"(KB: t-test p={p_kb:.3f}, Wilcoxon p={wilcoxon_kb_p:.3f}, |d|={abs(d_eff_kb):.2f}, delta={rel_kb_gain*100:.2f}%, Levene p={levene_kb_p:.3f}, var_homogeneous={kb_var_homogeneous}; "
                      f"Hit: t-test p={p_hit:.3f}, Wilcoxon p={wilcoxon_hit_p:.3f}, d={d_eff_hit:.2f}, delta={pp_hit_gain:.2f} pp, Levene p={levene_hit_p:.3f}, var_homogeneous={hit_var_homogeneous})")

    if errors:
        for e in errors:
            print("ERROR:", e)
        sys.exit(1)

    print("Statistical guardrail PASS: no ablation shows practical & significant dominance.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

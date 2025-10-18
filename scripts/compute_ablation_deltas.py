# scripts/compute_ablation_deltas.py
import pandas as pd
import os
from pathlib import Path

def compute_ablation_deltas(in_csv, out_csv):
    agg = pd.read_csv(in_csv)  # expects: scenario, ablation, mean_bytes_kb, mean_hit_rate, ci95_bytes_kb, ci95_hit_rate
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
    pd.DataFrame(out).to_csv(out_csv, index=False)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    out_dir = Path("results/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    compute_ablation_deltas(
        "data/bap_ablation_study_aggregates.csv",
        "results/ablation/ablation_summary_by_scenario.csv"
    )

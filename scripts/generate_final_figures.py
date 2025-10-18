# scripts/generate_final_figures.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from viz_style import paper_style, format_kb_axis, format_pct_axis, add_bar_labels
from utils.labels import label_scenario, label_policy, label_ablation

def create_final_figures():
    # Create output directory
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    policy_agg = pd.read_csv("data/bap_network_scenario_aggregates.csv")
    ablation_agg = pd.read_csv("results/ablation/ablation_summary_by_scenario.csv")
    
    # Add labels
    policy_agg["scenario_label"] = policy_agg["scenario"].map(label_scenario)
    policy_agg["policy_label"] = policy_agg["policy"].map(label_policy)
    ablation_agg["scenario_label"] = ablation_agg["scenario"].map(label_scenario)
    ablation_agg["ablation_label"] = ablation_agg["ablation"].map(label_ablation)
    
    # 1. BAP KB transferred
    with paper_style():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        
        for ax, scenario in [(axL, "nightly_wifi"), (axR, "spotty_cellular")]:
            sub = policy_agg[policy_agg["scenario"] == scenario]
            x = np.arange(len(sub))
            bars = ax.bar(x, sub["mean_bytes_kb"], yerr=sub["ci95_bytes_kb"], 
                        capsize=4, error_kw={"elinewidth": 1.5})
            ax.set_xticks(x)
            ax.set_xticklabels(sub["policy_label"])
            format_kb_axis(ax)
            add_bar_labels(ax, sub["mean_bytes_kb"])
            ax.set_title(f"KB transferred — {label_scenario(scenario)}")
        
        plt.savefig("results/figures/final_bap_kb.png", bbox_inches="tight")
        plt.close()
    
    # 2. BAP Hit rate
    with paper_style():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        
        for ax, scenario in [(axL, "nightly_wifi"), (axR, "spotty_cellular")]:
            sub = policy_agg[policy_agg["scenario"] == scenario]
            x = np.arange(len(sub))
            bars = ax.bar(x, 100*sub["mean_hit_rate"], yerr=100*sub["ci95_hit_rate"], 
                        capsize=4, error_kw={"elinewidth": 1.5})
            ax.set_xticks(x)
            ax.set_xticklabels(sub["policy_label"])
            format_pct_axis(ax)
            add_bar_labels(ax, 100*sub["mean_hit_rate"], is_percent=True)
            ax.set_title(f"Prefetch hit-rate (%) — {label_scenario(scenario)}")
        
        plt.savefig("results/figures/final_bap_hit.png", bbox_inches="tight")
        plt.close()
    
    # 3. Ablation KB transferred
    with paper_style():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        
        for ax, scenario in [(axL, "nightly_wifi"), (axR, "spotty_cellular")]:
            sub = ablation_agg[ablation_agg["scenario"] == scenario]
            # Order: full, alpha0, beta0, gamma0, delta0
            order = ["full", "alpha0", "beta0", "gamma0", "delta0"]
            sub = sub.set_index("ablation").loc[order].reset_index()
            x = np.arange(len(sub))
            bars = ax.bar(x, sub["mean_bytes_kb"], yerr=sub["ci95_bytes_kb"], 
                        capsize=4, error_kw={"elinewidth": 1.5})
            ax.set_xticks(x)
            ax.set_xticklabels(sub["ablation_label"])
            format_kb_axis(ax)
            add_bar_labels(ax, sub["mean_bytes_kb"])
            ax.set_title(f"KB transferred — {label_scenario(scenario)}")
        
        plt.savefig("results/figures/final_ablation_kb.png", bbox_inches="tight")
        plt.close()
    
    # 4. Ablation Hit rate
    with paper_style():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        
        for ax, scenario in [(axL, "nightly_wifi"), (axR, "spotty_cellular")]:
            sub = ablation_agg[ablation_agg["scenario"] == scenario]
            # Order: full, alpha0, beta0, gamma0, delta0
            order = ["full", "alpha0", "beta0", "gamma0", "delta0"]
            sub = sub.set_index("ablation").loc[order].reset_index()
            x = np.arange(len(sub))
            bars = ax.bar(x, 100*sub["mean_hit_rate"], yerr=100*sub["ci95_hit_rate"], 
                        capsize=4, error_kw={"elinewidth": 1.5})
            ax.set_xticks(x)
            ax.set_xticklabels(sub["ablation_label"])
            format_pct_axis(ax)
            add_bar_labels(ax, 100*sub["mean_hit_rate"], is_percent=True)
            ax.set_title(f"Prefetch hit-rate (%) — {label_scenario(scenario)}")
        
        plt.savefig("results/figures/final_ablation_hit.png", bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    create_final_figures()
    print("Generated 4 final figures in results/figures/")

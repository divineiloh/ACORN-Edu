#!/usr/bin/env python3
"""
ACORN-Edu Results Visualization
Creates clean, legible plots with straight labels and proper formatting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set matplotlib backend and style
os.environ.setdefault("MPLBACKEND", "Agg")
plt.style.use('default')

def create_network_plots():
    """Create network scenario comparison plots."""
    # Read data
    df = pd.read_csv("data/bap_network_scenario_results.csv")
    
    # Set up the plot style
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prepare data
    scenarios = ["nightly_wifi", "spotty_cellular"]
    schedulers = ["DeadlineFIFO_Downloader", "AcornScheduler"]
    
    # Plot 1: Bytes Transferred
    x = np.arange(len(scenarios))
    width = 0.35
    
    for i, scheduler in enumerate(schedulers):
        subset = df[df["scheduler"] == scheduler]
        means = subset["bytes_mean_(KB)"].values
        lows = subset["bytes_ci_lower_(KB)"].values
        highs = subset["bytes_ci_upper_(KB)"].values
        
        # Create error bars (ensure no negative values)
        yerr_lower = np.maximum(0, means - lows)
        yerr_upper = np.maximum(0, highs - means)
        yerr = np.vstack([yerr_lower, yerr_upper])
        
        bars = ax1.bar(x + i * width, means, width, 
                      yerr=yerr, capsize=5, 
                      label=scheduler.replace("_", " "),
                      alpha=0.8)
    
    ax1.set_xlabel("Network Scenario", fontweight='bold')
    ax1.set_ylabel("Bytes Transferred (KB)", fontweight='bold')
    ax1.set_title("ACORN-Edu: Bandwidth Usage", fontweight='bold', pad=20)
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(["Nightly WiFi", "Spotty Cellular"])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hit Rate
    for i, scheduler in enumerate(schedulers):
        subset = df[df["scheduler"] == scheduler]
        means = subset["hit_rate_mean_(%)"].values
        lows = subset["hit_rate_ci_lower_(%)"].values
        highs = subset["hit_rate_ci_upper_(%)"].values
        
        # Create error bars (ensure no negative values)
        yerr_lower = np.maximum(0, means - lows)
        yerr_upper = np.maximum(0, highs - means)
        yerr = np.vstack([yerr_lower, yerr_upper])
        
        bars = ax2.bar(x + i * width, means, width, 
                      yerr=yerr, capsize=5, 
                      label=scheduler.replace("_", " "),
                      alpha=0.8)
    
    ax2.set_xlabel("Network Scenario", fontweight='bold')
    ax2.set_ylabel("Prefetch Hit Rate (%)", fontweight='bold')
    ax2.set_title("ACORN-Edu: Prefetch Hit Rate", fontweight='bold', pad=20)
    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(["Nightly WiFi", "Spotty Cellular"])
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/acorn_network_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_ablation_plots():
    """Create ablation study plots."""
    # Read data
    df = pd.read_csv("data/bap_ablation_study_results.csv")
    
    # Set up the plot style
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Hit Rate
    configs = df["ablation_config"].values
    means = df["hit_rate_mean_(%)"].values
    lows = df["hit_rate_ci_lower_(%)"].values
    highs = df["hit_rate_ci_upper_(%)"].values
    
    # Create error bars (ensure no negative values)
    yerr_lower = np.maximum(0, means - lows)
    yerr_upper = np.maximum(0, highs - means)
    yerr = np.vstack([yerr_lower, yerr_upper])
    
    bars = ax1.bar(configs, means, yerr=yerr, capsize=5, alpha=0.8, color='skyblue')
    ax1.set_xlabel("Ablation Configuration", fontweight='bold')
    ax1.set_ylabel("Prefetch Hit Rate (%)", fontweight='bold')
    ax1.set_title("ACORN-Edu Ablation: Hit Rate", fontweight='bold', pad=20)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bytes Transferred
    means = df["bytes_mean_(KB)"].values
    lows = df["bytes_ci_lower_(KB)"].values
    highs = df["bytes_ci_upper_(KB)"].values
    
    # Create error bars (ensure no negative values)
    yerr_lower = np.maximum(0, means - lows)
    yerr_upper = np.maximum(0, highs - means)
    yerr = np.vstack([yerr_lower, yerr_upper])
    
    bars = ax2.bar(configs, means, yerr=yerr, capsize=5, alpha=0.8, color='lightcoral')
    ax2.set_xlabel("Ablation Configuration", fontweight='bold')
    ax2.set_ylabel("Bytes Transferred (KB)", fontweight='bold')
    ax2.set_title("ACORN-Edu Ablation: Bandwidth", fontweight='bold', pad=20)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/acorn_ablation_study.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    """Generate all plots."""
    # Create output directory
    os.makedirs("figures", exist_ok=True)
    
    print("Creating network comparison plots...")
    create_network_plots()
    
    print("Creating ablation study plots...")
    create_ablation_plots()
    
    print("All plots saved to figures/ directory")

if __name__ == "__main__":
    main()

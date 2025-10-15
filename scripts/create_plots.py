#!/usr/bin/env python3
"""
ACORN-Edu Results Visualization
Creates separate, clean plots with simple names and straight labels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set matplotlib backend
os.environ.setdefault("MPLBACKEND", "Agg")


def create_network_bytes_plot():
    """Create network bytes comparison plot."""
    df = pd.read_csv("data/bap_network_scenario_results.csv")

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ["nightly_wifi", "spotty_cellular"]
    schedulers = ["DeadlineFIFO_Downloader", "AcornScheduler"]

    x = np.arange(len(scenarios))
    width = 0.35

    for i, scheduler in enumerate(schedulers):
        subset = df[df["scheduler"] == scheduler]
        means = subset["bytes_mean_(KB)"].values
        lows = subset["bytes_ci_lower_(KB)"].values
        highs = subset["bytes_ci_upper_(KB)"].values

        # Simple error bars
        yerr_lower = np.maximum(0, means - lows)
        yerr_upper = np.maximum(0, highs - means)
        yerr = np.vstack([yerr_lower, yerr_upper])

        # Simple scheduler names
        label = "Deadline FIFO" if "Deadline" in scheduler else "AcornScheduler"

        ax.bar(
            x + i * width, means, width, yerr=yerr, capsize=5, label=label, alpha=0.8
        )

    ax.set_xlabel("Network Scenario", fontweight="bold")
    ax.set_ylabel("Bytes Transferred (KB)", fontweight="bold")
    ax.set_title("ACORN-Edu: Bandwidth Usage", fontweight="bold", pad=20)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Nightly WiFi", "Spotty Cellular"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/network_bytes.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_network_hitrate_plot():
    """Create network hit rate comparison plot."""
    df = pd.read_csv("data/bap_network_scenario_results.csv")

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ["nightly_wifi", "spotty_cellular"]
    schedulers = ["DeadlineFIFO_Downloader", "AcornScheduler"]

    x = np.arange(len(scenarios))
    width = 0.35

    for i, scheduler in enumerate(schedulers):
        subset = df[df["scheduler"] == scheduler]
        means = subset["hit_rate_mean_(%)"].values
        lows = subset["hit_rate_ci_lower_(%)"].values
        highs = subset["hit_rate_ci_upper_(%)"].values

        # Simple error bars
        yerr_lower = np.maximum(0, means - lows)
        yerr_upper = np.maximum(0, highs - means)
        yerr = np.vstack([yerr_lower, yerr_upper])

        # Simple scheduler names
        label = "Deadline FIFO" if "Deadline" in scheduler else "AcornScheduler"

        ax.bar(
            x + i * width, means, width, yerr=yerr, capsize=5, label=label, alpha=0.8
        )

    ax.set_xlabel("Network Scenario", fontweight="bold")
    ax.set_ylabel("Prefetch Hit Rate (%)", fontweight="bold")
    ax.set_title("ACORN-Edu: Prefetch Hit Rate", fontweight="bold", pad=20)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Nightly WiFi", "Spotty Cellular"])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/network_hitrate.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_ablation_hitrate_plot():
    """Create ablation hit rate plot."""
    df = pd.read_csv("data/bap_ablation_study_results.csv")

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Simple config names
    config_names = {
        "Full AcornScheduler": "Full",
        "No Deadline (alpha=0)": "No Deadline",
        "No Reuse (beta=0)": "No Reuse",
        "No Size (gamma=0)": "No Size",
        "No Network (delta=0)": "No Network",
    }

    configs = [config_names.get(name, name) for name in df["ablation_config"].values]
    means = df["hit_rate_mean_(%)"].values
    lows = df["hit_rate_ci_lower_(%)"].values
    highs = df["hit_rate_ci_upper_(%)"].values

    # Simple error bars
    yerr_lower = np.maximum(0, means - lows)
    yerr_upper = np.maximum(0, highs - means)
    yerr = np.vstack([yerr_lower, yerr_upper])

    ax.bar(configs, means, yerr=yerr, capsize=5, alpha=0.8, color="skyblue")
    ax.set_xlabel("Ablation Configuration", fontweight="bold")
    ax.set_ylabel("Prefetch Hit Rate (%)", fontweight="bold")
    ax.set_title("ACORN-Edu Ablation: Hit Rate", fontweight="bold", pad=20)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=0)  # Keep labels straight
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/ablation_hitrate.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_ablation_bytes_plot():
    """Create ablation bytes plot."""
    df = pd.read_csv("data/bap_ablation_study_results.csv")

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    # Simple config names
    config_names = {
        "Full AcornScheduler": "Full",
        "No Deadline (alpha=0)": "No Deadline",
        "No Reuse (beta=0)": "No Reuse",
        "No Size (gamma=0)": "No Size",
        "No Network (delta=0)": "No Network",
    }

    configs = [config_names.get(name, name) for name in df["ablation_config"].values]
    means = df["bytes_mean_(KB)"].values
    lows = df["bytes_ci_lower_(KB)"].values
    highs = df["bytes_ci_upper_(KB)"].values

    # Simple error bars
    yerr_lower = np.maximum(0, means - lows)
    yerr_upper = np.maximum(0, highs - means)
    yerr = np.vstack([yerr_lower, yerr_upper])

    ax.bar(configs, means, yerr=yerr, capsize=5, alpha=0.8, color="lightcoral")
    ax.set_xlabel("Ablation Configuration", fontweight="bold")
    ax.set_ylabel("Bytes Transferred (KB)", fontweight="bold")
    ax.set_title("ACORN-Edu Ablation: Bandwidth", fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=0)  # Keep labels straight
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/ablation_bytes.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Generate all separate plots."""
    os.makedirs("figures", exist_ok=True)

    print("Creating network bytes plot...")
    create_network_bytes_plot()

    print("Creating network hit rate plot...")
    create_network_hitrate_plot()

    print("Creating ablation hit rate plot...")
    create_ablation_hitrate_plot()

    print("Creating ablation bytes plot...")
    create_ablation_bytes_plot()

    print("All plots saved to figures/ directory")


if __name__ == "__main__":
    main()

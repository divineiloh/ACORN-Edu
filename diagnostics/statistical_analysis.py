#!/usr/bin/env python3
"""
Statistical Analysis for ACORN Ablation Dominance
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def load_data():
    """Load the ablation study results"""
    return pd.read_csv("results/data/bap_ablation_study_results.csv")

def perform_statistical_tests():
    """Perform t-tests on the dominance cases"""
    
    df = load_data()
    
    print("=== STATISTICAL ANALYSIS OF ABLATION DOMINANCE ===\n")
    
    # Test 1: alpha0 vs FULL in nightly_wifi
    print("1. ALPHA0 vs FULL in NIGHTLY_WIFI")
    print("   Hypothesis: alpha0 delivers more bytes than FULL")
    print("   H0: alpha0_bytes = full_bytes")
    print("   H1: alpha0_bytes > full_bytes\n")
    
    nightly_alpha0 = df[(df['scenario'] == 'nightly_wifi') & (df['ablation'] == 'alpha0')]['bytes_kb']
    nightly_full = df[(df['scenario'] == 'nightly_wifi') & (df['ablation'] == 'full')]['bytes_kb']
    
    t_stat, p_value = stats.ttest_ind(nightly_alpha0, nightly_full, alternative='greater')
    cohens_d = (nightly_alpha0.mean() - nightly_full.mean()) / np.sqrt((nightly_alpha0.var() + nightly_full.var()) / 2)
    
    print(f"   alpha0 mean: {nightly_alpha0.mean():.1f} KB")
    print(f"   FULL mean:   {nightly_full.mean():.1f} KB")
    print(f"   Difference:  {nightly_alpha0.mean() - nightly_full.mean():.1f} KB")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value:     {p_value:.6f}")
    print(f"   Cohen's d:   {cohens_d:.3f}")
    print(f"   Significant: {'YES' if p_value < 0.05 else 'NO'}\n")
    
    # Test 2: gamma0 vs FULL in spotty_cellular
    print("2. GAMMA0 vs FULL in SPOTTY_CELLULAR")
    print("   Hypothesis: gamma0 delivers more bytes than FULL")
    print("   H0: gamma0_bytes = full_bytes")
    print("   H1: gamma0_bytes > full_bytes\n")
    
    spotty_gamma0 = df[(df['scenario'] == 'spotty_cellular') & (df['ablation'] == 'gamma0')]['bytes_kb']
    spotty_full = df[(df['scenario'] == 'spotty_cellular') & (df['ablation'] == 'full')]['bytes_kb']
    
    t_stat, p_value = stats.ttest_ind(spotty_gamma0, spotty_full, alternative='greater')
    cohens_d = (spotty_gamma0.mean() - spotty_full.mean()) / np.sqrt((spotty_gamma0.var() + spotty_full.var()) / 2)
    
    print(f"   gamma0 mean: {spotty_gamma0.mean():.1f} KB")
    print(f"   FULL mean:   {spotty_full.mean():.1f} KB")
    print(f"   Difference:  {spotty_gamma0.mean() - spotty_full.mean():.1f} KB")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value:     {p_value:.6f}")
    print(f"   Cohen's d:   {cohens_d:.3f}")
    print(f"   Significant: {'YES' if p_value < 0.05 else 'NO'}\n")
    
    # Test 3: Hit rate comparison for alpha0 in nightly_wifi
    print("3. ALPHA0 vs FULL HIT RATE in NIGHTLY_WIFI")
    print("   Hypothesis: alpha0 has lower hit rate than FULL")
    print("   H0: alpha0_hit_rate = full_hit_rate")
    print("   H1: alpha0_hit_rate < full_hit_rate\n")
    
    nightly_alpha0_hit = df[(df['scenario'] == 'nightly_wifi') & (df['ablation'] == 'alpha0')]['hit_rate']
    nightly_full_hit = df[(df['scenario'] == 'nightly_wifi') & (df['ablation'] == 'full')]['hit_rate']
    
    t_stat, p_value = stats.ttest_ind(nightly_alpha0_hit, nightly_full_hit, alternative='less')
    cohens_d = (nightly_alpha0_hit.mean() - nightly_full_hit.mean()) / np.sqrt((nightly_alpha0_hit.var() + nightly_full_hit.var()) / 2)
    
    print(f"   alpha0 hit rate: {nightly_alpha0_hit.mean():.3f}")
    print(f"   FULL hit rate:   {nightly_full_hit.mean():.3f}")
    print(f"   Difference:      {nightly_alpha0_hit.mean() - nightly_full_hit.mean():.3f}")
    print(f"   t-statistic:     {t_stat:.3f}")
    print(f"   p-value:         {p_value:.6f}")
    print(f"   Cohen's d:       {cohens_d:.3f}")
    print(f"   Significant:     {'YES' if p_value < 0.05 else 'NO'}\n")
    
    return {
        'alpha0_nightly_bytes': (t_stat, p_value, cohens_d),
        'gamma0_spotty_bytes': (t_stat, p_value, cohens_d),
        'alpha0_nightly_hit': (t_stat, p_value, cohens_d)
    }

def create_box_plots():
    """Create box plots for the dominance cases"""
    
    df = load_data()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Nightly Wi-Fi Bytes
    nightly_data = []
    nightly_labels = []
    for ablation in ['full', 'alpha0', 'beta0', 'gamma0', 'delta0']:
        data = df[(df['scenario'] == 'nightly_wifi') & (df['ablation'] == ablation)]['bytes_kb']
        nightly_data.append(data)
        nightly_labels.append(ablation)
    
    ax1.boxplot(nightly_data, labels=nightly_labels)
    ax1.set_title('Nightly Wi-Fi: Bytes Delivered')
    ax1.set_ylabel('KB')
    ax1.tick_params(axis='x', rotation=45)
    
    # Spotty Cellular Bytes
    spotty_data = []
    spotty_labels = []
    for ablation in ['full', 'alpha0', 'beta0', 'gamma0', 'delta0']:
        data = df[(df['scenario'] == 'spotty_cellular') & (df['ablation'] == ablation)]['bytes_kb']
        spotty_data.append(data)
        spotty_labels.append(ablation)
    
    ax2.boxplot(spotty_data, labels=spotty_labels)
    ax2.set_title('Spotty Cellular: Bytes Delivered')
    ax2.set_ylabel('KB')
    ax2.tick_params(axis='x', rotation=45)
    
    # Nightly Wi-Fi Hit Rate
    nightly_hit_data = []
    for ablation in ['full', 'alpha0', 'beta0', 'gamma0', 'delta0']:
        data = df[(df['scenario'] == 'nightly_wifi') & (df['ablation'] == ablation)]['hit_rate']
        nightly_hit_data.append(data)
    
    ax3.boxplot(nightly_hit_data, labels=nightly_labels)
    ax3.set_title('Nightly Wi-Fi: Hit Rate')
    ax3.set_ylabel('Hit Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    # Spotty Cellular Hit Rate
    spotty_hit_data = []
    for ablation in ['full', 'alpha0', 'beta0', 'gamma0', 'delta0']:
        data = df[(df['scenario'] == 'spotty_cellular') & (df['ablation'] == ablation)]['hit_rate']
        spotty_hit_data.append(data)
    
    ax4.boxplot(spotty_hit_data, labels=spotty_labels)
    ax4.set_title('Spotty Cellular: Hit Rate')
    ax4.set_ylabel('Hit Rate')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('diagnostics/statistical_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Box plots saved to diagnostics/statistical_plots.png")

if __name__ == "__main__":
    results = perform_statistical_tests()
    create_box_plots()

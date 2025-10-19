#!/usr/bin/env python3
"""
Weight Sensitivity Analysis for ACORN Scheduler
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from acorn import compute_acorn_scores, Candidate

def test_weight_sensitivity():
    """Test how different weight combinations affect performance"""
    
    # Create a simple test scenario
    candidates = [
        Candidate(size_kb=100, deadline_ts=1000, reuse_score=0.8),
        Candidate(size_kb=200, deadline_ts=2000, reuse_score=0.6),
        Candidate(size_kb=150, deadline_ts=1500, reuse_score=0.9),
        Candidate(size_kb=300, deadline_ts=3000, reuse_score=0.4),
    ]
    
    now_ts = 0
    next_window_sec = 1000
    predicted_throughput_kBps = 50  # 50 KB/s
    
    print("=== WEIGHT SENSITIVITY ANALYSIS ===\n")
    
    # Test different alpha values (slack weight)
    print("1. ALPHA (slack) sensitivity:")
    print("   Testing alpha0 dominance in nightly_wifi scenario")
    print("   Fixed weights: beta=0.30, gamma=0.30, delta=0.10")
    print()
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for alpha in alphas:
        # Full configuration
        weights_full = (alpha, 0.30, 0.30, 0.10)
        scores_full, _ = compute_acorn_scores(
            candidates, now_ts=now_ts, next_window_sec=next_window_sec,
            predicted_throughput_kBps=predicted_throughput_kBps, weights=weights_full
        )
        
        # Alpha0 configuration (alpha=0)
        weights_alpha0 = (0.0, 0.30, 0.30, 0.10)
        scores_alpha0, _ = compute_acorn_scores(
            candidates, now_ts=now_ts, next_window_sec=next_window_sec,
            predicted_throughput_kBps=predicted_throughput_kBps, weights=weights_alpha0
        )
        
        # Calculate score differences
        score_diff = np.mean(scores_alpha0 - scores_full)
        
        results.append({
            'alpha': alpha,
            'full_mean_score': np.mean(scores_full),
            'alpha0_mean_score': np.mean(scores_alpha0),
            'score_difference': score_diff
        })
        
        print(f"   α={alpha:.1f}: FULL={np.mean(scores_full):.3f}, α0={np.mean(scores_alpha0):.3f}, Δ={score_diff:+.3f}")
    
    print()
    
    # Test different gamma values (size weight)
    print("2. GAMMA (size) sensitivity:")
    print("   Testing gamma0 dominance in spotty_cellular scenario")
    print("   Fixed weights: alpha=0.30, beta=0.30, delta=0.10")
    print()
    
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    gamma_results = []
    
    for gamma in gammas:
        # Full configuration
        weights_full = (0.30, 0.30, gamma, 0.10)
        scores_full, _ = compute_acorn_scores(
            candidates, now_ts=now_ts, next_window_sec=next_window_sec,
            predicted_throughput_kBps=predicted_throughput_kBps, weights=weights_full
        )
        
        # Gamma0 configuration (gamma=0)
        weights_gamma0 = (0.30, 0.30, 0.0, 0.10)
        scores_gamma0, _ = compute_acorn_scores(
            candidates, now_ts=now_ts, next_window_sec=next_window_sec,
            predicted_throughput_kBps=predicted_throughput_kBps, weights=weights_gamma0
        )
        
        # Calculate score differences
        score_diff = np.mean(scores_gamma0 - scores_full)
        
        gamma_results.append({
            'gamma': gamma,
            'full_mean_score': np.mean(scores_full),
            'gamma0_mean_score': np.mean(scores_gamma0),
            'score_difference': score_diff
        })
        
        print(f"   γ={gamma:.1f}: FULL={np.mean(scores_full):.3f}, γ0={np.mean(scores_gamma0):.3f}, Δ={score_diff:+.3f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Alpha sensitivity plot
    alphas = [r['alpha'] for r in results]
    alpha_diffs = [r['score_difference'] for r in results]
    ax1.plot(alphas, alpha_diffs, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Alpha (slack weight)')
    ax1.set_ylabel('Score Difference (α0 - FULL)')
    ax1.set_title('Alpha Sensitivity: α0 vs FULL')
    ax1.grid(True, alpha=0.3)
    
    # Gamma sensitivity plot
    gammas = [r['gamma'] for r in gamma_results]
    gamma_diffs = [r['score_difference'] for r in gamma_results]
    ax2.plot(gammas, gamma_diffs, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Gamma (size weight)')
    ax2.set_ylabel('Score Difference (γ0 - FULL)')
    ax2.set_title('Gamma Sensitivity: γ0 vs FULL')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagnostics/weight_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nWeight sensitivity plots saved to diagnostics/weight_sensitivity.png")
    
    return results, gamma_results

def analyze_normalization_effects():
    """Analyze how normalization changes between configurations"""
    
    print("\n3. NORMALIZATION ANALYSIS:")
    print("   How does removing one term affect other terms' normalization?")
    print()
    
    # Create candidates with varied values
    candidates = [
        Candidate(size_kb=50, deadline_ts=500, reuse_score=0.9),   # Small, urgent, high reuse
        Candidate(size_kb=200, deadline_ts=2000, reuse_score=0.3), # Large, not urgent, low reuse
        Candidate(size_kb=100, deadline_ts=1000, reuse_score=0.7), # Medium, medium urgent, medium reuse
        Candidate(size_kb=300, deadline_ts=3000, reuse_score=0.1), # Very large, not urgent, low reuse
    ]
    
    now_ts = 0
    next_window_sec = 1000
    predicted_throughput_kBps = 50
    
    # Extract raw values for analysis
    size_kb = np.array([c.size_kb for c in candidates])
    reuse_raw = np.array([c.reuse_score for c in candidates])
    deadlines = np.array([c.deadline_ts for c in candidates])
    
    # Calculate raw slack values
    dl_time_sec = size_kb / predicted_throughput_kBps
    slack_sec = (deadlines - now_ts) - dl_time_sec
    slack_nonneg = np.clip(slack_sec, 0.0, None)
    S_capped = np.clip(slack_nonneg, 0.0, next_window_sec)
    deadzone = 0.25 * next_window_sec
    S_eff = np.clip(S_capped, 0.0, deadzone)
    
    # Calculate raw connectivity values
    window_capacity_kb = predicted_throughput_kBps * next_window_sec
    frac_of_window = np.clip(size_kb / (window_capacity_kb + 1e-9), 0.0, 1.0)
    
    print("   Raw values before normalization:")
    print(f"   Sizes: {size_kb}")
    print(f"   Reuse: {reuse_raw}")
    print(f"   Slack:  {S_eff}")
    print(f"   Connectivity: {frac_of_window}")
    print()
    
    # Show normalization ranges
    def _norm01(x):
        x = x.astype(float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
            return np.zeros_like(x, dtype=float)
        return (x - lo) / (hi - lo)
    
    size_norm = _norm01(size_kb)
    reuse_norm = _norm01(reuse_raw)
    slack_norm = _norm01(S_eff)
    conn_norm = _norm01(frac_of_window)
    
    print("   Normalized values:")
    print(f"   Size norm: {size_norm}")
    print(f"   Reuse norm: {reuse_norm}")
    print(f"   Slack norm: {slack_norm}")
    print(f"   Conn norm: {conn_norm}")
    print()
    
    # Test FULL vs ALPHA0 scoring
    weights_full = (0.30, 0.30, 0.30, 0.10)
    scores_full, _ = compute_acorn_scores(
        candidates, now_ts=now_ts, next_window_sec=next_window_sec,
        predicted_throughput_kBps=predicted_throughput_kBps, weights=weights_full
    )
    
    weights_alpha0 = (0.0, 0.30, 0.30, 0.10)
    scores_alpha0, _ = compute_acorn_scores(
        candidates, now_ts=now_ts, next_window_sec=next_window_sec,
        predicted_throughput_kBps=predicted_throughput_kBps, weights=weights_alpha0
    )
    
    print("   Score comparison:")
    print(f"   FULL scores:  {scores_full}")
    print(f"   α0 scores:    {scores_alpha0}")
    print(f"   Difference:   {scores_alpha0 - scores_full}")
    print()
    
    # Analyze which candidates benefit from removing slack
    print("   Analysis:")
    for i, (c, full_score, alpha0_score) in enumerate(zip(candidates, scores_full, scores_alpha0)):
        diff = alpha0_score - full_score
        print(f"   Candidate {i}: {c.size_kb}KB, {c.deadline_ts}s, reuse={c.reuse_score:.1f}")
        print(f"     FULL={full_score:.3f}, α0={alpha0_score:.3f}, Δ={diff:+.3f}")
        if diff > 0:
            print(f"     → α0 BENEFITS this candidate (slack was hurting it)")
        elif diff < 0:
            print(f"     → FULL BENEFITS this candidate (slack was helping it)")
        else:
            print(f"     → No difference")
        print()

if __name__ == "__main__":
    alpha_results, gamma_results = test_weight_sensitivity()
    analyze_normalization_effects()

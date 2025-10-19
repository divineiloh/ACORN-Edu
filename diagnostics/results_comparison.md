# Results Comparison Analysis

## Data Source
- **File**: `results/data/bap_ablation_study_aggregates.csv`
- **Trials**: N=30 per configuration
- **Scenarios**: nightly_wifi, spotty_cellular
- **Configurations**: full, alpha0, beta0, gamma0, delta0

## Nightly Wi-Fi Scenario

| Configuration | Bytes Delivered (mean ± std) | Hit Rate (mean ± std) | Sample Size |
|---------------|------------------------------|------------------------|-------------|
| **FULL**      | 931,626.1 ± 14,470.9        | 0.8 ± 0.1             | 30          |
| **alpha0**    | 934,216.0 ± 11,257.9        | 0.7 ± 0.1             | 30          |
| **beta0**     | 926,343.3 ± 13,377.2        | 0.8 ± 0.1             | 30          |
| **gamma0**    | 931,903.9 ± 9,319.9         | 0.8 ± 0.1             | 30          |
| **delta0**    | 925,518.7 ± 13,118.9        | 0.7 ± 0.1             | 30          |

### Nightly Wi-Fi Delta Analysis
- **alpha0 vs FULL**: Δ Bytes = +0.28%, Δ Hit Rate = -12.5% ❌ **DOMINANCE DETECTED**
- **beta0 vs FULL**: Δ Bytes = -0.57%, Δ Hit Rate = 0.0% ✅ No dominance
- **gamma0 vs FULL**: Δ Bytes = +0.03%, Δ Hit Rate = 0.0% ✅ No dominance  
- **delta0 vs FULL**: Δ Bytes = -0.66%, Δ Hit Rate = -12.5% ✅ No dominance

## Spotty Cellular Scenario

| Configuration | Bytes Delivered (mean ± std) | Hit Rate (mean ± std) | Sample Size |
|---------------|------------------------------|------------------------|-------------|
| **FULL**      | 24,700.1 ± 1,120.0          | 0.2 ± 0.0             | 30          |
| **alpha0**    | 24,552.6 ± 972.8             | 0.2 ± 0.0             | 30          |
| **beta0**     | 24,472.2 ± 1,277.2           | 0.2 ± 0.0             | 30          |
| **gamma0**    | 25,050.2 ± 1,368.6           | 0.2 ± 0.0             | 30          |
| **delta0**    | 24,631.4 ± 849.2             | 0.2 ± 0.0             | 30          |

### Spotty Cellular Delta Analysis
- **alpha0 vs FULL**: Δ Bytes = -0.60%, Δ Hit Rate = 0.0% ✅ No dominance
- **beta0 vs FULL**: Δ Bytes = -0.92%, Δ Hit Rate = 0.0% ✅ No dominance
- **gamma0 vs FULL**: Δ Bytes = +1.42%, Δ Hit Rate = 0.0% ❌ **DOMINANCE DETECTED**
- **delta0 vs FULL**: Δ Bytes = -0.28%, Δ Hit Rate = 0.0% ✅ No dominance

## Dominance Summary

### ❌ DOMINANCE DETECTED:

1. **alpha0 in nightly_wifi**: 
   - Delivers MORE bytes (+0.28%) with LOWER hit rate (-12.5%)
   - This is NOT strict dominance (hit rate is worse)

2. **gamma0 in spotty_cellular**:
   - Delivers MORE bytes (+1.42%) with SAME hit rate (0.0%)
   - This IS strict dominance (better on one metric, same on other)

## Key Observations

1. **No strict dominance in nightly_wifi**: All ablations either deliver fewer bytes or have lower hit rates
2. **gamma0 dominates in spotty_cellular**: Removing size consideration actually improves performance
3. **Hit rates are very low in spotty_cellular**: All configurations achieve only 20% hit rate
4. **Large performance difference between scenarios**: 
   - Nightly Wi-Fi: ~930K bytes, 70-80% hit rate
   - Spotty Cellular: ~25K bytes, 20% hit rate

## Statistical Significance Questions

The dominance cases need statistical testing to determine if they're real or noise:
- alpha0 in nightly_wifi: Is the +0.28% bytes improvement significant?
- gamma0 in spotty_cellular: Is the +1.42% bytes improvement significant?

Both cases involve small percentage differences that could be within the margin of error for N=30 trials.

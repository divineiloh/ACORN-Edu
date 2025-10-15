# ACORN-Edu — Single-File Research Harness

This repo contains a **one-file** simulation harness for an offline-first delivery study.

## Standards
- Policies: `acorn` (deadline + reuse + inv size + availability), `LRU_whole` (whole-asset, deadline-ordered, LRU cache)
- Trials/Stats: **N=30** trials per scenario×policy; **two-sided 95% Student-t CIs**
- Units: **KB only** for sizes; any size column ends with `_kb`
- KB axes use plain integers with thousands separators; scientific tick offsets are disabled
- Outputs: `data/*.csv`, `data/run_metadata.json`, `figures/*.png`
- The script **self-verifies** and exits non-zero if anything is off

## Run (Python 3.11 recommended)
```bash
pip install -r requirements.txt
python acorn.py
```

# fast smoke (3 trials; skips statistical meaning)
python acorn.py --trials 3 --seed 2025

## Expected artifacts

* `data/bap_network_scenario_results.csv`
* `data/bap_network_scenario_aggregates.csv`
* `data/bap_ablation_study_results.csv`
* `data/bap_ablation_study_aggregates.csv`
* `data/run_metadata.json`
* `figures/bap_bytes_comparison.png`
* `figures/bap_hit_rate_comparison.png`
* `figures/bap_bytes_comparison_nightly_wifi.png`
* `figures/bap_hit_rate_comparison_nightly_wifi.png`
* `figures/bap_bytes_comparison_spotty_cellular.png`
* `figures/bap_hit_rate_comparison_spotty_cellular.png`
* `figures/ablation_hit_rate.png`
* `figures/ablation_bytes.png`
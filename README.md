# ACORN-Edu — Single-File Research Harness

This repo contains a **one-file** simulation harness for an offline-first delivery study.

## Standards
- Policies: `acorn` (deadline + reuse + inv size + availability), `lru_whole` (whole-asset, deadline-ordered, LRU cache)
- Trials/Stats: **N=30** trials per scenario×policy; **two-sided 95% Student-t CIs**
- Units: **KB only** for sizes; any size column ends with `_kb`
- Outputs: `data/*.csv`, `data/run_metadata.json`, `figures/*.png`
- The script **self-verifies** and exits non-zero if anything is off

## Run (Python 3.11 recommended)
```bash
pip install -r requirements.txt
python acorn.py
```

## Expected artifacts

* `data/bap_network_scenario_results.csv`
* `data/bap_network_scenario_aggregates.csv`
* `data/bap_ablation_study_results.csv`
* `data/bap_ablation_study_aggregates.csv`
* `data/run_metadata.json`
* `figures/bap_bytes_comparison.png`
* `figures/bap_hit_rate_comparison.png`
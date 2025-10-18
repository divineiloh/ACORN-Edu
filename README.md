# ACORN-Edu — Single-File Research Harness

This repo contains a **one-file** simulation harness for an offline-first delivery study.

## Standards
- Policies: `acorn` (slack-aware, normalized scheduler), `LRU_whole` (whole-asset, deadline-ordered, LRU cache)
- Trials/Stats: **N=30** trials per scenario×policy; **two-sided 95% Student-t CIs**
- Units: **KB only** for sizes; any size column ends with `_kb`
- KB axes use plain integers with thousands separators; scientific tick offsets are disabled
- Outputs: `data/*.csv`, `data/run_metadata.json`, `figures/*.png`
- The script **self-verifies** and exits non-zero if anything is off

## Policies

- **acorn**: Slack-aware, normalized priority scheduler with deadline urgency, reuse score, size preference, and network availability terms. Uses batch-normalized scoring with feasibility gating.
- **LRU_whole**: Whole-asset downloads with deadline-ordered queue and LRU cache eviction. Re-downloads entire assets on updates.

Results are means ±95% CI over N=30 seeded trials; **KB units only**.

## Names Used in Figures

**Scenarios:**
- *Nightly Wi-Fi Window* (internal id: `nightly_wifi`)
- *Bursty Cellular* (internal id: `spotty_cellular`)

**Policies:**
- *Acorn (Slack-Aware)* (internal id: `acorn`)
- *Whole-File LRU* (internal id: `LRU_whole`)

**Ablations (Acorn terms zeroed):**
- *Full (α,β,γ,δ)* → `full`
- *No-Deadline (α=0)* → `alpha0`
- *No-Reuse (β=0)* → `beta0`
- *No-Size (γ=0)* → `gamma0`
- *No-Connectivity (δ=0)* → `delta0`

## Run (Python 3.11 recommended)
```bash
pip install -r requirements.txt
python acorn.py
```

# fast smoke (3 trials; skips statistical meaning)
python acorn.py --trials 3 --seed-base 2025

## Reproduce Figures (N=30, 95% CI, KB units)

```bash
# Baseline comparisons
python acorn.py --policy acorn --trials 30 --seed-base 1337 --scenario nightly_wifi
python acorn.py --policy acorn --trials 30 --seed-base 1337 --scenario spotty_cellular
python acorn.py --policy LRU_whole --trials 30 --seed-base 1337 --scenario nightly_wifi
python acorn.py --policy LRU_whole --trials 30 --seed-base 1337 --scenario spotty_cellular

# Ablations (acorn only)
for f in alpha0 beta0 gamma0 delta0; do
  python acorn.py --policy acorn --trials 30 --seed-base 1337 --scenario nightly_wifi --$f
  python acorn.py --policy acorn --trials 30 --seed-base 1337 --scenario spotty_cellular --$f
done

# Reduce + plot
make figures
```

## Make Targets

- `make run_bench` → runs 4 baseline jobs (2 scenarios × 2 policies, N=30)
- `make ablation` → runs acorn ablations in both scenarios (N=30) and writes `ablation_summary.csv`
- `make figures` → generates the 4 final figures from the summaries

## Expected artifacts

```
results/
  run_meta.json
  bap/{bap_trials.csv, bap_summary.csv, fig_bap_kb_by_policy.png, fig_bap_hit_by_policy.png}
  ablation/{ablation_trials.csv, ablation_summary.csv, fig_ablation_kb.png, fig_ablation_hit.png}
```

* `data/bap_network_scenario_results.csv`
* `data/bap_network_scenario_aggregates.csv`
* `data/bap_ablation_study_results.csv`
* `data/bap_ablation_study_aggregates.csv`
* `data/run_metadata.json`
* `figures/bap_bytes_comparison.png` - KB transferred by Policy (Nightly Wi-Fi Window vs Bursty Cellular)
* `figures/bap_hit_rate_comparison.png` - Prefetch hit-rate (%) by Policy (Nightly Wi-Fi Window vs Bursty Cellular)
* `figures/bap_bytes_comparison_nightly_wifi.png` - KB transferred by Policy (Nightly Wi-Fi Window)
* `figures/bap_hit_rate_comparison_nightly_wifi.png` - Prefetch hit-rate (%) by Policy (Nightly Wi-Fi Window)
* `figures/bap_bytes_comparison_spotty_cellular.png` - KB transferred by Policy (Bursty Cellular)
* `figures/bap_hit_rate_comparison_spotty_cellular.png` - Prefetch hit-rate (%) by Policy (Bursty Cellular)
* `figures/ablation_hit_rate.png` - Ablations (Acorn): Prefetch Hit-Rate (%)
* `figures/ablation_bytes.png` - Ablations (Acorn): KB transferred

All figures show means ±95% CI over N=30 seeded trials; **KB units only**.
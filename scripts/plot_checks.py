import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12})

os.makedirs("figures", exist_ok=True)

def bars_with_ci(ax, x, means, lows, ups, width, label):
    err = np.vstack([means - lows, ups - means])
    ax.bar(x, means, width, yerr=err, capsize=4, label=label)

# --- Network ---
net = pd.read_csv("data/bap_network_scenario_results.csv")
scenarios = ["nightly_wifi", "spotty_cellular"]
schedulers = ["DeadlineFIFO_Downloader", "AcornScheduler"]
net["scenario"] = pd.Categorical(net["scenario"], categories=scenarios, ordered=True)
net["scheduler"] = pd.Categorical(net["scheduler"], categories=schedulers, ordered=True)
net = net.sort_values(["scenario", "scheduler"]).reset_index(drop=True)

# Bytes (KB)
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
x = np.arange(len(scenarios))
w = 0.36
for i, sch in enumerate(schedulers):
    sub = net[net["scheduler"] == sch]
    m = sub["bytes_mean_(KB)"].to_numpy()
    l = sub["bytes_ci_lower_(KB)"].to_numpy()
    u = sub["bytes_ci_upper_(KB)"].to_numpy()
    bars_with_ci(ax, x + i * w, m, l, u, w, sch)
ax.set_xticks(x + w / 2)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Total Bytes Transferred (KB)")
ax.set_title("ACORN-Edu: Bandwidth Usage (95% CI)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/acorn_network_bytes.png")

# Hit rate (%)
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
for i, sch in enumerate(schedulers):
    sub = net[net["scheduler"] == sch]
    m = sub["hit_rate_mean_(%)"].clip(0, 100).to_numpy()
    l = sub["hit_rate_ci_lower_(%)"].clip(0, 100).to_numpy()
    u = sub["hit_rate_ci_upper_(%)"].clip(0, 100).to_numpy()
    bars_with_ci(ax, x + i * w, m, l, u, w, sch)
ax.set_xticks(x + w / 2)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Prefetch Hit Rate (%)")
ax.set_ylim(0, 105)
ax.set_title("ACORN-Edu: Prefetch Hit Rate (95% CI)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/acorn_network_hitrate.png")

# --- Ablation (sort labels for stable order) ---
abl = pd.read_csv("data/bap_ablation_study_results.csv").sort_values("ablation_config")

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
m = abl["hit_rate_mean_(%)"].clip(0, 100).to_numpy()
l = abl["hit_rate_ci_lower_(%)"].clip(0, 100).to_numpy()
u = abl["hit_rate_ci_upper_(%)"].clip(0, 100).to_numpy()
err = np.vstack([m - l, u - m])
ax.bar(abl["ablation_config"], m, yerr=err, capsize=4)
ax.set_ylabel("Prefetch Hit Rate (%)")
ax.set_ylim(0, 105)
ax.set_title("ACORN-Edu Ablation: Hit Rate (95% CI)")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("figures/acorn_ablation_hitrate.png")

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
m = abl["bytes_mean_(KB)"].to_numpy()
l = abl["bytes_ci_lower_(KB)"].to_numpy()
u = abl["bytes_ci_upper_(KB)"].to_numpy()
err = np.vstack([m - l, u - m])
ax.bar(abl["ablation_config"], m, yerr=err, capsize=4)
ax.set_ylabel("Bytes Transferred (KB)")
ax.set_title("ACORN-Edu Ablation: Bandwidth (95% CI)")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig("figures/acorn_ablation_bytes.png")

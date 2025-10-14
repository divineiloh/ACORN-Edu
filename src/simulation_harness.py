# ACORN-Edu: Research Paper Simulation Harness (Gold Standard)
#
# This script provides a complete, runnable simulation of the core components
# described in the paper "Deadline-Aware Prefetch and Tamper-Evident Offline
# Exams for Low-Connectivity Learning."
#
# ==============================================================================
# README
# ==============================================================================
#
# PURPOSE:
# To generate the empirical results, tables, and figures needed for a
# submission to a journal like IEEE Access. This script is fully self-contained.
#
# REQUIREMENTS:
# - Python 3.7+
# - numpy
# - scipy (for confidence intervals)
# - matplotlib (optional, for generating plots)
#
# To install dependencies:
# pip install -r requirements.txt
#
# HOW TO RUN:
# This script is designed to be called by the main `run_pipeline.sh`.
# To run manually: `PYTHONPATH=. python3 src/simulation_harness.py`
#
# WHAT IT PRODUCES:
# The script will create two directories, `./output/data` and `./output/figures`:
# (Note: The pipeline script creates the parent /output directory)
# ==============================================================================

import hashlib
import random
import json
import os
import csv
import numpy as np
import datetime
from scipy import stats

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# --- Configuration ---
try:
    from .config import (
        NUM_SIMULATION_RUNS,
        RANDOM_SEED_BASE,
        OUTPUT_DATA_DIR,
        OUTPUT_FIGURES_DIR,
        AVG_CHUNK_SIZE,
        ASSETS,
        NETWORK_PROFILES,
        SCHEDULERS_TO_TEST,
        ABLATION_CONFIGS,
        DELTA_MAP,
        OAEC_NUM_CHAINS,
        OAEC_TIMESTAMP_BASE,
        CAG_STANDARDS,
        CAG_LIBRARY,
        CAG_GROUND_TRUTH,
        MOCK_PRIVATE_KEY,
        MOCK_PUBLIC_KEY,
    )
except ImportError:
    from config import (
        NUM_SIMULATION_RUNS,
        RANDOM_SEED_BASE,
        OUTPUT_DATA_DIR,
        OUTPUT_FIGURES_DIR,
        AVG_CHUNK_SIZE,
        ASSETS,
        NETWORK_PROFILES,
        SCHEDULERS_TO_TEST,
        ABLATION_CONFIGS,
        DELTA_MAP,
        OAEC_NUM_CHAINS,
        OAEC_TIMESTAMP_BASE,
        CAG_STANDARDS,
        CAG_LIBRARY,
        CAG_GROUND_TRUTH,
        MOCK_PRIVATE_KEY,
        MOCK_PUBLIC_KEY,
    )


# --- Utility Functions and Classes ---


class MockSigner:
    """A mock class to simulate device-key signing and verification."""

    def __init__(self):
        self.private_key = MOCK_PRIVATE_KEY
        self.public_key = MOCK_PUBLIC_KEY

    def sign(self, data):
        """Simulates signing data with a private key."""
        signature_base = f"{data}|{self.private_key}"
        return hashlib.sha256(signature_base.encode()).hexdigest()

    def verify(self, data, signature, public_key):
        """Simulates verifying a signature with a public key."""
        if public_key != self.public_key:
            return False
        expected_signature = hashlib.sha256(
            f"{data}|{self.private_key}".encode()
        ).hexdigest()
        return signature == expected_signature


def print_table(headers, rows):
    """Prints a list of lists as a formatted table."""
    col_widths = [
        max(len(str(item)) for item in col) for col in zip(*([headers] + rows))
    ]
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    print(header_line)
    print(separator)
    for row in rows:
        row_line = " | ".join(f"{str(r):<{w}}" for r, w in zip(row, col_widths))
        print(row_line)


def write_csv(filename, headers, rows):
    """Writes a list of lists to a CSV file."""
    filepath = os.path.join(OUTPUT_DATA_DIR, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Saved results to '{filepath}'")


# --- Section 1: BAP (Bandwidth-Aware Packaging) Simulation ---


class BAP_Simulator:
    """Simulates the BAP pipeline to measure bandwidth efficiency."""

    def __init__(self, network_profile="nightly_wifi"):
        self.avg_chunk_size = AVG_CHUNK_SIZE
        self.assets = ASSETS
        self.network_profile = network_profile
        self.network_trace = self._get_network_trace(network_profile)
        self.chunk_db = {}
        self._chunk_all_assets()

    def _get_network_trace(self, profile):
        """Returns a network trace: a list of (bytes/sec, type) tuples."""
        if profile == "nightly_wifi":
            return self._define_nightly_wifi_trace()
        elif profile == "spotty_cellular":
            return self._define_spotty_cellular_trace()
        else:
            raise ValueError("Unknown network profile")

    def _define_nightly_wifi_trace(self):
        """Stable but time-limited Wi-Fi. Speed reduced for realism."""
        trace = []
        for day in range(5):
            for hour in range(24):
                if 2 <= hour < 4:
                    trace.append((64 * 1024, "wifi"))  # 0.5 Mbps
                else:
                    trace.append((0, "none"))
        return trace

    def _define_spotty_cellular_trace(self):
        """Erratic, low-bandwidth cellular."""
        trace = []
        for day in range(5):
            for hour in range(24):
                if 8 <= hour <= 22 and random.random() < 0.5:
                    trace.append((64 * 1024, "cellular"))
                else:
                    trace.append((0, "none"))
        return trace

    def _chunk_asset(self, asset):
        """Simulate content-defined chunking."""
        num_chunks = max(1, round(asset["size"] / self.avg_chunk_size))
        chunks = [f"{asset['id']}_chunk_{i}" for i in range(num_chunks)]
        if "Video" in asset["id"]:
            for i in range(int(num_chunks * 0.1)):
                chunks[i] = f"shared_video_intro_chunk_{i}"
        return chunks

    def _chunk_all_assets(self):
        for asset in self.assets:
            self.chunk_db[asset["id"]] = self._chunk_asset(asset)

    def _calculate_hit_rate(self, download_log, assets):
        hits = 0
        for asset in assets:
            bytes_at_deadline = sum(
                size
                for aid, hour, size in download_log
                if aid == asset["id"] and hour < asset["deadline_hours"]
            )
            if bytes_at_deadline >= 0.95 * asset["size"]:
                hits += 1
        return (hits / len(assets)) * 100 if assets else 0

    def run_simulation_pass(self, scheduler_type, weights=None):
        """Runs a single simulation pass for a given scheduler type."""
        assets_by_deadline = sorted(self.assets, key=lambda x: x["deadline_hours"])

        download_log = []
        bytes_transferred = 0

        if scheduler_type == "AcornScheduler":
            downloaded_chunks = set()
        else:  # File-granular baselines
            remaining_bytes = {a["id"]: a["size"] for a in self.assets}

        for hour, (speed, net_type) in enumerate(self.network_trace):
            bandwidth_this_hour = speed * 3600
            if bandwidth_this_hour == 0:
                continue

            if scheduler_type == "AcornScheduler":
                # Chunk-granular download logic
                net_factor = 1.0 if net_type == "wifi" else 0.3
                bw = bandwidth_this_hour
                while bw > self.avg_chunk_size:
                    best_asset, max_priority = None, -1
                    for asset in self.assets:
                        if any(
                            c not in downloaded_chunks
                            for c in self.chunk_db[asset["id"]]
                        ):
                            priority = self._calculate_priority(
                                asset, hour, weights, net_factor
                            )
                            if priority > max_priority:
                                max_priority, best_asset = priority, asset
                    if not best_asset:
                        break

                    chunk_to_download = next(
                        (
                            c
                            for c in self.chunk_db[best_asset["id"]]
                            if c not in downloaded_chunks
                        ),
                        None,
                    )
                    if chunk_to_download and chunk_to_download not in downloaded_chunks:
                        bytes_transferred += self.avg_chunk_size
                        downloaded_chunks.add(chunk_to_download)
                        download_log.append(
                            (best_asset["id"], hour, self.avg_chunk_size)
                        )
                        bw -= self.avg_chunk_size

            else:  # File-granular download logic for baselines
                bw = bandwidth_this_hour
                for asset in assets_by_deadline:
                    if remaining_bytes[asset["id"]] > 0:
                        take = min(remaining_bytes[asset["id"]], bw)
                        remaining_bytes[asset["id"]] -= take
                        bytes_transferred += take
                        download_log.append((asset["id"], hour, take))
                        bw -= take
                        if bw <= 0:
                            break

        hit_rate = self._calculate_hit_rate(download_log, self.assets)
        return {"bytes": bytes_transferred, "hit_rate": hit_rate}

    def _calculate_priority(self, asset, current_hour, weights, net_factor):
        time_to_deadline = max(0.1, asset["deadline_hours"] - current_hour)
        norm_deadline = 1.0 / time_to_deadline
        norm_size = 1.0 / (asset["size"] / (1024 * 1024) + 1e-6)
        norm_reuse = asset["reuse_score"]
        alpha, beta, gamma, delta = weights
        return (
            (alpha * norm_deadline)
            + (beta * norm_reuse)
            + (gamma * norm_size)
            + (delta * net_factor)
        )

    def run_delta_sync_analysis(self):
        delta_rows = []
        for asset in self.assets:
            key = next((k for k in DELTA_MAP if k in asset["id"]), "Data")
            d1, d2 = DELTA_MAP[key]
            base_update_bytes = asset["size"] * 2
            cdc_update_bytes = asset["size"] * (d1 + d2)
            savings = (
                100 * (1 - cdc_update_bytes / base_update_bytes)
                if base_update_bytes > 0
                else 0
            )
            delta_rows.append([asset["id"], asset["size"], d1, d2, savings])

        write_csv(
            "delta_sync_per_asset.csv",
            ["asset_id", "size_(B)", "delta1_ratio", "delta2_ratio", "savings_(%)"],
            delta_rows,
        )


# --- Visualization Functions (with 95% CI Error Bars) ---
def plot_bap_network_comparison(results_df):
    if not MATPLOTLIB_AVAILABLE:
        return

    labels = results_df["network_profile"].unique()
    schedulers = results_df["scheduler"].unique()
    x = np.arange(len(labels))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "ACORN-Edu Performance Comparison with Baselines (30 Runs, 95% CI)", fontsize=16
    )

    # Bytes Transferred Plot
    for i, scheduler in enumerate(schedulers):
        subset = results_df[results_df["scheduler"] == scheduler]
        means = subset["bytes_mean_mb"]
        ci_lower = subset["bytes_ci_lower_mb"]
        ci_upper = subset["bytes_ci_upper_mb"]
        errors = [means - ci_lower, ci_upper - means]
        ax1.bar(x + i * width, means, width, yerr=errors, label=scheduler, capsize=4)
    ax1.set_ylabel("Total Bytes Transferred (MB)")
    ax1.set_title("Bandwidth Usage")
    ax1.set_xticks(x + width * (len(schedulers) - 1) / 2)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # Hit Rate Plot
    for i, scheduler in enumerate(schedulers):
        subset = results_df[results_df["scheduler"] == scheduler]
        means = subset["hit_rate_mean"]
        ci_lower = subset["hit_rate_ci_lower"]
        ci_upper = subset["hit_rate_ci_upper"]
        errors = [means - ci_lower, ci_upper - means]
        ax2.bar(x + i * width, means, width, yerr=errors, label=scheduler, capsize=4)
    ax2.set_ylabel("Prefetch Hit Rate (%)")
    ax2.set_title("Prefetch Hit Rate (Deadline Met)")
    ax2.set_xticks(x + width * (len(schedulers) - 1) / 2)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.set_ylim(0, 105)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(OUTPUT_FIGURES_DIR, "bap_network_comparison.png")
    plt.savefig(filename)
    print(f"\nSaved network comparison chart to '{filename}'")


def plot_bap_ablation_study(results_df):
    if not MATPLOTLIB_AVAILABLE:
        return

    labels = results_df["ablation_config"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "ACORN Scheduler Ablation Study (Spotty Cellular, 30 Runs, 95% CI)", fontsize=16
    )

    # Hit Rate Plot
    means_hr = results_df["hit_rate_mean"]
    ci_lower_hr = results_df["hit_rate_ci_lower"]
    ci_upper_hr = results_df["hit_rate_ci_upper"]
    errors_hr = [means_hr - ci_lower_hr, ci_upper_hr - means_hr]
    ax1.bar(labels, means_hr, yerr=errors_hr, color="c", capsize=5)
    ax1.set_ylabel("Prefetch Hit Rate (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Impact on Hit Rate")
    ax1.tick_params(axis="x", rotation=25)

    # Bytes Transferred Plot
    means_b = results_df["bytes_mean_mb"]
    ci_lower_b = results_df["bytes_ci_lower_mb"]
    ci_upper_b = results_df["bytes_ci_upper_mb"]
    errors_b = [means_b - ci_lower_b, ci_upper_b - means_b]
    ax2.bar(labels, means_b, yerr=errors_b, color="m", capsize=5)
    ax2.set_ylabel("Bytes Transferred (MB)")
    ax2.set_title("Impact on Bandwidth Usage")
    ax2.tick_params(axis="x", rotation=25)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = os.path.join(OUTPUT_FIGURES_DIR, "bap_ablation_study.png")
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved ablation study chart to '{filename}'")


# --- OAEC & CAG Simulators (Unchanged) ---
class OAEC_Simulator:
    def __init__(self, num_chains=OAEC_NUM_CHAINS):
        self.num_chains = num_chains
        self.signer = MockSigner()
        self.t0 = OAEC_TIMESTAMP_BASE

    def run_simulation(self):
        print("\n--- Running OAEC Simulation ---")
        chains = [self._generate_event_chain() for _ in range(self.num_chains)]
        tampered = [self._tamper_chain(c) for c in chains]
        tp, fp, tn, fn = 0, 0, 0, 0
        for chain in chains:
            if self._verify_chain(chain):
                tn += 1
            else:
                fp += 1
        for chain in tampered:
            if not self._verify_chain(chain):
                tp += 1
            else:
                fn += 1
        headers = ["actual", "predicted_tampered", "predicted_valid"]
        rows = [["tampered", tp, fn], ["valid", fp, tn]]
        print_table(headers, rows)
        write_csv("oaec_confusion_matrix.csv", headers, rows)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"\n- Tamper Detection Rate (Recall): {recall * 100:.1f}%")
        print(f"- False Positive Rate: {fpr * 100:.1f}%")

    def _generate_event_chain(self):
        chain, h_prev, counter = [], "0" * 64, 0
        for i, event in enumerate(["start", "ans1", "save", "ans2", "submit"]):
            payload = {"event": event, "ts": self.t0 + i}
            counter += 1
            record = f"{h_prev}|{json.dumps(payload, sort_keys=True)}|{counter}"
            h_curr = hashlib.sha256(record.encode()).hexdigest()
            chain.append(
                {
                    "p": payload,
                    "h_p": h_prev,
                    "mc": counter,
                    "h_c": h_curr,
                    "sig": self.signer.sign(h_curr),
                }
            )
            h_prev = h_curr
        return chain

    def _tamper_chain(self, chain):
        t_chain = [dict(c) for c in chain]
        act = random.choice(["del", "mod", "re"])
        if act == "del" and len(t_chain) > 1:
            del t_chain[1]
        elif act == "mod":
            t_chain[2]["p"]["event"] = "cheat"
        elif act == "re" and len(t_chain) > 2:
            t_chain[1], t_chain[2] = t_chain[2], t_chain[1]
        return t_chain

    def _verify_chain(self, chain):
        h_prev, last_mc = "0" * 64, 0
        for r in chain:
            if (
                r["h_p"] != h_prev
                or not self.signer.verify(r["h_c"], r["sig"], self.signer.public_key)
                or r["mc"] <= last_mc
            ):
                return False
            rec = f"{r['h_p']}|{json.dumps(r['p'], sort_keys=True)}|{r['mc']}"
            h_exp = hashlib.sha256(rec.encode()).hexdigest()
            if r["h_c"] != h_exp:
                return False
            h_prev, last_mc = r["h_c"], r["mc"]
        return True


class CAG_Simulator:
    def __init__(self):
        self.std = CAG_STANDARDS
        self.lib = CAG_LIBRARY
        self.gt = CAG_GROUND_TRUTH

    def run_simulation(self):
        print("\n--- Running CAG & Compiler Simulation ---")
        validated = {m["c"] for m in self.lib.values() if m.get("a")}
        tp, fp, fn = (
            len(validated & self.gt),
            len(validated - self.gt),
            len(self.gt - validated),
        )
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(
            f"\nCompiler Metrics: Precision: {prec:.2f}, "
            f"Recall: {rec:.2f}, F1: {f1:.2f}"
        )
        write_csv(
            "cag_metrics.csv", ["precision", "recall", "f1_score"], [[prec, rec, f1]]
        )
        headers = ["id", "status", "content"]
        rows = []
        for oid in self.std:
            status = (
                "Validated"
                if oid in validated
                else (
                    "Covered"
                    if any(m["c"] == oid for m in self.lib.values())
                    else "GAP"
                )
            )
            content = next((n for n, m in self.lib.items() if m["c"] == oid), "---")
            rows.append([oid, status, content])
        print_table(headers, rows)
        write_csv("cag_auditor_table.csv", headers, rows)


def calculate_cis(data, confidence=0.95):
    """Calculate mean and 95% confidence interval."""
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        return (
            np.mean(a) if n > 0 else 0,
            np.mean(a) if n > 0 else 0,
            np.mean(a) if n > 0 else 0,
        )
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)

    print(
        "=" * 60
        + f"\n ACORN-Edu Research Simulation (Running {NUM_SIMULATION_RUNS} trials) \n"
        + "=" * 60
    )

    # --- Section 1: BAP (Enhanced with Statistical Rigor) ---
    print("\n--- Running BAP Simulation: Network Scenarios ---")
    network_profiles = NETWORK_PROFILES
    schedulers_to_test = SCHEDULERS_TO_TEST

    network_results = []
    for run in range(NUM_SIMULATION_RUNS):
        print(f"  Running trial {run + 1}/{NUM_SIMULATION_RUNS}...")
        random.seed(RANDOM_SEED_BASE + run)
        np.random.seed(RANDOM_SEED_BASE + run)
        for profile in network_profiles:
            sim = BAP_Simulator(network_profile=profile)
            for scheduler in schedulers_to_test:
                res = sim.run_simulation_pass(scheduler, weights=(0.5, 0.2, 0.2, 0.1))
                network_results.append(
                    [profile, scheduler, res["bytes"], res["hit_rate"]]
                )

    # Aggregate network results with 95% CIs
    network_agg = {}
    for profile in network_profiles:
        for scheduler in schedulers_to_test:
            key = (profile, scheduler)
            runs = [r for r in network_results if r[0] == profile and r[1] == scheduler]
            bytes_m, bytes_l, bytes_u = calculate_cis([r[2] for r in runs])
            hr_m, hr_l, hr_u = calculate_cis([r[3] for r in runs])
            network_agg[key] = {
                "bytes_mean": bytes_m,
                "bytes_ci_lower": bytes_l,
                "bytes_ci_upper": bytes_u,
                "hit_rate_mean": hr_m,
                "hit_rate_ci_lower": hr_l,
                "hit_rate_ci_upper": hr_u,
            }

    # Format for CSV and plotting
    network_csv_rows, plot_df_data = [], []
    for (profile, scheduler), data in network_agg.items():
        network_csv_rows.append(
            [
                profile,
                scheduler,
                data["bytes_mean"],
                data["bytes_ci_lower"],
                data["bytes_ci_upper"],
                data["hit_rate_mean"],
                data["hit_rate_ci_lower"],
                data["hit_rate_ci_upper"],
            ]
        )
        plot_df_data.append(
            {
                "network_profile": profile,
                "scheduler": scheduler,
                "bytes_mean_mb": data["bytes_mean"] / 1e6,
                "bytes_ci_lower_mb": data["bytes_ci_lower"] / 1e6,
                "bytes_ci_upper_mb": data["bytes_ci_upper"] / 1e6,
                "hit_rate_mean": data["hit_rate_mean"],
                "hit_rate_ci_lower": data["hit_rate_ci_lower"],
                "hit_rate_ci_upper": data["hit_rate_ci_upper"],
            }
        )
    network_headers = [
        "scenario",
        "scheduler",
        "bytes_mean_(B)",
        "bytes_ci_lower_(B)",
        "bytes_ci_upper_(B)",
        "hit_rate_mean_(%)",
        "hit_rate_ci_lower_(%)",
        "hit_rate_ci_upper_(%)",
    ]
    write_csv("bap_network_scenario_results.csv", network_headers, network_csv_rows)
    import pandas as pd

    plot_bap_network_comparison(pd.DataFrame(plot_df_data))

    print("\n--- Running BAP Simulation: Ablation Study ---")
    ablation_configs = ABLATION_CONFIGS
    ablation_results = []
    for run in range(NUM_SIMULATION_RUNS):
        print(f"  Running trial {run + 1}/{NUM_SIMULATION_RUNS}...")
        random.seed(RANDOM_SEED_BASE + run)
        np.random.seed(RANDOM_SEED_BASE + run)
        sim = BAP_Simulator(network_profile="spotty_cellular")
        for name, weights in ablation_configs.items():
            res = sim.run_simulation_pass("AcornScheduler", weights=weights)
            ablation_results.append([name, res["bytes"], res["hit_rate"]])

    ablation_agg = {}
    for name in ablation_configs.keys():
        runs = [r for r in ablation_results if r[0] == name]
        bytes_m, bytes_l, bytes_u = calculate_cis([r[1] for r in runs])
        hr_m, hr_l, hr_u = calculate_cis([r[2] for r in runs])
        ablation_agg[name] = {
            "bytes_mean": bytes_m,
            "bytes_ci_lower": bytes_l,
            "bytes_ci_upper": bytes_u,
            "hit_rate_mean": hr_m,
            "hit_rate_ci_lower": hr_l,
            "hit_rate_ci_upper": hr_u,
        }

    ablation_csv_rows, plot_df_data_ablation = [], []
    for name, data in ablation_agg.items():
        ablation_csv_rows.append(
            [
                name,
                data["bytes_mean"],
                data["bytes_ci_lower"],
                data["bytes_ci_upper"],
                data["hit_rate_mean"],
                data["hit_rate_ci_lower"],
                data["hit_rate_ci_upper"],
            ]
        )
        plot_df_data_ablation.append(
            {
                "ablation_config": name,
                "bytes_mean_mb": data["bytes_mean"] / 1e6,
                "bytes_ci_lower_mb": data["bytes_ci_lower"] / 1e6,
                "bytes_ci_upper_mb": data["bytes_ci_upper"] / 1e6,
                "hit_rate_mean": data["hit_rate_mean"],
                "hit_rate_ci_lower": data["hit_rate_ci_lower"],
                "hit_rate_ci_upper": data["hit_rate_ci_upper"],
            }
        )

    ablation_headers = [
        "ablation_config",
        "bytes_mean_(B)",
        "bytes_ci_lower_(B)",
        "bytes_ci_upper_(B)",
        "hit_rate_mean_(%)",
        "hit_rate_ci_lower_(%)",
        "hit_rate_ci_upper_(%)",
    ]
    write_csv("bap_ablation_study_results.csv", ablation_headers, ablation_csv_rows)
    plot_bap_ablation_study(pd.DataFrame(plot_df_data_ablation))

    BAP_Simulator().run_delta_sync_analysis()
    print("\n" + "=" * 60)
    OAEC_Simulator().run_simulation()
    print("\n" + "=" * 60)
    CAG_Simulator().run_simulation()

    run_metadata = {
        "seed_base": RANDOM_SEED_BASE,
        "num_runs": NUM_SIMULATION_RUNS,
        "run_timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(OUTPUT_DATA_DIR, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=4)
    metadata_path = os.path.join(OUTPUT_DATA_DIR, 'run_metadata.json')
    print(f"\nSaved run metadata to '{metadata_path}'")

    print(
        "\n"
        + "=" * 60
        + "\n              Simulation Complete                  \n"
        + "=" * 60
    )

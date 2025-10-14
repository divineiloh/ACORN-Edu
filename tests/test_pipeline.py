import pandas as pd
import os


def test_outputs_exist():
    """Test that all required output files exist."""
    assert os.path.exists("data/bap_network_scenario_results.csv")
    assert os.path.exists("data/bap_ablation_study_results.csv")
    assert os.path.exists("data/oaec_confusion_matrix.csv")
    assert os.path.exists("data/delta_sync_per_asset.csv")
    assert os.path.exists("data/cag_metrics.csv")
    assert os.path.exists("data/cag_auditor_table.csv")


def test_units_kb():
    """Test that CSV headers use KB units only."""
    net = pd.read_csv("data/bap_network_scenario_results.csv")
    abl = pd.read_csv("data/bap_ablation_study_results.csv")

    # Check network CSV headers
    kb_cols = {"bytes_mean_(KB)", "bytes_ci_lower_(KB)", "bytes_ci_upper_(KB)"}
    assert kb_cols.issubset(set(net.columns))

    # Check ablation CSV headers
    assert kb_cols.issubset(set(abl.columns))

    # Ensure no MB/GB columns exist
    all_cols = set(net.columns) | set(abl.columns)
    assert not any("mb" in col.lower() or "gb" in col.lower() for col in all_cols)


def test_profiles_diverge():
    """Test that network profiles produce different results."""
    df = pd.read_csv("data/bap_network_scenario_results.csv")
    acorn = df[df["scheduler"] == "AcornScheduler"]

    # Check we have both profiles
    assert len(acorn["scenario"].unique()) >= 2

    # Check for meaningful differences
    wifi_data = acorn[acorn["scenario"] == "nightly_wifi"]
    cellular_data = acorn[acorn["scenario"] == "spotty_cellular"]

    # Either bytes or hit rate should differ significantly
    bytes_diff = abs(
        wifi_data["bytes_mean_(KB)"].iloc[0] - cellular_data["bytes_mean_(KB)"].iloc[0]
    )
    hit_rate_diff = abs(
        wifi_data["hit_rate_mean_(%)"].iloc[0]
        - cellular_data["hit_rate_mean_(%)"].iloc[0]
    )

    assert (
        bytes_diff > 1.0 or hit_rate_diff > 1.0
    ), f"Profiles too similar: bytes_diff={bytes_diff}, hit_rate_diff={hit_rate_diff}"


def test_ablation_varies():
    """Test that ablation configurations produce different results."""
    abl = pd.read_csv("data/bap_ablation_study_results.csv")

    # Check we have multiple configurations
    assert len(abl) > 1

    # Check for variation in results
    hit_rate_variation = abl["hit_rate_mean_(%)"].nunique() > 1
    bytes_variation = abl["bytes_mean_(KB)"].nunique() > 1

    assert hit_rate_variation or bytes_variation, "Ablation results are identical"


def test_oaec_no_fp_fn():
    """Test that OAEC has 0 false positives and 0 false negatives."""
    cm = pd.read_csv("data/oaec_confusion_matrix.csv")

    tampered_row = cm[cm["actual"] == "tampered"].iloc[0]
    valid_row = cm[cm["actual"] == "valid"].iloc[0]

    # Check for 0 false positives and 0 false negatives
    assert (
        tampered_row["predicted_valid"] == 0
    ), f"False negatives detected: {tampered_row['predicted_valid']}"
    assert (
        valid_row["predicted_tampered"] == 0
    ), f"False positives detected: {valid_row['predicted_tampered']}"


def test_figures_exist():
    """Test that all required figures exist."""
    required_figures = [
        "figures/acorn_bytes_comparison.png",
        "figures/acorn_hit_rate_comparison.png",
        "figures/acorn_ablation_hit_rate.png",
        "figures/acorn_ablation_bytes.png",
    ]

    for fig_path in required_figures:
        assert os.path.exists(fig_path), f"Missing figure: {fig_path}"


def test_scheduler_names():
    """Test that scheduler names are correct (no Sabi references)."""
    net = pd.read_csv("data/bap_network_scenario_results.csv")
    abl = pd.read_csv("data/bap_ablation_study_results.csv")

    # Check scheduler names
    schedulers = set(net["scheduler"].unique())
    assert "AcornScheduler" in schedulers
    assert "DeadlineFIFO_Downloader" in schedulers
    assert not any(
        "sabi" in s.lower() for s in schedulers
    ), "Found Sabi references in scheduler names"

    # Check ablation config names
    ablation_configs = set(abl["ablation_config"].unique())
    assert not any(
        "sabi" in config.lower() for config in ablation_configs
    ), "Found Sabi references in ablation configs"


def test_data_quality():
    """Test data quality and consistency."""
    net = pd.read_csv("data/bap_network_scenario_results.csv")
    abl = pd.read_csv("data/bap_ablation_study_results.csv")

    # Check for reasonable value ranges
    assert net["hit_rate_mean_(%)"].min() >= 0
    assert net["hit_rate_mean_(%)"].max() <= 100
    assert abl["hit_rate_mean_(%)"].min() >= 0
    assert abl["hit_rate_mean_(%)"].max() <= 100

    # Check for positive byte values
    assert net["bytes_mean_(KB)"].min() > 0
    assert abl["bytes_mean_(KB)"].min() > 0

    # Check confidence intervals are reasonable
    for df in [net, abl]:
        for col in ["bytes_mean_(KB)", "hit_rate_mean_(%)"]:
            mean_col = col
            lower_col = col.replace("mean", "ci_lower")
            upper_col = col.replace("mean", "ci_upper")

            if lower_col in df.columns and upper_col in df.columns:
                assert (
                    df[lower_col] <= df[mean_col]
                ).all(), f"CI lower > mean for {col}"
                assert (
                    df[mean_col] <= df[upper_col]
                ).all(), f"CI upper < mean for {col}"

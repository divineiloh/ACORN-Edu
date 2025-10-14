import pandas as pd
import os


def test_outputs_exist():
    assert os.path.exists("data/bap_network_scenario_results.csv")
    assert os.path.exists("data/bap_ablation_study_results.csv")
    assert os.path.exists("data/oaec_confusion_matrix.csv")


def test_headers_kb_only():
    net = pd.read_csv("data/bap_network_scenario_results.csv")
    kb_cols = {"bytes_mean_(KB)", "bytes_ci_lower_(KB)", "bytes_ci_upper_(KB)"}
    assert kb_cols.issubset(set(net.columns))


def test_profiles_diverge_for_acorn():
    df = pd.read_csv("data/bap_network_scenario_results.csv")
    acorn = df[df["scheduler"] == "AcornScheduler"]
    assert len(acorn["scenario"].unique()) >= 2
    # some divergence in either bytes or hit rate
    diff = abs(acorn.groupby("scenario")["hit_rate_mean_(%)"].mean().diff().fillna(0)).sum() + \
           abs(acorn.groupby("scenario")["bytes_mean_(KB)"].mean().diff().fillna(0)).sum()
    assert diff > 0.01


def test_ablation_varies():
    abl = pd.read_csv("data/bap_ablation_study_results.csv")
    varies = (abl["hit_rate_mean_(%)"].nunique() > 1) or (abl["bytes_mean_(KB)"].nunique() > 1)
    assert varies


def test_oaec_no_fp_fn():
    cm = pd.read_csv("data/oaec_confusion_matrix.csv")
    tampered = cm[cm["actual"] == "tampered"].iloc[0]
    valid = cm[cm["actual"] == "valid"].iloc[0]
    assert tampered["predicted_valid"] == 0
    assert valid["predicted_tampered"] == 0

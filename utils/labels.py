# utils/labels.py
SCENARIO_LABEL = {
    "nightly_wifi": "Nightly Wi-Fi Window",
    "spotty_cellular": "Bursty Cellular",
}
POLICY_LABEL = {
    "acorn": "Acorn (Slack-Aware)",
    "LRU_whole": "Whole-File LRU",
}
ABLATION_LABEL = {
    "full": "Full (α,β,γ,δ)",
    "alpha0": "No-Deadline (α=0)",
    "beta0":  "No-Reuse (β=0)",
    "gamma0": "No-Size (γ=0)",
    "delta0": "No-Connectivity (δ=0)",
}

def label_scenario(x: str) -> str: return SCENARIO_LABEL.get(x, x)
def label_policy(x: str) -> str:   return POLICY_LABEL.get(x, x)
def label_ablation(x: str) -> str: return ABLATION_LABEL.get(x, x)

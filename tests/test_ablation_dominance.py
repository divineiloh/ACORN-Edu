# tests/test_ablation_dominance.py
import pandas as pd
import sys
from pathlib import Path

def test_ablation_dominance():
    """Test that no ablation strictly dominates FULL on both metrics in any scenario."""
    results_path = Path("results/ablation/ablation_summary_by_scenario.csv")
    
    if not results_path.exists():
        print("ERROR: Ablation results file not found. Run the ablation study first.")
        sys.exit(1)
    
    agg = pd.read_csv(results_path)
    
    # Get full model results
    full_results = agg[agg.ablation == "full"].set_index("scenario")
    
    # Check each scenario
    for scenario in agg.scenario.unique():
        if scenario not in full_results.index:
            print(f"ERROR: No full model results found for scenario {scenario}")
            sys.exit(1)
            
        full_kb = float(full_results.loc[scenario, "mean_bytes_kb"])
        full_hit = float(full_results.loc[scenario, "mean_hit_rate"])
        
        # Check each ablation in this scenario
        scenario_data = agg[agg.scenario == scenario]
        for _, row in scenario_data.iterrows():
            if row["ablation"] == "full":
                continue
                
            ablation_kb = float(row["mean_bytes_kb"])
            ablation_hit = float(row["mean_hit_rate"])
            
            # Check for strict dominance: ablation is better on BOTH metrics
            kb_better = ablation_kb < full_kb  # Lower KB is better
            hit_better = ablation_hit > full_hit  # Higher hit rate is better
            
            if kb_better and hit_better:
                print(f"ERROR: Ablation {row['ablation']} strictly dominates FULL in {scenario}")
                print(f"  FULL: {full_kb:.1f} KB, {full_hit:.3f} hit rate")
                print(f"  {row['ablation']}: {ablation_kb:.1f} KB, {ablation_hit:.3f} hit rate")
                sys.exit(1)
    
    print("PASS: No ablation strictly dominates FULL on both metrics")

if __name__ == "__main__":
    test_ablation_dominance()

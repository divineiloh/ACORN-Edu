# ACORN-Edu Results Analysis
# Generates LaTeX tables and statistical analysis from simulation results

import os
import csv
import numpy as np
from scipy import stats
try:
    from .config import OUTPUT_DATA_DIR, CONFIDENCE_LEVEL
except ImportError:
    from config import OUTPUT_DATA_DIR, CONFIDENCE_LEVEL


def calculate_cis(data, confidence=CONFIDENCE_LEVEL):
    """Calculate mean and confidence interval."""
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


def generate_latex_tables():
    """Generate LaTeX tables from simulation results."""
    print("\n--- Generating LaTeX Tables ---")

    # Read network scenario results
    network_file = os.path.join(OUTPUT_DATA_DIR, "bap_network_scenario_results.csv")
    if os.path.exists(network_file):
        print("Network scenario results found - generating LaTeX table...")
        # TODO: Implement LaTeX table generation

    # Read ablation study results
    ablation_file = os.path.join(OUTPUT_DATA_DIR, "bap_ablation_study_results.csv")
    if os.path.exists(ablation_file):
        print("Ablation study results found - generating LaTeX table...")
        # TODO: Implement LaTeX table generation

    print("LaTeX table generation complete.")


if __name__ == "__main__":
    generate_latex_tables()

#!/usr/bin/env python3
"""
Unit tests for KB-only compliance and data quality.
"""

import os
import pandas as pd
import numpy as np


def test_all_size_columns_kb_only():
    """Test that all size columns use KB units and _kb suffix."""
    csv_files = [
        "data/bap_network_scenario_results.csv",
        "data/bap_ablation_study_results.csv",
        "data/delta_sync_per_asset.csv",
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # Find all size-related columns
            size_columns = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['bytes', 'size', 'bandwidth']
            )]
            
            for col in size_columns:
                # Check that column name ends with _kb
                assert col.endswith('_kb') or col.endswith('_(KB)'), (
                    f"Size column '{col}' must end with '_kb' or '_(KB)' for KB units"
                )
                
                # Check that values are numeric and positive
                assert df[col].dtype in ['int64', 'float64'], (
                    f"Size column '{col}' must be numeric"
                )
                assert (df[col] >= 0).all(), (
                    f"Size column '{col}' must contain non-negative values"
                )


def test_hit_rate_columns_percent():
    """Test that hit rate columns use percentage units."""
    csv_files = [
        "data/bap_network_scenario_results.csv",
        "data/bap_ablation_study_results.csv",
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # Find all hit rate columns
            hit_rate_columns = [col for col in df.columns if 'hit_rate' in col.lower()]
            
            for col in hit_rate_columns:
                # Check that column name indicates percentage
                assert '%' in col or 'percent' in col.lower(), (
                    f"Hit rate column '{col}' must indicate percentage units"
                )
                
                # Check that values are in reasonable percentage range
                assert (df[col] >= 0).all(), (
                    f"Hit rate column '{col}' must be non-negative"
                )
                assert (df[col] <= 100).all(), (
                    f"Hit rate column '{col}' must be <= 100%"
                )


def test_confidence_intervals_present():
    """Test that confidence interval columns are present and valid."""
    csv_files = [
        "data/bap_network_scenario_results.csv",
        "data/bap_ablation_study_results.csv",
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # Check for CI columns
            ci_columns = [col for col in df.columns if 'ci_' in col.lower()]
            assert len(ci_columns) > 0, (
                f"CSV file '{csv_file}' must contain confidence interval columns"
            )
            
            # Check that CI values are non-negative
            for col in ci_columns:
                assert (df[col] >= 0).all(), (
                    f"Confidence interval column '{col}' must be non-negative"
                )


def test_decimal_precision():
    """Test that numeric values are rounded to 1 decimal place."""
    csv_files = [
        "data/bap_network_scenario_results.csv",
        "data/bap_ablation_study_results.csv",
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # Check numeric columns for proper decimal precision
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                # Check that values don't have excessive decimal places
                for value in df[col].dropna():
                    if isinstance(value, float):
                        decimal_places = len(str(value).split('.')[-1]) if '.' in str(value) else 0
                        assert decimal_places <= 1, (
                            f"Column '{col}' has values with more than 1 decimal place: {value}"
                        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])

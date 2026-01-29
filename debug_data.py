import pandas as pd
from utils.data_loader import load_data
from utils.standardize import merge_datasets

print("Loading data...")
try:
    datasets = load_data(None, use_dummy=True)
    # Simulate user scenario: drop 'date' column from all datasets
    for k, v in datasets.items():
        if 'date' in v.columns:
            v = v.drop(columns=['date'])
        datasets[k] = v
        
    print(f"Loaded {len(datasets)} datasets.")
    for name, df in datasets.items():
        print(f"Dataset: {name}, Shape: {df.shape}")
        print(df.head())
        print(df.columns)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

print("\nMerging datasets...")
try:
    unified_df = merge_datasets(datasets)
    print(f"Unified DF Shape: {unified_df.shape}")
    print(unified_df.head())
except Exception as e:
    print(f"Error merging data: {e}")

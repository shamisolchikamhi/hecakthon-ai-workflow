import pandas as pd
import numpy as np

def standardize_columns(df):
    """
    Detects common column names and renames them to a standard schema.
    """
    df.columns = df.columns.str.lower().str.strip()
    
    # Mapping of potential names to standard names
    rename_map = {
        'date': 'date',
        'time': 'date',
        'day': 'date',
        'period': 'date',
        'store': 'store_name',
        'store name': 'store_name',
        'store_name': 'store_name',
        'shop': 'store_name',
        'outlet': 'store_name',
        'location name': 'store_name',
        'locationname': 'store_name',
        'impressions': 'impressions',
        'impression': 'impressions',
        'imps': 'impressions',
        'clicks': 'clicks',
        'ctr': 'ctr',
        'click through rate': 'ctr',
        'total visits': 'total_visits',
        'visits': 'total_visits',
        'exposed visits': 'exposed_visits',
        'exposed': 'exposed_visits',
        'search': 'search_metric',
        'total search': 'search_metric',
        'search volume': 'search_metric',
        'search score': 'search_metric',
        'sessions': 'web_sessions',
        'web sessions': 'web_sessions',
        'page views': 'web_sessions',
        'conversions': 'web_conversions',
        'web conversions': 'web_conversions',
        'actual': 'actual',
        'rank': 'rank',
        'sector': 'sector',
        'brand': 'brand_name'
    }
    
    # Also handle typical variations like "Store " -> "store_name" via regex-ish approach if needed, 
    # but for now explicit map is safer.
    
    df = df.rename(columns=rename_map)
    return df

def parse_dates(df, date_col='date'):
    """
    Parses the date column to datetime objects.
    """
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df

def standardize_store_names(df, store_col='store_name'):
    """
    Standardises store names (trim, uppercase, remove double spaces).
    """
    if store_col in df.columns:
        df[store_col] = df[store_col].astype(str).str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)
    return df

def calculate_derived_metrics(df):
    """
    Calculates CTR or Clicks if missing.
    """
    if 'impressions' in df.columns:
        if 'clicks' in df.columns and 'ctr' not in df.columns:
             # avoided division by zero by replacing 0 with NaN temporarily or handling it
             # For MVP simple division
             df['ctr'] = df['clicks'] / df['impressions'].replace(0, np.nan)
             df['ctr'] = df['ctr'].fillna(0)
             
        elif 'ctr' in df.columns and 'clicks' not in df.columns:
            df['clicks'] = (df['impressions'] * df['ctr']).round().astype(int)
            
    return df

def merge_datasets(datasets):
    """
    Merges a dictionary of dataframes into a unified dataframe keyed by date and store_name.
    Uses full outer join to preserve all data.
    """
    processed_dfs = []
    
    # First pass: Check if date exists in majority or mandatory? 
    # Logic: If date missing in ALL, we merge on store_name only.
    # If date missing in SOME, that's tricky. For MVP, we'll auto-detect keys per logic below.
    
    for name, df in datasets.items():
        if df is None or df.empty:
            continue
            
        # Pre-process
        df = standardize_columns(df)
        df = parse_dates(df)
        df = standardize_store_names(df)
        df = calculate_derived_metrics(df)
        
        # Check for key columns
        # Critical: "store_name" is mandatory. "date" is preferred but optional if missing (aggregated data).
        
        if 'store_name' not in df.columns:
            print(f"Warning: Dataset {name} missing keys 'store_name'. Columns: {df.columns}")
            continue
            
        # If 'screen' is present, it might be a key too.
        
        processed_dfs.append(df)

    if not processed_dfs:
        return pd.DataFrame()
        
    # Merge all using outer join
    from functools import reduce
    
    def merge_outer(left, right):
        # Determine common keys for this pair
        left_keys = set(left.columns)
        right_keys = set(right.columns)
        
        # Potential keys
        potential_keys = ['date', 'store_name', 'screen', 'campaign', 'campaignname']
        
        # Find intersection
        on_keys = [k for k in potential_keys if k in left_keys and k in right_keys]
        
        # Fallback if only one exists (should include store_name at minimum)
        if not on_keys:
             # Should not happen if store_name is mandatory
             on_keys = ['store_name']
             
        return pd.merge(left, right, on=on_keys, how='outer', suffixes=('', '_dup'))
        
    unified_df = reduce(merge_outer, processed_dfs)
    return unified_df

def generate_missingness_report(df):
    """
    Creates a missingness report showing missing values per metric per store.
    """
    if df.empty:
        return pd.DataFrame()
        
    # Group by store and count nulls
    missingness = df.groupby('store_name').apply(lambda x: x.isnull().sum())
    return missingness

import pandas as pd
from utils.dummy_data import generate_dummy_data

def load_data(uploaded_files, use_dummy=True):
    """
    Loads data from uploaded files or generates dummy data if specified.
    Returns a dictionary of dataframes.
    """
    datasets = {}
    
    # Map file names/keys to expected dataset names
    # In Streamlit, uploaded_files is usually a dict or list
    
    if use_dummy:
        return generate_dummy_data()
        
    if not uploaded_files:
        return {}

    for name, file in uploaded_files.items():
        if file is not None:
            try:
                df = pd.read_csv(file)
                datasets[name] = df
            except Exception as e:
                print(f"Error loading {name}: {e}")
                
    return datasets

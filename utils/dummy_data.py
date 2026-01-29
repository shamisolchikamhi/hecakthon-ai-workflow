import pandas as pd
import numpy as np
from datetime import timedelta, date

def generate_dummy_data(start_date=None, end_date=None, stores=None):
    """
    Generates dummy data for impressions, visits, search, and web analytics.
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=30)
    if end_date is None:
        end_date = date.today()
    
    if stores is None:
        stores = [f"Store {i}" for i in range(1, 6)]
        
    date_range = pd.date_range(start=start_date, end=end_date)
    data = []

    for store in stores:
        for d in date_range:
            # Base logic
            impressions = np.random.randint(1000, 5000)
            ctr = np.random.uniform(0.005, 0.02)
            clicks = int(impressions * ctr)
            
            total_visits = np.random.randint(50, 200)
            # Ensure exposed visits is realistic (10-14% of total usually, but add noise for QA checks)
            exposed_share = np.random.uniform(0.08, 0.16) 
            exposed_visits = int(total_visits * exposed_share)
            
            # Introducting some QA failures/warnings occasionally
            if np.random.random() < 0.02:
                exposed_visits = total_visits + 5 # Fail: exposed > total
            if np.random.random() < 0.02:
                impressions = -100 # Fail: negative value

            search_metric = np.random.randint(10, 100)
            web_sessions = np.random.randint(20, 100)
            web_conversions = int(web_sessions * np.random.uniform(0.01, 0.05))
            

            row = {
                "date": d,
                "store_name": store,
                "impressions": impressions,
                "clicks": clicks,
                "ctr": ctr,
                "total_visits": total_visits,
                "exposed_visits": exposed_visits,
                "search_metric": search_metric,
                "web_sessions": web_sessions,
                "web_conversions": web_conversions,

            }
            data.append(row)
            
    df = pd.DataFrame(data)
    
    # Split into separate dataframes to mimic CSV uploads
    impressions_df = df[["date", "store_name", "impressions", "clicks", "ctr"]].copy()
    visits_df = df[["date", "store_name", "total_visits", "exposed_visits"]].copy()
    search_df = df[["date", "store_name", "search_metric"]].copy()
    web_df = df[["date", "store_name", "web_sessions", "web_conversions"]].copy()
    
    return {
        "impressions": impressions_df,
        "visits": visits_df,
        "search": search_df,
        "web": web_df
    }


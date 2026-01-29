import pandas as pd
import numpy as np

class QARules:
    def __init__(self, df):
        self.df = df
        self.results = []
        self.summary = {
            "total_fails": 0,
            "total_warns": 0,
            "affected_stores": set(),
            "top_rules": {}
        }

    def log_result(self, store, rule_name, severity, value_json, reason):
        self.results.append({
            "store_name": store,
            "rule_name": rule_name,
            "severity": severity,
            "metric_values_json": value_json,
            "reason": reason
        })
        
        if severity == "FAIL":
            self.summary["total_fails"] += 1
            self.summary["affected_stores"].add(store)
        elif severity == "WARN":
            self.summary["total_warns"] += 1
            self.summary["affected_stores"].add(store)
            
        self.summary["top_rules"][rule_name] = self.summary["top_rules"].get(rule_name, 0) + 1

    def run_all(self):
        if self.df.empty:
            return pd.DataFrame(), self.summary
            
        self.check_exposed_vs_total()
        self.check_exposed_share()
        self.check_impressions_exposed_mismatch()
        self.check_ctr_sanity()
        self.check_negative_values()
        self.check_visits_plausibility()
        self.check_completeness()
        
        qa_df = pd.DataFrame(self.results)
        return qa_df, self.summary

    def check_exposed_vs_total(self):
        # Rule 1: FAIL if exposed_visits > total_visits
        if 'exposed_visits' in self.df.columns and 'total_visits' in self.df.columns:
            mask = self.df['exposed_visits'] > self.df['total_visits']
            failures = self.df[mask]
            for _, row in failures.iterrows():
                self.log_result(
                    row['store_name'],
                    "Exposed > Total",
                    "FAIL",
                    f"Exposed: {row['exposed_visits']}, Total: {row['total_visits']}",
                    "Exposed visits cannot be greater than total visits."
                )

    def check_exposed_share(self):
        # Rule 2: Exposed Share
        if 'exposed_visits' in self.df.columns and 'total_visits' in self.df.columns:
            # Calculate share, handling division by zero
            share = self.df['exposed_visits'] / self.df['total_visits'].replace(0, np.nan)
            
            # FAIL if < 0.05 or > 0.20
            fail_mask = (share < 0.05) | (share > 0.20)
            for _, row in self.df[fail_mask].iterrows():
                self.log_result(
                    row['store_name'],
                    "Exposed Share Out of Bounds",
                    "FAIL",
                    f"Share: {share[row.name]:.2%}",
                    "Exposed share is < 5% or > 20%."
                )

            # WARN if outside 0.10 - 0.14 but inside FAIL bounds (handled by exclusion logic or just simple check)
            # Actually simpler: Check warns first then fails, or check disjoint sets.
            warn_mask = ((share < 0.10) | (share > 0.14)) & (~fail_mask)
            for _, row in self.df[warn_mask].iterrows():
                self.log_result(
                   row['store_name'],
                   "Exposed Share Warning",
                   "WARN",
                   f"Share: {share[row.name]:.2%}",
                   "Exposed share is outside optimal range 10-14%."
                )

    def check_impressions_exposed_mismatch(self):
        # Rule 3: Impressions vs Exposed Mismatch (Quartiles per store total)
        if 'impressions' in self.df.columns and 'exposed_visits' in self.df.columns:
            store_totals = self.df.groupby('store_name')[['impressions', 'exposed_visits']].sum()
            
            # skip if not enough data
            if len(store_totals) < 4:
                return

            imp_q1 = store_totals['impressions'].quantile(0.25)
            exp_q3 = store_totals['exposed_visits'].quantile(0.75)
            
            mismatch = store_totals[
                (store_totals['impressions'] <= imp_q1) & 
                (store_totals['exposed_visits'] >= exp_q3)
            ]
            
            for store in mismatch.index:
                self.log_result(
                    store,
                    "Impression-Exposed Mismatch",
                    "WARN",
                    f"Imp: {mismatch.loc[store, 'impressions']}, Exp: {mismatch.loc[store, 'exposed_visits']}",
                    "Low impressions (Bottom Q) but high exposed visits (Top Q)."
                )

    def check_ctr_sanity(self):
        # Rule 4: CTR Sanity
        if 'ctr' in self.df.columns:
            # WARN if ctr > 0.10 or ctr < 0.001
            mask = (self.df['ctr'] > 0.10) | (self.df['ctr'] < 0.001)
            # Filter out 0 CTR if that's considered valid (e.g. no clicks), but prompt says < 0.001
            # Assuming strictly positive checks for simplicity, but let's allow 0 if 0 clicks.
            # actually if ctr < 0.001 usually implies non-zero but very small. 0 is a separate case usually.
            # Let's stick to prompt literal: WARN if ctr < 0.001 (includes 0)
            
            for _, row in self.df[mask].iterrows():
                 self.log_result(
                    row['store_name'],
                    "CTR Sanity",
                    "WARN",
                    f"CTR: {row['ctr']:.4f}",
                    "CTR is extremely high (>10%) or low (<0.1%)."
                )

    def check_negative_values(self):
        # Rule 5: Negative values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (self.df[col] < 0).any():
                neg_rows = self.df[self.df[col] < 0]
                for _, row in neg_rows.iterrows():
                    self.log_result(
                        row['store_name'],
                        "Negative Value",
                        "FAIL",
                        f"{col}: {row[col]}",
                        f"Found negative value for {col}."
                    )

    def check_visits_plausibility(self):
        # Rule 6: Visits Plausibility vs Baseline
        # Create dummy baseline (median of history or simple dummy)
        if 'total_visits' in self.df.columns:
            store_medians = self.df.groupby('store_name')['total_visits'].median()
            
            for store, median in store_medians.items():
                store_data = self.df[self.df['store_name'] == store]
                mask = store_data['total_visits'] > (2.5 * median)
                
                for _, row in store_data[mask].iterrows():
                    self.log_result(
                        store,
                        "Visits Spike",
                        "WARN",
                        f"Visits: {row['total_visits']}, Baseline: {median}",
                        "Visits > 2.5x baseline (median)."
                    )

    def check_completeness(self):
        # Rule 7: Completeness (>30% missing days)
        if 'date' in self.df.columns:
            # Assuming date range is full range of min/max per store, or global?
            # Let's check vs global min/max
            min_date = self.df['date'].min()
            max_date = self.df['date'].max()
            full_days = (max_date - min_date).days + 1
            
            store_counts = self.df.groupby('store_name')['date'].nunique()
            
            for store, count in store_counts.items():
                missing_days = full_days - count
                if missing_days / full_days > 0.30:
                    self.log_result(
                        store,
                        "Data Completeness",
                        "WARN",
                        f"Missing Days: {missing_days}/{full_days}",
                        "Store has > 30% missing days in range."
                    )

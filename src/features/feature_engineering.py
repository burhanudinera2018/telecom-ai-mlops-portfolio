import pandas as pd
import numpy as np

class ChurnFeatureBuilder:
    @staticmethod
    def add_arpu_features(df):
        """Average Revenue Per User features"""
        df['arpu'] = df['total_charges'] / (df['tenure'] + 1)
        df['monthly_arpu'] = df['monthly_charges']
        df['arpu_growth'] = df.groupby('customer_id')['monthly_charges'].pct_change().fillna(0)
        return df
    
    @staticmethod
    def add_engagement_features(df):
        """Engagement-based features"""
        df['avg_monthly_gb'] = df['monthly_avg_gb']
        df['support_interaction_rate'] = df['num_calls_to_care'] / (df['tenure'] + 1)
        df['complaint_rate'] = df['num_complaints'] / (df['tenure'] + 1)
        return df
    
    @staticmethod
    def add_contract_features(df):
        """Contract risk features"""
        df['is_monthly_contract'] = (df['contract_type'] == 'Month-to-month').astype(int)
        df['has_electronic_check'] = (df['payment_method'] == 'Electronic check').astype(int)
        return df
    
    @staticmethod
    def build_features(df):
        df = ChurnFeatureBuilder.add_arpu_features(df)
        df = ChurnFeatureBuilder.add_engagement_features(df)
        df = ChurnFeatureBuilder.add_contract_features(df)
        
        # Drop unnecessary columns
        cols_to_drop = ['customer_id']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        return df

class ARPUFeatureBuilder:
    @staticmethod
    def build_features(df):
        """Features specific for ARPU prediction"""
        df = df.copy()
        df['log_total_charges'] = np.log1p(df['total_charges'])
        df['tenure_squared'] = df['tenure'] ** 2
        return df
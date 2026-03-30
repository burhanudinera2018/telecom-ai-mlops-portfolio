# src/monitoring/model_monitor.py
import pandas as pd
import numpy as np
import joblib
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    def __init__(self, model_path='model_registry/churn_model/model.pkl',
                 reference_data_path='data/processed/reference_data.csv'):
        
        # Load model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}")
            self.model = None
        
        # Load or create reference data
        if os.path.exists(reference_data_path):
            self.reference_data = pd.read_csv(reference_data_path)
            print(f"✅ Reference data loaded from {reference_data_path}")
        else:
            print(f"⚠️ Reference data not found at {reference_data_path}")
            print(f"📝 Creating reference data from raw data...")
            
            # Create reference data from raw data
            raw_data_path = 'data/raw/customer_data.csv'
            if os.path.exists(raw_data_path):
                df = pd.read_csv(raw_data_path)
                # Sample 1000 rows or use all if less than 1000
                sample_size = min(1000, len(df))
                self.reference_data = df.sample(n=sample_size, random_state=42)
                
                # Save for future use
                os.makedirs(os.path.dirname(reference_data_path), exist_ok=True)
                self.reference_data.to_csv(reference_data_path, index=False)
                print(f"✅ Reference data created and saved to {reference_data_path}")
                print(f"   Shape: {self.reference_data.shape}")
            else:
                print(f"❌ Raw data not found at {raw_data_path}")
                print(f"   Please run 'python scripts/generate_data.py' first")
                self.reference_data = None
    
    def calculate_psi(self, current_data, reference_data=None):
        """Population Stability Index"""
        if reference_data is None:
            reference_data = self.reference_data
        
        if reference_data is None or current_data is None:
            print("⚠️ Cannot calculate PSI: missing data")
            return {}
        
        psi_values = {}
        
        # Only calculate for common numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numeric_cols if col in reference_data.columns]
        
        for col in common_cols:
            try:
                # Bin data
                ref_bins = pd.qcut(reference_data[col], q=10, duplicates='drop')
                cur_bins = pd.qcut(current_data[col], q=10, duplicates='drop')
                
                ref_dist = ref_bins.value_counts(normalize=True).sort_index()
                cur_dist = cur_bins.value_counts(normalize=True).sort_index()
                
                # Align indices
                all_bins = ref_dist.index.union(cur_dist.index)
                ref_dist = ref_dist.reindex(all_bins, fill_value=0)
                cur_dist = cur_dist.reindex(all_bins, fill_value=0)
                
                # Calculate PSI
                # Avoid division by zero
                cur_dist = cur_dist.replace(0, 0.0001)
                ref_dist = ref_dist.replace(0, 0.0001)
                
                psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
                psi_values[col] = psi
            except Exception as e:
                print(f"⚠️ Could not calculate PSI for {col}: {e}")
                psi_values[col] = np.nan
        
        return psi_values
    
    def ks_test(self, current_data, reference_data=None):
        """Kolmogorov-Smirnov test for drift detection"""
        if reference_data is None:
            reference_data = self.reference_data
        
        if reference_data is None or current_data is None:
            print("⚠️ Cannot perform KS test: missing data")
            return {}
        
        ks_results = {}
        
        # Only test common numeric columns
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numeric_cols if col in reference_data.columns]
        
        for col in common_cols:
            try:
                ks_stat, p_value = ks_2samp(
                    reference_data[col].dropna(), 
                    current_data[col].dropna()
                )
                ks_results[col] = {'statistic': ks_stat, 'p_value': p_value}
            except Exception as e:
                print(f"⚠️ Could not perform KS test for {col}: {e}")
                ks_results[col] = {'statistic': np.nan, 'p_value': np.nan}
        
        return ks_results
    
    def check_drift(self, current_data, threshold=0.1):
        """Main drift detection function"""
        if self.reference_data is None:
            print("❌ No reference data available for drift detection")
            return {}
        
        if current_data is None:
            print("❌ No current data provided")
            return {}
        
        print("\n" + "="*60)
        print("📊 MODEL DRIFT MONITORING REPORT")
        print("="*60)
        
        # Calculate PSI
        print("\n📈 Population Stability Index (PSI):")
        psi = self.calculate_psi(current_data)
        drift_alerts = {}
        
        for col, psi_value in psi.items():
            status = "✅" if psi_value <= threshold else "⚠️" if psi_value <= 0.25 else "❌"
            print(f"   {status} {col:25} PSI = {psi_value:.4f}")
            
            if psi_value > threshold:
                drift_alerts[col] = psi_value
        
        # KS Test for significant features
        print("\n📉 Kolmogorov-Smirnov Test (p-values):")
        ks_results = self.ks_test(current_data)
        
        for col, result in ks_results.items():
            p_value = result['p_value']
            status = "✅" if p_value > 0.05 else "⚠️"
            print(f"   {status} {col:25} p-value = {p_value:.4f}")
        
        print("\n" + "="*60)
        
        if drift_alerts:
            print(f"\n⚠️ DRIFT ALERT: Significant drift detected in features: {list(drift_alerts.keys())}")
        else:
            print("\n✅ No significant drift detected")
        
        print("="*60)
        
        return drift_alerts
    
    def generate_report(self, current_data, output_path='monitoring_report.json'):
        """Generate complete monitoring report"""
        if self.reference_data is None or current_data is None:
            print("❌ Cannot generate report: missing data")
            return
        
        report = {
            'psi_values': self.calculate_psi(current_data),
            'ks_test': self.ks_test(current_data),
            'drift_alerts': self.check_drift(current_data),
            'reference_shape': self.reference_data.shape,
            'current_shape': current_data.shape
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Report saved to {output_path}")
        return report


if __name__ == '__main__':
    print("🔍 Initializing Model Monitor...")
    monitor = ModelMonitor()
    
    if monitor.reference_data is not None:
        # Simulate current data (sample from reference data)
        current_sample = monitor.reference_data.sample(min(100, len(monitor.reference_data)))
        print(f"\n📊 Analyzing {len(current_sample)} samples...")
        
        # Check for drift
        drift = monitor.check_drift(current_sample)
        
        # Generate report
        monitor.generate_report(current_sample)
    else:
        print("\n❌ Cannot run monitoring: No data available")
        print("Please ensure data/raw/customer_data.csv exists")
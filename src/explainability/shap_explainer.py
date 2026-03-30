import shap
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, model_path='model_registry/churn_model/model.pkl', 
                 preprocessor_path='model_registry/churn_model/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.explainer = None
    
    def fit_explainer(self, X_sample):
        """Initialize SHAP explainer with background data"""
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_sample)
        return self.shap_values
    
    def plot_global_importance(self, save_path='model_cards/shap_summary.png'):
        shap.summary_plot(self.shap_values, X_sample, show=False)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def explain_prediction(self, X_instance):
        """Local explanation for a single prediction"""
        shap_values = self.explainer.shap_values(X_instance)
        return shap_values
    
    def generate_report(self, X_test, y_test, output_path='model_cards/shap_report.html'):
        """Generate HTML explanation report"""
        # Implementation for detailed report
        pass
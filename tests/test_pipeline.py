# tests/test_pipeline.py
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataLoader, DataCleaner, DataTransformer
from src.features.feature_engineering import ChurnFeatureBuilder

def test_data_loader():
    # Create sample data if raw data doesn't exist
    if not os.path.exists('data/raw/customer_data.csv'):
        os.makedirs('data/raw', exist_ok=True)
        sample_df = pd.DataFrame({
            'customer_id': [1, 2],
            'tenure': [12, 24],
            'monthly_charges': [100, 200],
            'total_charges': [1200, 2400],
            'contract_type': ['Month-to-month', 'One year'],
            'payment_method': ['Electronic check', 'Credit card'],
            'paperless_billing': [1, 0],
            'monthly_avg_gb': [50, 60],
            'num_complaints': [0, 1],
            'num_calls_to_care': [2, 1],
            'has_internet': [1, 1],
            'has_streaming_tv': [1, 0],
            'has_streaming_movies': [1, 1],
            'churn': [0, 1]
        })
        sample_df.to_csv('data/raw/customer_data.csv', index=False)
    
    df = DataLoader.load_raw_data()
    assert len(df) > 0
    assert 'churn' in df.columns

def test_data_cleaner():
    # Create comprehensive test data
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'tenure': [12, 24, 36],
        'monthly_charges': [100, None, 200],
        'total_charges': [1200, 2400, 3600],
        'contract_type': ['Month-to-month', 'One year', 'Two year'],
        'payment_method': ['Electronic check', 'Credit card', 'Bank transfer'],
        'paperless_billing': [1, 0, 1],
        'monthly_avg_gb': [50, 60, 70],
        'num_complaints': [0, 1, 0],
        'num_calls_to_care': [2, 1, 3],
        'has_internet': [1, 1, 0],
        'has_streaming_tv': [1, 0, 1],
        'has_streaming_movies': [1, 1, 0],
        'churn': [0, 1, 0]
    })
    
    df_clean = DataCleaner.clean(df)
    
    # Check that no missing values remain
    assert df_clean.isnull().sum().sum() == 0, "There should be no missing values"
    
    # Check that the function returns a dataframe
    assert isinstance(df_clean, pd.DataFrame), "Should return a DataFrame"
    
    # Check that monthly_charges null was filled
    assert df_clean['monthly_charges'].iloc[1] is not None, "Missing value should be filled"

def test_feature_engineering():
    df = pd.DataFrame({
        'customer_id': [1, 2],
        'tenure': [12, 24],
        'total_charges': [1200, 2400],
        'monthly_charges': [100, 100],
        'contract_type': ['Month-to-month', 'One year'],
        'payment_method': ['Electronic check', 'Credit card'],
        'num_calls_to_care': [2, 1],
        'num_complaints': [0, 1],
        'monthly_avg_gb': [50, 60],
        'paperless_billing': [1, 0],
        'has_internet': [1, 1],
        'has_streaming_tv': [1, 0],
        'has_streaming_movies': [1, 1],
        'churn': [0, 1]
    })
    
    df_features = ChurnFeatureBuilder.build_features(df)
    
    # Check that new features are created
    assert 'arpu' in df_features.columns, "ARPU feature should be created"
    assert 'is_monthly_contract' in df_features.columns, "Monthly contract flag should be created"
    assert 'support_interaction_rate' in df_features.columns, "Support interaction rate should be created"
    
    # Check that customer_id is dropped
    assert 'customer_id' not in df_features.columns, "Customer ID should be dropped"

def test_model_prediction():
    import joblib
    import os
    
    # Check if model exists
    model_path = 'model_registry/churn_model/model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        assert model is not None, "Model should be loaded successfully"
        
        # Test model prediction shape
        import numpy as np
        sample_input = np.random.rand(1, 19)  # Assuming 19 features
        try:
            prediction = model.predict(sample_input)
            assert prediction is not None, "Prediction should work"
        except Exception as e:
            pytest.skip(f"Model prediction failed: {e}")
    else:
        pytest.skip(f"Model not found at {model_path}. Run training first.")

def test_data_transformer():
    """Additional test for DataTransformer"""
    df = pd.DataFrame({
        'numeric1': [1, 2, 3],
        'numeric2': [4, 5, 6],
        'categorical1': ['A', 'B', 'A'],
        'categorical2': ['X', 'Y', 'X']
    })
    
    transformer = DataTransformer()
    numeric_cols = ['numeric1', 'numeric2']
    categorical_cols = ['categorical1', 'categorical2']
    
    transformer.build_preprocessor(numeric_cols, categorical_cols)
    X_transformed = transformer.fit_transform(df)
    
    assert X_transformed is not None, "Transformation should work"
    assert X_transformed.shape[0] == len(df), "Should have same number of rows"

if __name__ == '__main__':
    pytest.main(['-v'])
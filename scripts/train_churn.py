#!/usr/bin/env python
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import yaml

from src.data.preprocessing import DataLoader, DataCleaner, DataTransformer
from src.features.feature_engineering import ChurnFeatureBuilder

def train_churn_model(algorithm='RandomForestClassifier', **kwargs):
    # Load data
    df_raw = DataLoader.load_raw_data()
    df_clean = DataCleaner.clean(df_raw)
    
    # Feature engineering
    df_features = ChurnFeatureBuilder.build_features(df_clean)
    
    # Separate features and target
    X = df_features.drop(columns=['churn'])
    y = df_features['churn']
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Preprocessing
    transformer = DataTransformer()
    transformer.build_preprocessor(numeric_cols, categorical_cols)
    X_transformed = transformer.fit_transform(X)
    
    # Get feature names after transformation
    feature_names = (numeric_cols + 
                    list(transformer.preprocessor.named_transformers_['cat']
                         .named_steps['onehot'].get_feature_names_out(categorical_cols)))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Select model
    if algorithm == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            class_weight='balanced',
            random_state=42
        )
    elif algorithm == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            random_state=42
        )
    else:
        model = LogisticRegression(class_weight='balanced', random_state=42)
    
    # Train
    model.fit(X_train, y_train)
    
    # Calibrate
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'auc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'accuracy': (y_pred == y_test).mean(),
        'feature_names': feature_names,
        'feature_importances': dict(zip(feature_names, model.feature_importances_.tolist()))
    }
    
    # Save artifacts
    joblib.dump(calibrated_model, 'model_registry/churn_model/model.pkl')
    joblib.dump(transformer.preprocessor, 'model_registry/churn_model/scaler.pkl')
    
    with open('model_registry/churn_model/features.json', 'w') as f:
        json.dump(feature_names, f)
    
    with open('model_registry/churn_model/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open('model_registry/churn_model/feature_importances.json', 'w') as f:
        json.dump(metrics['feature_importances'], f, indent=2)
    
    print(f"✅ Model trained: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
    return calibrated_model, transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='RandomForestClassifier')
    parser.add_argument('--n_estimators', type=int, default=100)
    args = parser.parse_args()
    
    train_churn_model(algorithm=args.algorithm, n_estimators=args.n_estimators)
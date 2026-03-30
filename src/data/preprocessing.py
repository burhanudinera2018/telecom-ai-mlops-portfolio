import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    @staticmethod
    def load_raw_data(path='data/raw/customer_data.csv'):
        return pd.read_csv(path)
    
    @staticmethod
    def load_processed_data(path='data/processed/feature_set.csv'):
        return pd.read_csv(path)


class DataCleaner:
    @staticmethod
    def clean(df, remove_outliers=True):
        """
        Clean the dataframe with robust error handling
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe to clean
        remove_outliers : bool
            Whether to remove outliers from monthly_charges
        
        Returns:
        --------
        pandas.DataFrame : Cleaned dataframe
        """
        df = df.copy()
        
        # Log original shape
        original_shape = df.shape
        logger.info(f"Cleaning dataframe with shape: {original_shape}")
        
        # Handle missing values for all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {df[col].isnull().sum()} missing values in {col} with median: {median_val}")
        
        # Remove outliers only if column exists and flag is True
        if remove_outliers and 'monthly_charges' in df.columns:
            q99 = df['monthly_charges'].quantile(0.99)
            original_count = len(df)
            df = df[df['monthly_charges'] < q99]
            removed_count = original_count - len(df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} outlier rows with monthly_charges >= {q99}")
        
        logger.info(f"Cleaning complete. New shape: {df.shape}")
        return df
    
    @staticmethod
    def validate_required_columns(df, required_columns):
        """
        Validate that required columns exist in dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe to validate
        required_columns : list
            List of required column names
        
        Returns:
        --------
        bool : True if all columns exist
        """
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False
        return True

class DataTransformer:
    def __init__(self):
        self.preprocessor = None
    
    def build_preprocessor(self, numeric_cols, categorical_cols):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        return self.preprocessor
    
    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X)
    
    def transform(self, X):
        return self.preprocessor.transform(X)
    
    def save(self, path='model_registry/churn_model/preprocessor.pkl'):
        joblib.dump(self.preprocessor, path)
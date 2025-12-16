import pandas as pd
import numpy as np
import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# If you want WOE encoding
from category_encoders.woe import WOEEncoder  

# -----------------------------
# Setup basic logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# Data Loading
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load raw transaction data."""
    try:
        df = pd.read_csv(path)
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        logging.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# -----------------------------
# Aggregate Features
# -----------------------------
def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['CustomerId', 'TransactionId', 'Amount']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    agg = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_amount=('Amount', 'std')
    ).reset_index()

    agg['std_amount'] = agg['std_amount'].fillna(0)
    logging.info("Aggregate features created.")
    return agg

# -----------------------------
# Time-Based Features
# -----------------------------
def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'TransactionStartTime' not in df.columns:
        raise ValueError("TransactionStartTime column missing")
    df = df.copy()
    df['tx_hour'] = df['TransactionStartTime'].dt.hour
    df['tx_day'] = df['TransactionStartTime'].dt.day
    df['tx_month'] = df['TransactionStartTime'].dt.month
    df['tx_year'] = df['TransactionStartTime'].dt.year
    logging.info("Time-based features extracted.")
    return df

# -----------------------------
# Merge Aggregates
# -----------------------------
def merge_features(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    df_merged = df.merge(agg, on='CustomerId', how='left')
    logging.info("Aggregates merged back to main dataframe.")
    return df_merged

# -----------------------------
# Feature Groups
# -----------------------------
NUMERIC_FEATURES = [
    'Value', 'Amount',
    'total_amount', 'avg_amount',
    'transaction_count', 'std_amount',
    'tx_hour', 'tx_day', 'tx_month', 'tx_year'
]

CATEGORICAL_FEATURES = [
    'ChannelId', 'ProductCategory',
    'ProviderId', 'PricingStrategy'
]

# -----------------------------
# Pipelines
# -----------------------------
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

def build_preprocessing_pipeline():
    """Build full preprocessing pipeline for numeric + categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )
    logging.info("Preprocessing pipeline created.")
    return preprocessor

# -----------------------------
# Full Feature Preparation
# -----------------------------
def prepare_features(df: pd.DataFrame, target_column: str = None):
    """
    Prepares features for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw data.
    target_column : str, optional
        If provided, the target column will be used for WOE encoding or supervised transformers.

    Returns
    -------
    X : np.ndarray
        Preprocessed feature array.
    preprocessor : ColumnTransformer
        The fitted preprocessing pipeline.
    feature_names : list
        Names of transformed features.
    """
    df = extract_time_features(df)
    agg = create_aggregate_features(df)
    df = merge_features(df, agg)

    preprocessor = build_preprocessing_pipeline()

    if target_column and target_column in df.columns:
        y = df[target_column]
        X = preprocessor.fit_transform(df, y)  # For supervised transformers like WOE
    else:
        X = preprocessor.fit_transform(df)
        y = None

    feature_names = preprocessor.get_feature_names_out()
    logging.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
    return X, preprocessor, feature_names

# -----------------------------
# Save Processed Features
# -----------------------------
def save_processed_data(X, path: str):
    np.save(path, X)
    logging.info(f"Processed features saved to {path}")

# -----------------------------
# Main block to run directly
# -----------------------------
if __name__ == "__main__":
    raw_data_path = "data/raw/data.csv"
    processed_path = "data/processed/features.npy"
    # processed_path = "data/processed/model_input.csv"

    df = load_data(raw_data_path)
    X, preprocessor, feature_names = prepare_features(df)
    save_processed_data(X, processed_path)

    print(f"Processed features saved successfully at '{processed_path}'")
    print(f"Feature array shape: {X.shape}")
    print(f"First 10 features: {feature_names[:10]}")

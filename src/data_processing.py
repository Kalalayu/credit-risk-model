import os
import logging
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders.woe import WOEEncoder

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------------
# File paths
# -----------------------------
RAW_DATA_PATH = "data/raw/data.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "features.npy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
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
# Datetime Features
# -----------------------------
def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['tx_hour'] = df['TransactionStartTime'].dt.hour
    df['tx_day'] = df['TransactionStartTime'].dt.day
    df['tx_month'] = df['TransactionStartTime'].dt.month
    df['tx_year'] = df['TransactionStartTime'].dt.year
    logging.info("Datetime features extracted.")
    return df

# -----------------------------
# Merge Aggregates
# -----------------------------
def merge_features(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    df_merged = df.merge(agg, on='CustomerId', how='left')
    logging.info("Aggregates merged into main dataframe.")
    return df_merged

# -----------------------------
# Feature Groups
# -----------------------------
NUMERIC_FEATURES = [
    'Amount', 'total_amount', 'avg_amount',
    'transaction_count', 'std_amount',
    'tx_hour', 'tx_day', 'tx_month', 'tx_year'
]

CATEGORICAL_FEATURES = [
    'ChannelId', 'ProductCategory', 'ProviderId', 'PricingStrategy'
]

# -----------------------------
# Pipelines
# -----------------------------
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('woe', WOEEncoder())  # WoE encoding for categorical variables
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, NUMERIC_FEATURES),
    ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
])

# -----------------------------
# Full Feature Preparation
# -----------------------------
def prepare_features(df: pd.DataFrame, target_column: str = None):
    df = extract_time_features(df)
    agg = create_aggregate_features(df)
    df = merge_features(df, agg)

    if target_column and target_column in df.columns:
        y = df[target_column]
        X = preprocessor.fit_transform(df, y)  # pass target to WOEEncoder
    else:
        raise ValueError("WOEEncoder requires a target column. Please provide 'target_column'.")

    feature_names = preprocessor.get_feature_names_out()
    logging.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
    return X, feature_names, preprocessor

# -----------------------------
# Save Processed Features
# -----------------------------
def save_processed_data(X: np.ndarray, path: str):
    np.save(path, X)
    logging.info(f"Processed features saved to {path}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # 1️⃣ Load raw data
    df = load_data(RAW_DATA_PATH)

    # 2️⃣ Create a binary target column if it doesn't exist
    if 'is_high_risk' not in df.columns:
        logging.info("Target column 'is_high_risk' not found. Creating it using RFM logic.")

        # Compute RFM per customer
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
            'TransactionId': 'count',                                          # Frequency
            'Amount': 'sum'                                                     # Monetary
        }).reset_index()

        rfm.rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary'
        }, inplace=True)

        # Assign high-risk customers (e.g., top 20% recency)
        threshold = rfm['Recency'].quantile(0.8)
        rfm['is_high_risk'] = (rfm['Recency'] >= threshold).astype(int)

        # Merge back to main DataFrame
        df = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
        logging.info("Target column 'is_high_risk' created successfully.")

    # 3️⃣ Prepare features (pass target to WOEEncoder)
    target_column = 'is_high_risk'
    X, feature_names, pipeline = prepare_features(df, target_column=target_column)

    # 4️⃣ Save processed features
    save_processed_data(X, OUTPUT_PATH)

    print(f"Processed features saved at '{OUTPUT_PATH}'")
    print(f"Feature array shape: {X.shape}")
    print(f"First 10 features: {feature_names[:10]}")

# tests/test_data_processing.py
import pytest
import numpy as np
import pandas as pd
from src.data_processing import (
    create_aggregate_features,
    extract_time_features,
    merge_features,
    prepare_features,
    save_processed_data
)

# -----------------------------
# Sample Data
# -----------------------------
@pytest.fixture
def sample_data():
    data = {
        'TransactionId': [1, 2, 3],
        'CustomerId': [101, 101, 102],
        'Amount': [100, 200, 150],
        'Value': [100, 200, 150],
        'TransactionStartTime': pd.to_datetime([
            '2025-12-01 10:00:00',
            '2025-12-01 11:00:00',
            '2025-12-01 15:00:00'
        ]),
        'ChannelId': ['WEB', 'APP', 'IOS'],
        'ProductCategory': ['financial_services', 'financial_services', 'ecommerce'],
        'ProviderId': ['P1', 'P2', 'P3'],
        'PricingStrategy': ['A', 'B', 'A']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_with_target(sample_data):
    df = sample_data.copy()
    df['is_high_risk'] = [0, 1, 0]  # Binary target
    return df

# -----------------------------
# Test Aggregate Features
# -----------------------------
def test_create_aggregate_features(sample_data):
    agg = create_aggregate_features(sample_data)
    assert 'total_amount' in agg.columns
    assert 'avg_amount' in agg.columns
    assert 'transaction_count' in agg.columns
    assert 'std_amount' in agg.columns
    assert agg.shape[0] == 2  # 2 unique customers

# -----------------------------
# Test Time Features
# -----------------------------
def test_extract_time_features(sample_data):
    df = extract_time_features(sample_data)
    assert 'tx_hour' in df.columns
    assert 'tx_day' in df.columns
    assert 'tx_month' in df.columns
    assert 'tx_year' in df.columns

# -----------------------------
# Test Merge Features
# -----------------------------
def test_merge_features(sample_data):
    agg = create_aggregate_features(sample_data)
    df_merged = merge_features(sample_data, agg)
    assert 'total_amount' in df_merged.columns
    assert df_merged.shape[0] == sample_data.shape[0]

# -----------------------------
# Test Prepare Features without target
# -----------------------------
def test_prepare_features_no_target(sample_data):
    X, preprocessor, feature_names = prepare_features(sample_data)
    assert X.shape[0] == sample_data.shape[0]
    assert len(feature_names) == X.shape[1]

# -----------------------------
# Test Prepare Features with target
# -----------------------------
def test_prepare_features_with_target(sample_data_with_target):
    df = sample_data_with_target.copy()
    X, preprocessor, feature_names = prepare_features(df, target_column='is_high_risk')
    assert X.shape[0] == df.shape[0]
    assert len(feature_names) == X.shape[1]

# -----------------------------
# Test Saving Processed Data
# -----------------------------
def test_save_processed_data(tmp_path, sample_data):
    X, _, _ = prepare_features(sample_data)
    save_path = tmp_path / "features.npy"
    save_processed_data(X, save_path)
    loaded = np.load(save_path)
    assert np.array_equal(X, loaded)

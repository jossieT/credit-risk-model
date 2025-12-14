import pandas as pd
import pytest
from src.data_processing import map_risk_target, load_data
import os

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")

def test_map_risk_target_rfm_logic():
    # Create sample data
    data = {
        'CustomerId': ['C1', 'C2', 'C3', 'C4'],
        'TransactionStartTime': [
            '2023-01-01T00:00:00Z', # C1: Old (Recency High), Low Value
            '2023-01-10T00:00:00Z', # C2: Recent, High Value
            '2023-01-10T00:00:00Z', # C3: Recent, High Value
            '2023-01-05T00:00:00Z'  # C4: Middle
        ],
        'Amount': [10.0, 5000.0, 5000.0, 100.0],
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'FraudResult': [0, 0, 1, 0] # C3 has fraud -> Automatic Risk
    }
    df = pd.DataFrame(data)
    
    # Expected behavior:
    # C1: Dormant (oldest date) + Low Monetary -> Likely Risk=1
    # C2: Active, High Monetary -> Risk=0
    # C3: Has Fraud -> Risk=1
    # C4: Middle -> Likely Risk=0 (depending on quantile thresholds)
    
    # Note: With only 4 points, quantiles can be tricky.
    # 0.75 Recency Quantile of [9 days, 0, 0, 5] -> 5 days approx
    # C1 Recency=9 > 5 -> Risk
    
    result_df = map_risk_target(df)
    
    assert 'RiskTarget' in result_df.columns
    
    # Check C1 (High Recency = Risk)
    assert result_df.loc[result_df['CustomerId'] == 'C1', 'RiskTarget'].values[0] == 1
    
    # Check C2 (Good behavior)
    assert result_df.loc[result_df['CustomerId'] == 'C2', 'RiskTarget'].values[0] == 0
    
    # Check C3 (Fraud)
    assert result_df.loc[result_df['CustomerId'] == 'C3', 'RiskTarget'].values[0] == 1

def test_map_risk_target_empty_df():
    df = pd.DataFrame()
    result = map_risk_target(df)
    assert result.empty
    assert 'RiskTarget' in result.columns

import pytest
import pandas as pd
import numpy as np
import os
from src.data_processing import DateTimeFeatures, CustomerAggregator, compute_rfm

def test_datetime_features():
    df = pd.DataFrame({
        'TransactionStartTime': ['2018-11-15T02:19:08Z', '2018-11-16T10:00:00Z']
    })
    dtf = DateTimeFeatures()
    df_transformed = dtf.transform(df)
    
    assert 'transaction_hour' in df_transformed.columns
    assert 'transaction_day' in df_transformed.columns
    assert 'transaction_month' in df_transformed.columns
    assert 'transaction_year' in df_transformed.columns
    assert df_transformed['transaction_hour'].iloc[0] == 2
    assert df_transformed['transaction_day'].iloc[1] == 16

def test_rfm_calculation():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionStartTime': ['2018-11-15', '2018-11-16', '2018-11-10'],
        'TransactionId': ['T1', 'T2', 'T3'],
        'Amount': [100, 200, 50]
    })
    rfm = compute_rfm(df)
    
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    assert rfm.loc['C1', 'Frequency'] == 2
    assert rfm.loc['C1', 'Monetary'] == 300
    assert rfm.loc['C2', 'Frequency'] == 1

def test_customer_aggregator():
    df = pd.DataFrame({
        'CustomerId': ['C1', 'C1'],
        'Amount': [100, 200],
        'TransactionStartTime': ['2018-11-15', '2018-11-16'],
        'ProductCategory': ['fin', 'fin'],
        'ChannelId': ['2', '2'],
        'ProviderId': ['4', '4']
    })
    # Pre-extract temporal features for aggregator
    dtf = DateTimeFeatures()
    df = dtf.transform(df)
    
    aggregator = CustomerAggregator()
    df_agg = aggregator.transform(df)
    
    assert 'total_transaction_amount' in df_agg.columns
    assert df_agg.loc[0, 'total_transaction_amount'] == 300
    assert df_agg.loc[0, 'transaction_count'] == 2
    assert df_agg.loc[0, 'ProductCategory'] == 'fin'

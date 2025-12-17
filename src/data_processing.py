import pandas as pd
import numpy as np
import os
import joblib
import logging
from typing import List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from xverse.transformer import WOE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from TransactionStartTime.
    """
    def __init__(self, date_col: str = 'TransactionStartTime'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X['transaction_hour'] = X[self.date_col].dt.hour
        X['transaction_day'] = X[self.date_col].dt.day
        X['transaction_month'] = X[self.date_col].dt.month
        X['transaction_year'] = X[self.date_col].dt.year
        return X

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates transactions per CustomerId.
    """
    def __init__(self, group_col: str = 'CustomerId'):
        self.group_col = group_col
        self.agg_features = {
            'Amount': ['sum', 'mean', 'count', 'std'],
            'transaction_hour': ['mean', 'std'],
            'transaction_day': ['mean'],
            'transaction_month': ['mean']
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Aggregate numeric features
        agg_df = X.groupby(self.group_col).agg(self.agg_features)
        
        # Flatten columns
        agg_df.columns = [
            'total_transaction_amount', 'avg_transaction_amount', 
            'transaction_count', 'std_transaction_amount',
            'avg_transaction_hour', 'std_transaction_hour',
            'avg_transaction_day', 'avg_transaction_month'
        ]
        
        # Fill NaN for std if count is 1
        agg_df = agg_df.fillna(0)
        
        # Aggregate categorical features (Mode)
        cat_cols = ['ProductCategory', 'ChannelId', 'ProviderId']
        for col in cat_cols:
            if col in X.columns:
                mode_df = X.groupby(self.group_col)[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
                agg_df[col] = mode_df
        
        return agg_df.reset_index()

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM metrics for each customer.
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    return rfm

def create_is_high_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define high risk using KMeans clustering on RFM.
    """
    rfm = compute_rfm(df)
    
    # Scale RFM before clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify the high risk cluster (Lowest Frequency, Lowest Monetary, Highest Recency)
    # We find the cluster with the minimum (Frequency + Monetary - Recency) or similar logic.
    # Centroids analysis:
    centroids = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Define "Least Engaged" cluster: 
    # High Recency is bad, Low Frequency is bad, Low Monetary is bad.
    # score = (scaled_recency) - (scaled_frequency) - (scaled_monetary)
    # The cluster with the HIGHEST score is the most risky.
    
    # Let's just use the centroids directly to find the one that fits "least engaged"
    # least engaged: max recency, min frequency, min monetary
    
    # Normalizing centroids to compare them
    norm_centroids = (centroids - centroids.mean()) / centroids.std()
    risk_score = norm_centroids['Recency'] - norm_centroids['Frequency'] - norm_centroids['Monetary']
    high_risk_cluster = risk_score.idxmax()
    
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    logging.info(f"High risk cluster identified as Cluster {high_risk_cluster}")
    logging.info(f"Risk distribution:\n{rfm['is_high_risk'].value_counts()}")
    
    return rfm[['is_high_risk']]

def get_pipeline(numerical_cols: List[str], categorical_cols: List[str]):
    """
    Creates the sklearn Pipeline for feature engineering.
    """
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_cols),
            ('cat', cat_transformer, categorical_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

class SimpleWoETransformer(BaseEstimator, TransformerMixin):
    """
    Custom Weight of Evidence (WoE) transformer.
    """
    def __init__(self, columns: List[str] = None):
        self.columns = columns
        self.woe_maps = {}
        self.iv_dict = {}

    def fit(self, X, y):
        X = X.copy()
        if self.columns is None:
            self.columns = X.columns
        
        for col in self.columns:
            # Simple binning for numerical
            if X[col].nunique() > 10:
                X_binned = pd.qcut(X[col], q=10, duplicates='drop').astype(str)
            else:
                X_binned = X[col].astype(str)
            
            df = pd.DataFrame({'feature': X_binned, 'target': y})
            counts = df.groupby('feature')['target'].agg(['count', 'sum'])
            counts['none_event'] = counts['count'] - counts['sum']
            counts['event_rate'] = counts['sum'] / counts['sum'].sum()
            counts['none_event_rate'] = counts['none_event'] / counts['none_event'].sum()
            
            # Avoid division by zero
            counts['event_rate'] = counts['event_rate'].replace(0, 0.0001)
            counts['none_event_rate'] = counts['none_event_rate'].replace(0, 0.0001)
            
            counts['woe'] = np.log(counts['none_event_rate'] / counts['event_rate'])
            counts['iv'] = (counts['none_event_rate'] - counts['event_rate']) * counts['woe']
            
            self.woe_maps[col] = counts['woe'].to_dict()
            self.iv_dict[col] = counts['iv'].sum()
            
        return self

    def transform(self, X):
        X = X.copy()
        for col, woe_map in self.woe_maps.items():
            if X[col].nunique() > 10:
                X_binned = pd.qcut(X[col], q=10, duplicates='drop').astype(str)
            else:
                X_binned = X[col].astype(str)
            X[col] = X_binned.map(woe_map).fillna(0)
        return X

def process_data(raw_data_path: str, processed_data_path: str):
    """
    Main function to process data and save it.
    """
    logging.info(f"Reading raw data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Step 1: Extract Temporal Features
    dt_features = DateTimeFeatures()
    df = dt_features.transform(df)
    
    # Step 2: Create Target (High Risk) from RFM
    risk_df = create_is_high_risk(df)
    
    # Step 3: Aggregate to Customer Level
    aggregator = CustomerAggregator()
    df_agg = aggregator.transform(df)
    
    # Step 4: Merge Target
    df_agg = df_agg.merge(risk_df, on='CustomerId', how='left')
    
    # Step 5: Define features for Pipeline
    numerical_features = [
        'total_transaction_amount', 'avg_transaction_amount', 
        'transaction_count', 'std_transaction_amount',
        'avg_transaction_hour', 'std_transaction_hour',
        'avg_transaction_day', 'avg_transaction_month'
    ]
    categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId']
    
    # Step 6: Pipeline for Encoding and Scaling
    pipeline = get_pipeline(numerical_features, categorical_features)
    
    X = df_agg[numerical_features + categorical_features]
    y = df_agg['is_high_risk']
    
    X_processed_array = pipeline.fit_transform(X)
    
    # Get feature names from OneHotEncoder
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_features = numerical_features + cat_features_encoded
    
    X_processed = pd.DataFrame(X_processed_array, columns=all_features)
    
    # Save objects for API
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(model_dir, "pipeline.pkl"))
    
    # Step 7: WoE and IV
    logging.info("Applying WoE transformation")
    woe = SimpleWoETransformer(columns=numerical_features)
    woe.fit(X_processed, y)
    X_woe = woe.transform(X_processed)
    
    joblib.dump(woe, os.path.join(model_dir, "woe.pkl"))
    
    # Log IV scores
    iv_dict = woe.iv_dict
    logging.info(f"Information Value (IV) scores: {iv_dict}")
    
    # Drop features with IV below 0.02
    threshold = 0.02
    features_to_drop = [feat for feat, iv in iv_dict.items() if iv < threshold]
    logging.info(f"Dropping features with IV < {threshold}: {features_to_drop}")
    X_woe = X_woe.drop(columns=features_to_drop)
    
    # Final data
    final_df = pd.concat([X_woe, y.reset_index(drop=True)], axis=1)
    
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    final_df.to_csv(processed_data_path, index=False)
    logging.info(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    process_data('data/raw/data.csv', 'data/processed/credit_model_features.csv')

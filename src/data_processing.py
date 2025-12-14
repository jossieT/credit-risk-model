import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or missing critical columns.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at {path}")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError("The loaded dataframe is empty.")

        # Basic validation (checking for some expected columns based on EDA)
        expected_cols = ['CustomerId', 'TransactionStartTime', 'Amount', 'TransactionId', 'FraudResult']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
             logging.warning(f"Missing expected columns: {missing_cols}. Proceeding, but risk mapping might fail if these are required.")

        logging.info(f"Data loaded successfully from {path}. Shape: {df.shape}")
        return df

    except FileNotFoundError as e:
        logging.error(e)
        raise
    except pd.errors.EmptyDataError:
        logging.error("File is empty.")
        raise ValueError("File is empty.")
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading data: {e}")
        raise

def map_risk_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps a proxy risk target (default/non-default) based on RFM analysis.
    
    Proxy Definition:
    - High Risk (1): Bottom quartile of Frequency (Activity) AND Monetary (Value), 
                     OR Top quartile of Recency (Inactivity),
                     OR FraudResult == 1.
    - Low Risk (0): Otherwise.

    Args:
        df (pd.DataFrame): Input dataframe with transaction data.

    Returns:
        pd.DataFrame: Dataframe with an additional 'RiskTarget' column (1 = High Risk, 0 = Low Risk).
    """
    if df.empty:
        logging.warning("Input dataframe is empty. Returning empty dataframe with RiskTarget column.")
        df['RiskTarget'] = []
        return df

    required_cols = ['CustomerId', 'TransactionStartTime', 'Amount', 'TransactionId', 'FraudResult']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input dataframe missing required columns for RFM: {required_cols}")

    try:
        # Preprocessing
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # RFM Calculation per Customer
        # Recency: Days since last transaction (relative to the latest date in dataset)
        # Frequency: Count of transactions
        # Monetary: Sum of Amount (Total Spend/Volume)
        
        latest_date = df['TransactionStartTime'].max()
        
        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (latest_date - x.max()).days,
            'TransactionId': 'count',
            'Amount': 'sum',
            'FraudResult': 'max' # If any fraud, flag as potential risk (or filter out depending on business logic, here we treat as risky)
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Amount': 'Monetary',
            'FraudResult': 'HasFraud'
        })
        
        # Quantiles for classification
        # High Recency = Risk (Dormant)
        # Low Frequency = Risk (Low Engagement)
        # Low Monetary = Risk (Low Value/Low Capacity)
        
        # Using 0.75 for Recency (Top 25% are most dormant)
        # Using 0.25 for Frequency/Monetary (Bottom 25% are least active)
        
        r_thresh = rfm['Recency'].quantile(0.75)
        f_thresh = rfm['Frequency'].quantile(0.25)
        m_thresh = rfm['Monetary'].quantile(0.25)
        
        def define_risk(row):
            # Known Fraud is automatic High Risk
            if row['HasFraud'] == 1:
                return 1
            
            # High Risk conditions
            is_dormant = row['Recency'] > r_thresh
            is_low_value = (row['Frequency'] < f_thresh) and (row['Monetary'] < m_thresh)
            
            if is_dormant or is_low_value:
                return 1
            else:
                return 0

        rfm['RiskTarget'] = rfm.apply(define_risk, axis=1)
        
        # Merge back to original DF (assigning the customer's risk profile to all their transactions)
        # Or usually, we return the Customer Level DF for modeling. 
        # The prompt implies "map_risk_target" might act on the main DF. 
        # Let's drop the RFM metrics from merge to keep it clean, or keep them if useful features.
        # For now, we'll map the target back to the main DF.
        
        df = df.merge(rfm[['RiskTarget']], on='CustomerId', how='left')
        
        logging.info("Risk target mapping completed.")
        logging.info(f"Risk Distribution:\n{df['RiskTarget'].value_counts(normalize=True)}")
        
        return df

    except Exception as e:
        logging.error(f"Error during risk target mapping: {e}")
        raise

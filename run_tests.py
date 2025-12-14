import unittest
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.data_processing import map_risk_target, load_data

class TestDataProcessing(unittest.TestCase):
    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_data("non_existent_file.csv")

    def test_map_risk_target_rfm_logic(self):
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
            'FraudResult': [0, 0, 1, 0] # C3 has fraud
        }
        df = pd.DataFrame(data)
        
        result_df = map_risk_target(df)
        
        self.assertIn('RiskTarget', result_df.columns)
        
        # C1: High Recency should be 1
        self.assertEqual(result_df.loc[result_df['CustomerId'] == 'C1', 'RiskTarget'].values[0], 1, "C1 should be High Risk (Recency)")
        
        # C2: Active should be 0
        self.assertEqual(result_df.loc[result_df['CustomerId'] == 'C2', 'RiskTarget'].values[0], 0, "C2 should be Low Risk")
        
        # C3: Fraud should be 1
        self.assertEqual(result_df.loc[result_df['CustomerId'] == 'C3', 'RiskTarget'].values[0], 1, "C3 should be High Risk (Fraud)")

    def test_map_risk_target_empty_df(self):
        df = pd.DataFrame()
        result = map_risk_target(df)
        self.assertTrue(result.empty)
        self.assertIn('RiskTarget', result.columns)

if __name__ == '__main__':
    unittest.main()

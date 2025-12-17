import pandas as pd
import numpy as np
import os
import joblib
import logging
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads processed data and splits into features and target.
    """
    df = pd.read_csv(path)
    X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']
    return X, y

def train_and_track_model(
    model_name: str, 
    model: Any, 
    params: Dict[str, Any], 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
):
    """
    Trains a model with GridSearchCV, logs to MLflow, and returns the best model.
    """
    with mlflow.start_run(run_name=model_name):
        logging.info(f"Starting MLflow run for {model_name}")
        
        grid_search = GridSearchCV(
            model, params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        
        # Log confusion matrix as a text file artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_path = f"outputs/confusion_matrix_{model_name}.txt"
        os.makedirs("outputs", exist_ok=True)
        with open(cm_path, "w") as f:
            f.write(str(cm))
        mlflow.log_artifact(cm_path)
        
        # Log model
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(best_model, model_name, signature=signature)
        
        logging.info(f"Model {model_name} trained. ROC-AUC: {metrics['roc_auc']:.4f}")
        return best_model, metrics['roc_auc']

def main():
    # Set experiment
    mlflow.set_experiment("Credit_Risk_Model_Training")
    
    # Load data
    processed_path = 'data/processed/credit_model_features.csv'
    if not os.path.exists(processed_path):
        logging.error(f"Processed data not found at {processed_path}. Run data_processing.py first.")
        return

    X, y = load_processed_data(processed_path)
    
    # Train/Test Split (80/20, Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model 1: Logistic Regression
    lr_params = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    }
    lr_model, lr_auc = train_and_track_model(
        "Logistic_Regression", LogisticRegression(random_state=42), 
        lr_params, X_train, X_test, y_train, y_test
    )
    
    # Model 2: Random Forest
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf_model, rf_auc = train_and_track_model(
        "Random_Forest", RandomForestClassifier(random_state=42), 
        rf_params, X_train, X_test, y_train, y_test
    )
    
    # Select and Register best model
    best_model = lr_model if lr_auc > rf_auc else rf_model
    best_name = "Logistic_Regression" if lr_auc > rf_auc else "Random_Forest"
    
    logging.info(f"Best model: {best_name} with AUC: {max(lr_auc, rf_auc):.4f}")
    
    # Save best model locally for API
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))
    logging.info(f"Best model saved to {os.path.join(model_dir, 'best_model.pkl')}")

if __name__ == "__main__":
    main()

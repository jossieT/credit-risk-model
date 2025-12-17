import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import mlflow.sklearn

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="API for predicting Probability of Default (PD) for customers.",
    version="1.0.0"
)

# Configuration for model loading
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
TRANSFORMER_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
WOE_PATH = os.path.join(MODEL_DIR, "woe.pkl")

# Global variables for model components
model = None
pipeline = None
woe_transformer = None

@app.on_event("startup")
def load_models():
    global model, pipeline, woe_transformer
    try:
        # Priority 1: Load from local files (set by train.py)
        if os.path.exists(BEST_MODEL_PATH):
            model = joblib.load(BEST_MODEL_PATH)
            pipeline = joblib.load(TRANSFORMER_PATH)
            woe_transformer = joblib.load(WOE_PATH)
            print("Model and transformers loaded from local storage.")
        else:
            # Priority 2: Try MLflow (Placeholder for production environment)
            # model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")
            print("Warning: Model files not found. Predict endpoint will fail.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Risk Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None or pipeline is None or woe_transformer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic model to DataFrame for transformation
        input_df = pd.DataFrame([features.dict()])
        
        # 1. Transform featuring Pipeline (Impute, Scale, OHE)
        # Note: We need the column names for WOE to work correctly
        processed_array = pipeline.transform(input_df)
        
        # Get feature names for DataFrame
        cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        numerical_features = [
            'total_transaction_amount', 'avg_transaction_amount', 
            'transaction_count', 'std_transaction_amount',
            'avg_transaction_hour', 'std_transaction_hour',
            'avg_transaction_day', 'avg_transaction_month'
        ]
        categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId']
        cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features).tolist()
        all_features = numerical_features + cat_features_encoded
        
        processed_df = pd.DataFrame(processed_array, columns=all_features)
        
        # 2. Apply WoE
        woe_df = woe_transformer.transform(processed_df)
        
        # 3. Ensure columns match training (handle dropped IV features)
        # The model expects specific features.
        model_features = model.feature_names_in_
        woe_df = woe_df[model_features]
        
        # 4. Predict
        prob = model.predict_proba(woe_df)[0][1]
        prediction = int(model.predict(woe_df)[0])
        
        return PredictionResponse(
            is_high_risk=prediction,
            probability_of_default=float(prob)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# src/api/main.py

from fastapi import FastAPI
from .pydantic_models import CustomerData, PredictionResponse  # relative import
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Credit Risk Prediction API")

# Load your best model from MLflow run
MODEL_URI = "runs:/923ec0127ed94873baf68f6e93034706/model"  # replace with your actual run ID
model = mlflow.sklearn.load_model(MODEL_URI)

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([data.dict()])
    # Get probability of positive class (risk)
    risk_prob = model.predict_proba(df)[:, 1][0]
    return {"risk_probability": risk_prob}

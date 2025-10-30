import pandas as pd
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from typing import List
from pathlib import Path
import main
import joblib


router = APIRouter(
    prefix="/forms",
   tags=["Forms"]
)

# --- model caching and robust path resolution ---
_model = None

def _load_model():
    global _model
    if _model is None:
        # Resolve project root reliably from this file's location
        project_root = Path(__file__).resolve().parents[2]  # Cognicare-Backend
        model_path = project_root / "ml_models" / "asd_rf_model.pkl"
        if not model_path.exists():
            raise HTTPException(status_code=500, detail=f"Model file not found at: {model_path}")
        _model = joblib.load(model_path)
    return _model

async def predict_autism(input_data):
    """Predict autism likelihood using the saved model.
    
    Args:
        input_data (dict): Dictionary containing:
            - A1 to A10: Questionnaire responses (0 = No, 1 = Yes)
            - Age_Mons: Age in months
            - Sex: 'm' or 'f'
            - Ethnicity: ethnicity value
            - Jaundice: 'yes' or 'no'
            - Family_mem_with_ASD: 'yes' or 'no'
    
    Returns:
        tuple: (prediction (0 or 1), probability of ASD)
    """
    # Load the saved model (cached)
    saved_model = _load_model()

    # Try to get feature names from the model
    if not hasattr(saved_model, 'feature_names_in_'):
        raise HTTPException(status_code=500, detail="Model is missing feature names information.")
    
    columns = saved_model.feature_names_in_
    
    # Create a DataFrame with one row (all zeros)
    test_data = pd.DataFrame([{col: 0 for col in columns}])
    
    # Update with provided values
    for feature, value in input_data.items():
        if feature in test_data.columns:
            test_data[feature] = value
        else:
            # Handle categorical features
            col_name = f"{feature}_{value}"
            if col_name in test_data.columns:
                # Reset all related columns
                related_cols = [c for c in test_data.columns if c.startswith(f"{feature}_")]
                for c in related_cols:
                    test_data[c] = 0
                test_data[col_name] = 1
    
    # Make prediction
    pred = saved_model.predict(test_data)[0]
    prob = saved_model.predict_proba(test_data)[0][1] if hasattr(saved_model, "predict_proba") else None
    
    return int(pred), prob
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from typing import List
from tensorflow.keras import utils
from .auth import get_current_user

router = APIRouter(
    prefix="/predict",
    dependencies=[Depends(get_current_user)],
)
# Define the input data model based on your model's features

class PredictionInput(BaseModel):
    # This type definition matches the shape (num_frames, 224, 224, 3)
    frames: List[List[List[List[float]]]]

# Load model once at module level
_model = None

def _load_video_model():
    """Load and cache the video prediction model"""
    global _model
    if _model is None:
        try:
            _model = tf.keras.models.load_model("ml_models/best_model_fine_tuned.h5")
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Video model not loaded. Please check server configuration."
            )
    return _model

def make_prediction(input_data: np.ndarray) -> np.ndarray:
    """
    Make a prediction using the video model.
    
    Args:
        input_data: numpy array of shape (num_frames, 224, 224, 3)
    
    Returns:
        numpy array of predictions
    
    Raises:
        HTTPException: if model not loaded or prediction fails
    """
    model = _load_video_model()
    
    if input_data.size == 0:
        raise HTTPException(status_code=400, detail="Input data is empty")
    
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
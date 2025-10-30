from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from typing import List, Annotated, Optional
from sqlalchemy.orm import Session
import os
import asyncio
import tempfile

from .auth import get_current_user
from database import SessionLocal
import models
from image import detect_blur_and_save
from .Mlpredict.form import predict_autism
from services.reporting import generate_and_store_report
from services.notifications import notification_manager

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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@router.post("/combined", status_code=201)
async def combined_prediction(
    file: Optional[UploadFile] = File(None),
    A1: int = Form(...),
    A2: int = Form(...),
    A3: int = Form(...),
    A4: int = Form(...),
    A5: int = Form(...),
    A6: int = Form(...),
    A7: int = Form(...),
    A8: int = Form(...),
    A9: int = Form(...),
    A10: int = Form(...),
    Age_Mons: int = Form(...),
    Sex: str = Form(...),
    Ethnicity: str = Form(...),
    Jaundice: str = Form(...),
    Family_mem_with_ASD: str = Form(...),
    current_user: dict = Depends(get_current_user),
    db: db_dependency = None
):
    input_data = {
        "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5,
        "A6": A6, "A7": A7, "A8": A8, "A9": A9, "A10": A10,
        "Age_Mons": Age_Mons,
        "Sex": Sex,
        "Ethnicity": Ethnicity,
        "Jaundice": Jaundice,
        "Family_mem_with_ASD": Family_mem_with_ASD,
    }
    form_prediction, form_probability = await predict_autism(input_data)
    form_probability_value = float(form_probability) if form_probability is not None else None

    video_label: Optional[str] = None
    video_confidence: Optional[float] = None
    gaze_percentage: Optional[float] = None
    video_path: Optional[str] = None

    if file is not None:
        if not file.content_type or not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Invalid video file")
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty video file")
        if len(contents) > 1000 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 1000MB")

        try:
            suffix = os.path.splitext(file.filename or "")[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                video_path = tmp.name

            frames, gaze_percentage = await detect_blur_and_save(video_path, max_frames=250)
            if frames.size == 0:
                raise HTTPException(status_code=400, detail="No sharp frames were detected in the video. Please upload a clearer video.")

            video_predictions = await asyncio.to_thread(make_prediction, frames)
            if video_predictions.size == 0:
                raise HTTPException(status_code=500, detail="Model returned an empty prediction.")

            avg_prediction = np.mean(video_predictions, axis=0)
            predicted_class_index = int(np.argmax(avg_prediction))
            video_label = ['Non_Autistic', 'Autistic'][predicted_class_index]
            video_confidence = float(np.max(avg_prediction) * 100)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(exc)}")
        finally:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)

    record = models.Data(
        user_email=current_user["email"],
        video_prediction=video_label,
        video_confidence=video_confidence,
        form_prediction=str(form_prediction),
        form_confidence=form_probability_value,
        eye_gaze_percentage=gaze_percentage,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    report_summary = generate_and_store_report(
        db=db,
        record=record,
        max_words=120,
        additional_notes=None,
    )

    await notification_manager.notify_report_ready(
        email=current_user["email"],
        payload={
            "type": "report_ready",
            "data_id": record.id,
            "report": report_summary,
            "video_prediction": video_label,
            "video_confidence": video_confidence,
            "form_prediction": form_prediction,
            "form_confidence": form_probability_value,
        },
    )

    response = {
        "form": {
            "predicted_class": int(form_prediction),
            "confidence": f"{form_probability_value:.2f}%" if form_probability_value is not None else None,
        },
        "report": report_summary,
    }
    if video_label is not None and video_confidence is not None:
        response["video"] = {
            "predicted_class": video_label,
            "confidence": f"{video_confidence:.2f}%",
        }
        response["eye_gaze"] = {
            "percentage": f"{gaze_percentage:.2f}%" if gaze_percentage is not None else None,
        }
    else:
        response["video"] = None
        response["eye_gaze"] = None

    return response
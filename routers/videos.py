import os
import asyncio
import numpy as np
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from typing import Annotated
import models
from database import SessionLocal
from image import detect_blur_and_save
from .auth import get_current_user
from .predictions import make_prediction

CLASS_LABELS = ['Non_Autistic', 'Autistic']
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
MAX_FILE_SIZE = 300 * 1024 * 1024
MAX_FRAMES_TO_PROCESS = 100
VIDEO_DIR = "videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

router = APIRouter(
    prefix="/video",
    tags=["video"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

async def validate_video_file(file: UploadFile) -> bool:
    """Validate if the uploaded file is a valid video"""
    if not file.content_type or not file.content_type.startswith('video/'):
        return False
    file_extension = Path(file.filename).suffix.lower()
    return file_extension in ALLOWED_VIDEO_EXTENSIONS

@router.post("", status_code=201)
async def create_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: db_dependency = None
):
    """Process video file and return autism prediction"""
    if not await validate_video_file(file):
        raise HTTPException(status_code=400, detail="Invalid video file")
    
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 300MB")
    
    video_path = os.path.join(VIDEO_DIR, file.filename)
    with open(video_path, "wb") as f:
        f.write(contents)
    
    try:
        frames = await detect_blur_and_save(video_path, max_frames=MAX_FRAMES_TO_PROCESS)
        
        if frames.size == 0:
            raise HTTPException(status_code=400, detail="No sharp frames were detected in the video. Please upload a clearer video.")

        predictions = await asyncio.to_thread(make_prediction, frames)
        
        if predictions.size == 0:
            raise HTTPException(status_code=500, detail="Model returned an empty prediction.")

        avg_prediction = np.mean(predictions, axis=0)
        predicted_class_index = np.argmax(avg_prediction)
        predicted_class_label = CLASS_LABELS[predicted_class_index]
        confidence = float(np.max(avg_prediction) * 100)
        
        data = models.Data(
            user_email=current_user['email'],
            prediction_type="video",
            predicted_class=predicted_class_label,
            confidence_probability=confidence
        )
        db.add(data)
        db.commit()

        return {"predicted_class": predicted_class_label, "confidence": f"{confidence:.2f}%"}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

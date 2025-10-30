from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Annotated, List
from pydantic import BaseModel
from datetime import datetime
import models
from database import SessionLocal
from .auth import get_current_user

router = APIRouter(
    prefix="/data",
    tags=["data"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

class PredictionHistoryItem(BaseModel):
    id: int
    prediction_type: str
    video_prediction: str | None
    video_confidence: float | None
    form_prediction: str | None
    form_confidence: float | None
    eye_gaze_percentage: float | None
    report_text: str | None
    overall: float | None
    timestamp: datetime

    class Config:
        from_attributes = True

@router.get("/history", response_model=List[PredictionHistoryItem])
async def get_prediction_history(
    current_user: dict = Depends(get_current_user),
    db: db_dependency = None
):
    """Get all prediction history for the authenticated user"""
    records = (
        db.query(models.Data)
        .filter(models.Data.user_email == current_user['email'])
        .order_by(models.Data.timestamp.desc())
        .all()
    )

    history: List[PredictionHistoryItem] = []
    for item in records:
        has_video = item.video_prediction is not None
        has_form = item.form_prediction is not None
        prediction_type = "combined" if has_video and has_form else ("video" if has_video else "form")
        predicted_class = item.video_prediction if has_video else item.form_prediction
        confidence = item.video_confidence if has_video else item.form_confidence

        history.append(
            PredictionHistoryItem(
                id=item.id,
                prediction_type=prediction_type,
                video_prediction=item.video_prediction,
                video_confidence=item.video_confidence,
                form_prediction=item.form_prediction,
                form_confidence=item.form_confidence,
                eye_gaze_percentage=item.eye_gaze_percentage,
                report_text=item.report_text,
                overall=(item.form_confidence*100+item.video_confidence+item.eye_gaze_percentage)/2 if has_video and has_form else (item.video_confidence*100 if has_video else (item.form_confidence*100 if has_form else None)),
                timestamp=item.timestamp,

            )
        )
    return history

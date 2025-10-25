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
    predicted_class: str | int
    confidence_probability: float
    timestamp: datetime

    class Config:
        from_attributes = True

@router.get("/history", response_model=List[PredictionHistoryItem])
async def get_prediction_history(
    current_user: dict = Depends(get_current_user),
    db: db_dependency = None
):
    """Get all prediction history for the authenticated user"""
    predictions = (
        db.query(models.Data)
        .filter(models.Data.user_email == current_user['email'])
        .order_by(models.Data.timestamp.desc())
        .all()
    )
    return predictions

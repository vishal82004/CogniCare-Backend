from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session
from typing import Annotated
import models
from database import SessionLocal
from .auth import get_current_user
from .Mlpredict.form import predict_autism

router = APIRouter(
    prefix="/forms",
    tags=["forms"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@router.post("", status_code=201)
async def submit_form(
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
    """Submit form-based autism assessment and get prediction"""
    input_data = {
        "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5,
        "A6": A6, "A7": A7, "A8": A8, "A9": A9, "A10": A10,
        "Age_Mons": Age_Mons,
        "Sex": Sex,
        "Ethnicity": Ethnicity,
        "Jaundice": Jaundice,
        "Family_mem_with_ASD": Family_mem_with_ASD,
    }

    prediction, probability = await predict_autism(input_data)

    data = models.Data(
        user_email=current_user['email'],
        prediction_type="form",
        predicted_class=int(prediction),
        confidence_probability=float(probability) if probability is not None else None
    )
    db.add(data)
    db.commit()

    return {"prediction": int(prediction), "probability": float(probability) if probability is not None else None}

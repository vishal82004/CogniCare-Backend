import os
from typing import Optional

from fastapi import HTTPException
from groq import Groq
from sqlalchemy.orm import Session

import models

def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is required.")
    return Groq(api_key=api_key)

def generate_and_store_report(
    db: Session,
    record: models.Data,
    max_words: int = 120,
    additional_notes: Optional[str] = None,
) -> str:
    primary_result = record.video_prediction or record.form_prediction or "Unavailable"
    prediction_kind = "combined" if record.video_prediction and record.form_prediction else (
        "video" if record.video_prediction else "form"
    )

    prompt_parts = [
        "Create a concise, supportive summary for parents based on the following autism assessment data.",
        f"Prediction type: {prediction_kind}",
        f"Primary result: {primary_result}",
    ]

    if record.video_prediction:
        prompt_parts.append(f"Video outcome: {record.video_prediction} (confidence {record.video_confidence:.2f}%).")
    if record.eye_gaze_percentage is not None:
        prompt_parts.append(f"Eye gaze stability: {record.eye_gaze_percentage:.2f}%.")
    if record.form_prediction:
        prompt_parts.append(f"Form assessment: {record.form_prediction} (confidence {record.form_confidence:.2f}%).")
    if additional_notes:
        prompt_parts.append(f"Additional notes: {additional_notes}")

    prompt = " ".join(prompt_parts)

    client = _client()
    try:
        result = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a compassionate clinician writing brief, easy-to-understand summaries for parents."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=min(800, max_words * 4),
            temperature=0.3,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Groq API error: {exc}") from exc

    summary = ""
    if result.choices:
        summary = (result.choices[0].message.content or "").strip()

    if not summary:
        raise HTTPException(status_code=502, detail="LLM returned an empty response.")

    record.report_text = summary
    db.commit()
    db.refresh(record)
    return summary

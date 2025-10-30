from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    username = Column(String)
    email = Column(String, primary_key=True, index=True)
    hashed_password = Column(String)

    data = relationship("Data", back_populates="user")

class Data(Base):
    __tablename__ = "data"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, ForeignKey("users.email"))
    video_prediction = Column(String, nullable=True)
    video_confidence = Column(Float, nullable=True)
    form_prediction = Column(String, nullable=True)
    form_confidence = Column(Float, nullable=True)
    eye_gaze_percentage = Column(Float, nullable=True)
    report_text = Column(Text, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="data")
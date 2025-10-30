import os
from fastapi import FastAPI
from dotenv import load_dotenv
import models
from database import engine
from routers import auth, data, predictions, notifications

# Load environment variables first
load_dotenv()

app = FastAPI(title="Cognicare API", description="Autism Detection API")

# Create the database tables
models.Base.metadata.create_all(bind=engine)

# Include routers
app.include_router(auth.router)
app.include_router(data.router)
app.include_router(predictions.router)
app.include_router(notifications.router)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
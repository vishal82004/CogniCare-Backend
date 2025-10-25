import os
from fastapi import FastAPI
from dotenv import load_dotenv
import models
from database import engine
from routers import auth, videos, forms, data

# Load environment variables first
load_dotenv()

app = FastAPI(title="Cognicare API", description="Autism Detection API")

# Create the database tables
models.Base.metadata.create_all(bind=engine)

# Include routers
app.include_router(auth.router)
app.include_router(videos.router)
app.include_router(forms.router)
app.include_router(data.router)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
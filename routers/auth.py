import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Annotated
from pydantic import BaseModel, EmailStr
from starlette import status
from models import User
from database import SessionLocal
from passlib.context import CryptContext 
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

MAX_PASSWORD_BYTES = 72
load_dotenv()

router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
OAuth2_Bearer = OAuth2PasswordBearer(tokenUrl="/auth/token")
secret_key = os.getenv("SECRET_KEY")
algorithm = os.getenv("ALGORITHM")

if not secret_key or not algorithm:
    raise RuntimeError("SECRET_KEY and ALGORITHM must be set in environment variables.")

class CreateUserRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@router.post("/", status_code=status.HTTP_201_CREATED)
def create_user(user: CreateUserRequest, db: db_dependency):
    if len(user.password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise HTTPException(status_code=400, detail="Password must be 72 bytes or fewer")
    
    existing_user = (
        db.query(User)
        .filter((User.username == user.username) | (User.email == user.email))
        .first()
    )
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=bcrypt_context.hash(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

def authenticate_user(email: str, password: str, db: Session):
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    user = db.query(User).filter(User.email == email).first()
    if not user or not bcrypt_context.verify(password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return user

def create_access_token(email: str, user_email: str, expires_delta: timedelta = timedelta(minutes=260)):
    encode = {"sub": email, "email": user_email}
    expires = datetime.now(timezone.utc) + expires_delta
    encode.update({"exp": expires}) 
    return jwt.encode(encode, secret_key, algorithm=algorithm)

def get_current_user(token: str = Depends(OAuth2_Bearer)):
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        email: str = payload.get("sub")
        user_email: str = payload.get("email")
        if email is None or user_email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return {"email": email, "user_email": user_email}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

@router.get("/")
def get_users(db: db_dependency):
    users = db.query(User).all()
    return users

@router.post("/token", response_model=Token)
def get_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: db_dependency):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.email, user.email, timedelta(minutes=15))
    return {"access_token": token, "token_type": "bearer"}
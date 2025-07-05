import os
import sys
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
import secrets
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import get_db, create_user, authenticate_user, create_session, get_user_by_session

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBasic()

# Pydantic models
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    # Register a new user
    # Check if username or email already exists
    from database import User
    existing_user = db.query(User).filter((User.username == user_data.username) | (User.email == user_data.email)).first()
    if existing_user:
        if existing_user.username == user_data.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    # Create new user
    user = create_user(db, user_data.username, user_data.email, user_data.password)
    return UserResponse(id=user.id, username=user.username, email=user.email)

@router.post("/login")
async def login(
    response: Response,
    credentials: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """Login user and create session"""
    user = authenticate_user(db, credentials.username, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    # Create session
    session_id = secrets.token_urlsafe(32)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    create_session(db, user.id, session_id, ip_address, user_agent)
    # Set session cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=True,  # Set to True in production with HTTPS
        samesite="lax",
        max_age=86400  # 24 hours
    )
    return {
        "message": "Login successful",
        "user": UserResponse(id=user.id, username=user.username, email=user.email)
    }

@router.post("/logout")
async def logout(response: Response, request: Request, db: Session = Depends(get_db)):
    """Logout user and delete session"""
    session_id = request.cookies.get("session_id")
    
    if session_id:
        from database import UserSession
        session = db.query(UserSession).filter(UserSession.session_id == session_id).first()
        
        if session:
            db.delete(session)
            db.commit()
    
    # Clear cookie
    response.delete_cookie("session_id")
    
    return {"message": "Logout successful"}

@router.get("/me", response_model=UserResponse)
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Get current user info"""
    session_id = request.cookies.get("session_id")
    
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    user = get_user_by_session(db, session_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session"
        )
    
    return UserResponse(id=user.id, username=user.username, email=user.email)

def get_current_user_dependency(request: Request, db: Session = Depends(get_db)):
    """Dependency to get current user"""
    session_id = request.cookies.get("session_id")
    
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    user = get_user_by_session(db, session_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session"
        )
    
    return user
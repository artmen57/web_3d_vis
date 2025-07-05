from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, LargeBinary, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import bcrypt
import os

# Database URL from environment or default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://myuser:mypassword@localhost/3dmodels")

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()




class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    
    # Relationships
    models = relationship("Model3D", back_populates="owner", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password: str):
        """Hash and set password"""
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str) -> bool:
        """Verify password"""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class Model3D(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Store OBJ file content
    obj_content = Column(Text, nullable=False)
    
    # Model metadata
    vertex_count = Column(Integer)
    face_count = Column(Integer)
    file_size = Column(Integer)
    
    # Bounding box for spatial queries (could use PostGIS here if needed)
    bbox_min_x = Column(Float)
    bbox_min_y = Column(Float)
    bbox_min_z = Column(Float)
    bbox_max_x = Column(Float)
    bbox_max_y = Column(Float)
    bbox_max_z = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_accessed = Column(DateTime(timezone=True))
    
    # Thumbnail (base64 encoded JPEG)
    thumbnail = Column(Text)
    
    # Relationships
    owner = relationship("User", back_populates="models")
    shared_with = relationship("ModelShare", back_populates="model", cascade="all, delete-orphan")

class ModelShare(Base):
    __tablename__ = "model_shares"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    shared_with_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    permission = Column(String(20), default="view")  # view, edit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    model = relationship("Model3D", back_populates="shared_with")
    shared_with_user = relationship("User")

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Session data
    ip_address = Column(String(45))
    user_agent = Column(String(255))
    
    # Currently loaded model
    current_model_id = Column(Integer, ForeignKey("models.id"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    current_model = relationship("Model3D")

# Database functions
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def create_user(db, username: str, email: str, password: str):
    """Create a new user"""
    user = User(username=username, email=email)
    user.set_password(password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db, username: str, password: str):
    """Authenticate user by username/email and password"""
    user = db.query(User).filter(
        (User.username == username) | (User.email == username)
    ).first()
    
    if user and user.verify_password(password):
        user.last_login = datetime.utcnow() #now()
        db.commit()
        return user
    return None

def create_session(db, user_id: int, session_id: str, ip_address: str = None, user_agent: str = None):
    """Create a new user session"""
    session = UserSession(
        session_id=session_id,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent
    )
    db.add(session)
    db.commit()
    return session

def get_user_by_session(db, session_id: str):
    """Get user by session ID"""
    session = db.query(UserSession).filter(
        UserSession.session_id == session_id
    ).first()
    
    if session:
        # Update last activity
        session.last_activity = datetime.utcnow()
        db.commit()
        return session.user
    return None

def save_model(db, user_id: int, name: str, obj_content: str, vertex_count: int, face_count: int, thumbnail: str = None):
    """Save a 3D model to database"""
    model = Model3D(
        user_id=user_id,
        name=name,
        obj_content=obj_content,
        vertex_count=vertex_count,
        face_count=face_count,
        file_size=len(obj_content),
        thumbnail=thumbnail
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model

def get_user_models(db, user_id: int):
    """Get all models for a user"""
    return db.query(Model3D).filter(Model3D.user_id == user_id).all()

def get_model_by_id(db, model_id: int, user_id: int):
    """Get model by ID with permission check"""
    # Check if user owns the model
    model = db.query(Model3D).filter(
        Model3D.id == model_id,
        Model3D.user_id == user_id
    ).first()
    
    if model:
        return model
    
    # Check if model is shared with user
    share = db.query(ModelShare).filter(
        ModelShare.model_id == model_id,
        ModelShare.shared_with_user_id == user_id
    ).first()
    
    if share:
        return share.model
    
    return None

if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
    print("Database initialized successfully!")
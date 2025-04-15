from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy import (
    Integer, String, Boolean, 
    DateTime, ForeignKey, JSON, Text
)
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import List, Optional, Dict

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    spotify_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String)
    email: Mapped[Optional[str]] = mapped_column(String)
    profile_image: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    allow_data_usage: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    playlists: Mapped[List["Playlist"]] = relationship("Playlist", back_populates="user")
    feedback: Mapped[List["Feedback"]] = relationship("Feedback", back_populates="user")
    profile: Mapped[Optional["UserProfile"]] = relationship(
        "UserProfile", uselist=False, back_populates="user"
    )
    conversations: Mapped[List["UserConversation"]] = relationship(
        "UserConversation", back_populates="user"
    )
    preferences: Mapped[Optional["UserPreference"]] = relationship(
        "UserPreference", uselist=False, back_populates="user"
    )

class Playlist(Base):
    __tablename__ = "playlists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String)
    spotify_playlist_id: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="playlists")

class TrackRecommendation(BaseModel):
    id: str
    name: str
    artists: List[str]
    preview_url: Optional[str]
    external_url: str
    features: Dict[str, float]

class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    track_id: Mapped[str] = mapped_column(String)
    liked: Mapped[bool] = mapped_column(Boolean)
    feedback_type: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="feedback")

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    mood_preference: Mapped[Optional[str]] = mapped_column(String)
    genre_preference: Mapped[Optional[str]] = mapped_column(String)
    activity_preference: Mapped[Optional[str]] = mapped_column(String)

    user: Mapped["User"] = relationship("User", back_populates="profile")

class UserConversation(Base):
    __tablename__ = "user_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    message: Mapped[str] = mapped_column(Text)
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship("User", back_populates="conversations")

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    preferences: Mapped[Dict] = mapped_column(JSON, default={})

    user: Mapped["User"] = relationship("User", back_populates="preferences")

def test_models():
    """Test all models with synchronous SQLAlchemy"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Setup in-memory database
    engine = create_engine("sqlite:///:memory:", echo=True)
    Session = sessionmaker(bind=engine)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Test operations
    with Session() as session:
        print("\n🔧 Testing User model...")
        user = User(
            spotify_id="test_123",
            display_name="Test User",
            email="test@example.com"
        )
        session.add(user)
        session.commit()
        
        print("✅ User created:", user.display_name)
        
        # Test relationships
        print("\n🔍 Testing relationships...")
        user.profile = UserProfile(
            mood_preference="happy",
            genre_preference="rock"
        )
        user.playlists.append(Playlist(
            name="My Playlist",
            spotify_playlist_id="spotify:123"
        ))
        user.feedback.append(Feedback(
            track_id="track_123",
            liked=True,
            feedback_type="explicit"
        ))
        user.conversations.append(UserConversation(
            message="Hello bot!",
            is_bot=False
        ))
        user.preferences = UserPreference(preferences={"theme": "dark"})
        
        session.commit()
        
        # Verify data
        test_user = session.get(User, user.id)
        print(f"📊 Profile mood: {test_user.profile.mood_preference}")
        print(f"📊 Playlists: {len(test_user.playlists)}")
        print(f"📊 Feedback items: {len(test_user.feedback)}")
        print(f"📊 Conversations: {len(test_user.conversations)}")
        print(f"📊 Preferences: {test_user.preferences.preferences}")
    
    print("\n🎉 All tests passed successfully!")

if __name__ == "__main__":
    test_models()




'''from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, 
    ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func
from db_base import Base 

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    spotify_id = Column(String, unique=True, index=True)
    display_name = Column(String)
    email = Column(String)
    profile_image = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    allow_data_usage = Column(Boolean, default=True)

    playlists = relationship("Playlist", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")
    profile = relationship("UserProfile", uselist=False, back_populates="user")
    conversations = relationship("UserConversation", back_populates="user")
    preferences = relationship("UserPreference", uselist=False, back_populates="user")
    user_conversations = relationship("UserConversation", back_populates="user")

class Playlist(Base):
    __tablename__ = "playlists"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    spotify_playlist_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="playlists")

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    track_id = Column(String)
    liked = Column(Boolean)
    feedback_type = Column(String)  # 'explicit' or 'implicit'
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="feedback")

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mood_preference = Column(String)
    genre_preference = Column(String)
    activity_preference = Column(String)

    user = relationship("User", back_populates="profile")

class UserConversation(Base):
    __tablename__ = 'user_conversations'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    message = Column(String)
    is_bot = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="user_conversations")

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    preferences = Column(JSON, default={})  # Stores learned preferences

    user = relationship("User", back_populates="preferences")

'''







'''import os
import asyncio
import logging
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, ForeignKey, Text, text
)
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded environment variables (set your values here)
SPOTIFY_CLIENT_ID = "your_spotify_client_id"
SPOTIFY_CLIENT_SECRET = "your_spotify_client_secret"
DATABASE_URL = "sqlite+aiosqlite:///./music_recommender.db"

# SQLAlchemy setup
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# -------------------- MODELS --------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    spotify_id = Column(String, unique=True, index=True)
    display_name = Column(String)
    email = Column(String)
    profile_image = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    allow_data_usage = Column(Boolean, default=True)

    playlists = relationship("Playlist", back_populates="user", cascade="all, delete")
    feedbacks = relationship("Feedback", back_populates="user", cascade="all, delete")
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete")


class Playlist(Base):
    __tablename__ = "playlists"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="playlists")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="feedbacks")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mood_preference = Column(String)
    genre_preference = Column(String)

    user = relationship("User", back_populates="profile")

# -------------------- DB TEST & MAIN --------------------

async def test_database_connection():
    """Test DB connection using SELECT 1"""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful!")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

async def main():
    # Create tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Tables created successfully!")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")

    # Test DB connection
    await test_database_connection()

if __name__ == "__main__":
    asyncio.run(main())


'''





'''import asyncio
import logging
from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ----------------------
# Config: No .env, just hardcoded
# ----------------------
DATABASE_URL = "sqlite+aiosqlite:///./music_recommender.db"

# ----------------------
# Logging config
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# SQLAlchemy base and engine
# ----------------------
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# ----------------------
# Models
# ----------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    spotify_id = Column(String, unique=True, index=True)
    display_name = Column(String)
    email = Column(String)
    profile_image = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow)
    allow_data_usage = Column(Boolean, default=True)

    playlists = relationship("Playlist", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")
    profile = relationship("UserProfile", back_populates="user", uselist=False)


class Playlist(Base):
    __tablename__ = "playlists"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="playlists")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    comment = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="feedbacks")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mood_preference = Column(String)
    genre_preference = Column(String)

    user = relationship("User", back_populates="profile")


# ----------------------
# Utility: Create tables and test connection
# ----------------------

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Tables created successfully!")

    # Simple connection test
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        logger.info("✅ Database connection successful!")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")


# ----------------------
# Run on script exec
# ----------------------
if __name__ == "__main__":
    asyncio.run(init_db())

'''

'''import os
import asyncio
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

REQUIRED_ENV_VARS = [
    "SPOTIFY_CLIENT_ID",
    "SPOTIFY_CLIENT_SECRET",
    "DATABASE_URL"
]

def check_dependencies():
    """Check for required Python packages"""
    required_packages = ['async_timeout', 'asyncpg', 'sqlalchemy']
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing

def check_env_variables():
    """Check required environment variables"""
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    return missing_vars

async def test_database_connection():
    """Test database connection with proper error handling"""
    database_url = os.getenv("DATABASE_URL")
    try:
        engine = create_async_engine(database_url, echo=True)
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

async def main():
    # Check dependencies first
    if missing_packages := check_dependencies():
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install them with: pip install " + " ".join(missing_packages))
        return

    # Check environment variables
    if missing_vars := check_env_variables():
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return

    logger.info("All environment variables are present")

    # Test database connection
    if await test_database_connection():
        logger.info("✅ Database connection successful")
    else:
        logger.error("Failed to connect to database")

if __name__ == "__main__":
    asyncio.run(main())

'''
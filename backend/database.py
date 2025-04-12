from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.future import select
from sqlalchemy import text
from models import (
    User, Playlist, Feedback, UserProfile,
    UserConversation
)

DATABASE_URL = "sqlite+aiosqlite:///./music_recommender.db"

# Engine and session
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Async session generator
async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session

# Create tables
async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Connection test
async def test_connection():
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        print("✅ Database connection successful!")

# User profile fetch
async def get_user_profile(user_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        return result.scalars().first()

# Save user message to conversation
async def save_user_message(user_id: int, message: str):
    async with AsyncSessionLocal() as session:
        convo = UserConversation(user_id=user_id, message=message)
        session.add(convo)
        await session.commit()

# Get full conversation history
async def get_user_conversation_history(user_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserConversation).where(UserConversation.user_id == user_id)
        )
        conversations = result.scalars().all()
        return [{"message": c.message, "timestamp": c.timestamp} for c in conversations]

# Save interaction with recommendations
async def save_interaction(user_id: int, description: str, mood_vector: dict, recommendations: list):
    """Save user interaction to database"""
    async with AsyncSessionLocal() as session:
        # Save conversation
        convo = UserConversation(user_id=user_id, message=description)
        session.add(convo)
        await session.commit()

# Get user history for recommendations
async def get_user_history(user_id: int) -> list:
    """Get user's conversation history"""
    return await get_user_conversation_history(user_id)

# Manual run
if __name__ == "__main__":
    import asyncio
    asyncio.run(init())
    asyncio.run(test_connection())



'''import os
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, JSON,
    ForeignKey, Float, DateTime,
    Boolean, Text
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from sqlalchemy.future import select

# ✅ Hardcoded SQLite URL
DATABASE_URL = "sqlite+aiosqlite:///./music.db"

# Create engine
def get_engine():
    return create_async_engine(DATABASE_URL, echo=True)

engine = get_engine()

# Session
AsyncSessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession
)

Base = declarative_base()

# Session dependency
async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    spotify_id = Column(String, unique=True, index=True)
    display_name = Column(String)
    email = Column(String)
    profile_image = Column(String)
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime)
    allow_data_usage = Column(Boolean, default=True)

    playlists = relationship("Playlist", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")
    profile = relationship("UserProfile", uselist=False, back_populates="user")

class Playlist(Base):
    __tablename__ = "playlists"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="playlists")

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    comment = Column(Text)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="feedback")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mood_preference = Column(String)
    genre_preference = Column(String)

    user = relationship("User", back_populates="profile")

# ✅ Create tables
async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Tables created successfully!")

# ✅ Test connection
async def test_connection():
    from sqlalchemy import text
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        print("✅ Database connection successful!")

# Run script
if __name__ == "__main__":
    import asyncio
    asyncio.run(init())
    asyncio.run(test_connection())

# ✅ Helper functions
async def get_user_profile(user_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        return result.scalars().first()

# Simulated function
async def get_song_features(song_id: int):
    return {
        "genre": "pop",
        "tempo": 120,
        "mood": "happy"
    }

'''








'''import os

print("📂 .env DATABASE_URL from OS:", repr(os.environ.get("DATABASE_URL")))
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, JSON, 
    ForeignKey, Float, DateTime, 
    Boolean, Text
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from sqlalchemy.future import select
from dotenv import load_dotenv
load_dotenv()


raw_url = os.getenv("DATABASE_URL")
print("🔍 URL character by character:")
for i, ch in enumerate(raw_url):
    print(f"{i}: {repr(ch)}")


# ✅ Fix: Correct way to get environment variable with fallback default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./music.db")

# Initialize engine
def get_engine():
    return create_async_engine(DATABASE_URL, echo=True)

engine = get_engine()

# Session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession
)

Base = declarative_base()

# Database session dependency
async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    spotify_id = Column(String, unique=True, index=True)
    display_name = Column(String)
    email = Column(String)
    profile_image = Column(String)
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime)
    allow_data_usage = Column(Boolean, default=True)
    
    playlists = relationship("Playlist", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")
    profile = relationship("UserProfile", uselist=False, back_populates="user")

# Sample placeholders for other models (add your own fields)
class Playlist(Base):
    __tablename__ = "playlists"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    created_at = Column(DateTime, default=func.now())
    
    user = relationship("User", back_populates="playlists")

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    comment = Column(Text)
    created_at = Column(DateTime, default=func.now())

    user = relationship("User", back_populates="feedback")

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mood_preference = Column(String)
    genre_preference = Column(String)
    
    user = relationship("User", back_populates="profile")

# Test connection
async def test_connection():
    from sqlalchemy import text
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        print("✅ Database connection successful!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_connection())

# Function to get user profile
async def get_user_profile(user_id: int):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        return result.scalars().first()

# Dummy function to simulate song feature retrieval
# You should replace this with your actual DB model logic if needed
async def get_song_features(song_id: int):
    return {
        "genre": "pop",
        "tempo": 120,
        "mood": "happy"
    }

'''
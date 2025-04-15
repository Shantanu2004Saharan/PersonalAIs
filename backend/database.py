import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, Integer, String, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from datetime import datetime
import os
from typing import List, Dict, Optional
from sqlalchemy import text 
from models import UserConversation



# --- Database Setup ---
DATABASE_URL = "sqlite+aiosqlite:///./music_recommender.db"
Base = declarative_base()

# --- Model Definitions (SQLAlchemy 2.0 style) ---
class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    conversations: Mapped[List["UserConversation"]] = relationship(back_populates="user")
    preferences: Mapped[Optional["UserPreference"]] = relationship(back_populates="user")
    feedback: Mapped[List["Feedback"]] = relationship(back_populates="user")

class UserConversation(Base):
    __tablename__ = 'user_conversations'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    message: Mapped[str] = mapped_column(Text)
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="conversations")

class UserPreference(Base):
    __tablename__ = 'user_preferences'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    preferences: Mapped[Dict] = mapped_column(JSON)
    
    user: Mapped["User"] = relationship(back_populates="preferences")

class Feedback(Base):
    __tablename__ = 'feedback'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id'))
    track_id: Mapped[str] = mapped_column(String(50))
    liked: Mapped[bool] = mapped_column(Boolean)
    feedback_type: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="feedback")

# --- Database Engine & Session ---
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# --- Core Functions ---
async def init_db():
    """Initialize database and create tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created!")

async def get_db():
    """Generator function to get database session"""
    async with AsyncSessionLocal() as session:
        yield session

async def create_user(username: str, email: str = None) -> User:
    """Create a new user"""
    async with AsyncSessionLocal() as session:
        user = User(username=username, email=email)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

async def save_conversation(user_id: int, message: str, is_bot: bool = False) -> UserConversation:
    """Save a conversation message"""
    async with AsyncSessionLocal() as session:
        conv = UserConversation(
            user_id=user_id,
            message=message,
            is_bot=is_bot
        )
        session.add(conv)
        await session.commit()
        await session.refresh(conv)
        return conv

async def get_conversations(user_id: int, limit: int = 10) -> List[UserConversation]:
    """Get conversation history for a user (raw SQL version)"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT * FROM user_conversations WHERE user_id = :user_id ORDER BY timestamp DESC LIMIT :limit")
            .bindparams(user_id=user_id, limit=limit)
        )
        return result.mappings().all()

async def get_user_conversation_history(user_id: int, limit: int = 10) -> List[UserConversation]:
    """Get conversation history for a user (ORM version)"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserConversation)
            .where(UserConversation.user_id == user_id)
            .order_by(UserConversation.timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()

async def update_user_preferences(user_id: int, preferences: Dict) -> UserPreference:
    """Update user preferences in database"""
    async with AsyncSessionLocal() as session:
        # Get existing preferences
        result = await session.execute(
            select(UserPreference)
            .where(UserPreference.user_id == user_id)
        )
        pref = result.scalars().first()
        
        if not pref:
            pref = UserPreference(user_id=user_id, preferences=preferences)
            session.add(pref)
        else:
            pref.preferences.update(preferences)
        
        await session.commit()
        await session.refresh(pref)
        return pref

async def save_feedback(user_id: int, track_id: str, liked: bool, feedback_type: str = "explicit") -> Feedback:
    """Save user feedback"""
    async with AsyncSessionLocal() as session:
        feedback = Feedback(
            user_id=user_id,
            track_id=track_id,
            liked=liked,
            feedback_type=feedback_type
        )
        session.add(feedback)
        await session.commit()
        await session.refresh(feedback)
        return feedback
    
save_user_message = save_conversation

# --- Test Functions ---
async def run_tests():
    """Comprehensive test of all database functions"""
    print("\n🚀 Starting database tests...")
    
    # Initialize database
    await init_db()
    
    # Test user creation
    user = await create_user("test_user", "test@example.com")
    print(f"✅ Created user: {user.username} (ID: {user.id})")
    
    # Test conversation saving
    await save_conversation(user.id, "Hello, AI!")
    await save_conversation(user.id, "How are you?", is_bot=True)
    print("✅ Saved conversation messages")
    
    # Test both conversation retrieval methods
    convs_sql = await get_conversations(user.id)
    convs_orm = await get_user_conversation_history(user.id)
    print(f"📜 Conversation history ({len(convs_orm)} messages via ORM):")
    for conv in convs_orm:
        print(f"- {'Bot' if conv.is_bot else 'User'}: {conv.message}")
    
    # Test preferences
    prefs = await update_user_preferences(user.id, {"theme": "dark", "volume": 80})
    print(f"⚙️ Updated user preferences: {prefs.preferences}")
    
    # Test feedback
    feedback = await save_feedback(user.id, "track_123", True)
    print(f"👍 Saved feedback for track {feedback.track_id}")
    
    # Verify all data
    async with AsyncSessionLocal() as session:
        # Check user count
        users = await session.execute(select(User))
        print(f"\n📊 Database contains {len(users.scalars().all())} users")
        
        # Check tables
        tables = await session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        print("Existing tables:", [t[0] for t in tables])
    
    print("\n🎉 All tests completed successfully!")

if __name__ == "__main__":
    # Clear old database file if exists
    if os.path.exists("./music_recommender.db"):
        os.remove("./music_recommender.db")
    
    # Run tests
    asyncio.run(run_tests())







'''from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text
from backend.models import user, playlist, Feedback, UserProfile, UserConversation, UserPreference
from db_base import Base 
import sys
import os
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite+aiosqlite:///./music_recommender.db"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Async engine for SQLite
engine = create_async_engine(DATABASE_URL, echo=True, future=True)

# Session local for async
AsyncSessionLocal = sessionmaker(
    bind=engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Async session getter
async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session

# Create tables in the database asynchronously
async def init_db():
    async with engine.begin() as conn:
        print("✨ Creating database tables...")
        # Drop all tables first (for development only!)
        await conn.run_sync(Base.metadata.drop_all)
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created!")
        # Verify tables were created
        tables = await conn.run_sync(
            lambda sync_conn: sync_conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        )
        print("📊 Existing tables:", tables)

# Test database connection
async def test_connection():
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        print("✅ Database connection successful!")

# Fetch user conversation history
async def get_user_conversation_history(user_id: int, limit: int = 10):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserConversation)
            .where(UserConversation.user_id == user_id)
            .order_by(UserConversation.timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()

# Save user message to the database
async def save_user_message(user_id: int, message: str, is_bot: bool = False):
    async with AsyncSessionLocal() as session:
        convo = UserConversation(
            user_id=user_id,
            message=message,
            is_bot=is_bot
        )
        session.add(convo)
        await session.commit()

# Update user preferences
async def update_user_preferences(user_id: int, preferences: dict):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(UserPreference)
            .where(UserPreference.user_id == user_id)
        )
        pref = result.scalars().first()
        
        if not pref:
            pref = UserPreference(user_id=user_id, preferences={})
            session.add(pref)
        
        pref.preferences.update(preferences)
        await session.commit()

# Save feedback on recommendations
async def save_recommendation_feedback(user_id: str, track_id: str, liked: bool):
    async with AsyncSessionLocal() as session:
        feedback = Feedback(
            user_id=user_id,
            track_id=track_id,
            liked=liked,
            feedback_type="explicit"  # Direct user feedback
        )
        session.add(feedback)
        await session.commit()

print("Database script ran successfully.")


async def run_tests():
    print("\n🔍 Checking tables...")
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        print("Existing tables:", result.fetchall())

# Main test function to run all the tests
if __name__ == "__main__":
    import asyncio

    async def run_tests():
        print("🔧 Running database tests...\n")
        print("🛠️ Initializing database...")
        # Initialize database (ensure tables are created)
        await init_db()
        # Test connection
        await test_connection()

        # Save a user message
        print("💬 Saving user message...")
        await save_user_message(user_id=1, message="Hello, AI!", is_bot=False)

        # Fetch user conversation history
        print("📜 Fetching conversation history...")
        history = await get_user_conversation_history(user_id=1)
        print("🗂️ Conversation History:", history)

        # Update user preferences
        print("⚙️ Updating preferences...")
        await update_user_preferences(user_id=1, preferences={"genre": "jazz"})

        # Save recommendation feedback
        print("👍 Saving feedback...")
        await save_recommendation_feedback(user_id="1", track_id="track_123", liked=True)

        print("\n✅ All tests executed.")

    asyncio.run(run_tests())



'''


'''from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
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

'''

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
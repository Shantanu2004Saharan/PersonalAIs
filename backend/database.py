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

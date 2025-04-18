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
# === Imports ===
from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy import (
    Integer, String, Boolean,
    DateTime, ForeignKey, JSON, Text, create_engine
)
from sqlalchemy.orm import (
    relationship, declarative_base,
    Mapped, mapped_column, sessionmaker
)
from sqlalchemy.sql import func
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import logging
import unittest
from unittest.mock import MagicMock, patch

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Base ===
Base = declarative_base()

# === SQLAlchemy Models ===

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

    playlists: Mapped[List["Playlist"]] = relationship("Playlist", back_populates="user")
    feedback: Mapped[List["Feedback"]] = relationship("Feedback", back_populates="user")
    profile: Mapped[Optional["UserProfile"]] = relationship("UserProfile", uselist=False, back_populates="user")
    conversations: Mapped[List["UserConversation"]] = relationship("UserConversation", back_populates="user")
    preferences: Mapped[Optional["UserPreference"]] = relationship("UserPreference", uselist=False, back_populates="user")

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
    preview_url: Optional[str] = None
    external_url: Optional[str] = None
    features: Dict[str, float]
    valence: Optional[float] = None

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

# === Music Matcher ===

class MusicMatcher:
    def __init__(self, spotify_client):
        self.spotify = spotify_client
        self.emotion_cache = {}

    async def find_recommendations(self, emotion_profile: Dict) -> List[Dict]:
        params = self._create_spotify_params(emotion_profile)
        tracks = await self.spotify.get_recommendations(**params)
        return self._rerank_tracks(tracks, emotion_profile)

    def _create_spotify_params(self, emotion_profile) -> Dict:
        music_profile = emotion_profile['music_profile']
        return {
            'limit': 20,  # Request more songs, e.g., 20
            'target_valence': music_profile['valence'],
            'target_energy': music_profile['energy'],
            'target_danceability': music_profile['danceability'],
            'min_acousticness': max(0, 1 - music_profile['complexity'] - 0.2),
            'max_acousticness': min(1, 1 - music_profile['complexity'] + 0.2),
            'seed_genres': self._select_genres(emotion_profile)
    }

    '''async def find_recommendations(self, emotion_profile: Dict) -> List[Dict]:
        params = self._create_spotify_params(emotion_profile)
        tracks = await self.spotify.get_recommendations(**params)
    
    # Ensure at least 10 songs are suggested
        if len(tracks) < 10:
            logger.warning(f"Only {len(tracks)} songs found, relaxing filters.")
        # Optionally, relax filters here, e.g., reduce complexity or energy requirements
            tracks = await self.spotify.get_recommendations(**self._relax_filters(params))
    
            return self._rerank_tracks(tracks, emotion_profile) '''
    
    async def find_recommendations(self, emotion_profile: Dict) -> List[Dict]:
        params = self._create_spotify_params(emotion_profile)
        tracks = await self.spotify.get_recommendations(**params)

    # Keep relaxing filters until we have at least 10 songs
        relaxation_attempts = 0
        while len(tracks) < 10 and relaxation_attempts < 5:  # Limit to 5 relax attempts to avoid infinite loop
            logger.warning(f"Only {len(tracks)} songs found, relaxing filters.")
            tracks = await self.spotify.get_recommendations(**self._relax_filters(params))
            relaxation_attempts += 1

    # If more than 12 songs are found, limit to 12
        tracks = tracks[:12]
        return self._rerank_tracks(tracks, emotion_profile)
    
    def _relax_filters(self, params):
        """Relax the filters to ensure more songs are returned."""
        relaxed_params = params.copy()
        relaxed_params['target_energy'] = 0.3  # Significantly lower energy
        relaxed_params['target_danceability'] = 0.3  # Significantly lower danceability
        relaxed_params['min_acousticness'] = 0  # Relax the minimum acousticness further
        relaxed_params['max_acousticness'] = 1  # Keep maximum as 1 (unrestricted)
        relaxed_params['target_valence'] = 0.5  # Adjust valence if necessary

        return relaxed_params

    def _select_genres(self, emotion_profile) -> str:
        genre_map = {
            'joy': 'pop,dance,disco',
            'sadness': 'blues,jazz,soul',
            'anger': 'rock,metal,punk',
            'love': 'r-n-b,pop,soul',
            'fear': 'ambient,classical,new-age',
            'surprise': 'edm,indie,alternative'
        }
        return genre_map.get(emotion_profile['primary_emotion'], 'pop')

    def _rerank_tracks(self, tracks, emotion_profile) -> List[Dict]:
        for track in tracks:
            track['emotional_fit'] = self._calculate_emotional_fit(track, emotion_profile)
        return sorted(tracks, key=lambda x: x['emotional_fit'], reverse=True)

    def _calculate_emotional_fit(self, track, emotion_profile) -> float:
        valence_diff = abs(track['valence'] - emotion_profile['music_profile']['valence'])
        energy_diff = abs(track['energy'] - emotion_profile['music_profile']['energy'])
        dance_diff = abs(track['danceability'] - emotion_profile['music_profile']['danceability'])

        valence_score = round(1 - valence_diff, 2)
        energy_score = round(1 - energy_diff, 2)
        dance_score = round(1 - dance_diff, 2)

        total_score = round((valence_score + energy_score + dance_score) / 3, 2)
        return max(0.0, min(1.0, total_score))

# === NLP Module ===

try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentiment_analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

class TextAnalyzer:
    def __init__(self):
        self.activity_keywords = {
            'working': [
                'work', 'study', 'code', 'write', 'read', 'research', 'assignment', 'deadline',
                'project', 'homework', 'typing', 'focus', 'brainstorm', 'grind', 'task', 'writing',
                'notebook', 'sheets', 'exam', 'presentation', 'thesis', 'document', 'laptop',
                'zoom call', 'meeting', 'email', 'debugging', 'notes', 'whiteboard', 'lecture',
                'conference', 'slides', 'analysis', 'problem solving', 'studying', 'report'
            ],
            'exercising': [
                'jogging', 'workout', 'gym', 'training', 'lifting', 'run', 'running', 'cycling',
                'cardio', 'sweat', 'treadmill', 'squats', 'pushups', 'abs', 'yoga', 'stretching',
                'warmup', 'cooldown', 'hiit', 'fitness', 'weights', 'deadlift', 'bench press',
                'core', 'fitness class', 'aerobics', 'pilates', 'rowing', 'elliptical', 'boxing',
                'martial arts', 'sprinting', 'strength', 'endurance', 'physical activity'
            ],
            'driving': [
                'drive', 'road trip', 'traffic', 'highway', 'commute', 'steering', 'car ride',
                'long drive', 'cruise', 'engine', 'road', 'riding', 'wheels', 'vehicle',
                'navigation', 'seatbelt', 'driver seat', 'gas station', 'music in car', 'headlights',
                'horn', 'lane', 'petrol', 'route', 'turnpike', 'ride', 'carpool', 'passenger',
                'brakes', 'dashboard', 'car stereo', 'tires', 'mileage'
            ],
            'relaxing': [
                'relax', 'chill', 'unwind', 'rest', 'sleep', 'nap', 'soothing', 'meditate',
                'calm', 'breathe', 'peaceful', 'lazy', 'wind down', 'quiet time', 'decompress',
                'mindfulness', 'spa', 'serene', 'sunset', 'hammock', 'leisure', 'candlelight',
                'comfy', 'blanket', 'me time', 'bubble bath', 'sofa', 'slow vibes', 'tranquility',
                'snuggle', 'self-care', 'soft tunes', 'cozy', 'daydream', 'afternoon nap'
            ],
            'partying': [
                'party', 'celebrate', 'dance', 'club', 'music night', 'drinks', 'shots', 'rave',
                'get together', 'dj', 'birthday bash', 'festival', 'hangout', 'night out',
                'house party', 'celebration', 'booze', 'weekend vibes', 'turn up', 'lit',
                'crowd', 'dance floor', 'bar', 'champagne', 'nightlife', 'concert', 'afterparty',
                'confetti', 'loud music', 'hookah', 'karaoke', 'saturday night', 'party mood', 'dancing'
            ],
            'hindi': [
                'hindi', 'bollywood', 'indian', 'desi', 'filmy', 'bolly', 'hindustani', 'indipop',
                'indian song', 'old hindi', 'retro bollywood', 'arijit', 'shraddha songs',
                'romantic hindi', 'salman song', 'kumar sanu', 'kishore', 'sonu nigam', 'bolly vibe',
                'film songs', 'emotional hindi', 'bolly beat', 'bollywood mix', 'hindi playlist',
                'bollywood love', 'desi song', 'hindipop', 'hindi dance', 'desi chill'
            ],
            'punjabi': [
                'punjabi', 'bhangra', 'desi beats', 'panjabi', 'gurdas', 'sidhu', 'apna', 'punjabi rap',
                'moosewala', 'baari', 'chakna', 'lohri', 'punjab rock', 'jatt', 'ammy virk',
                'romantic punjabi', 'beat punjabi', 'punjabi party', 'balle balle', 'gippy grewal',
                'diljit', 'urban punjabi', 'sufi punjabi', 'punjabi folk', 'bass punjabi',
                'rural beats', 'farmhouse vibe', 'punjabi mix', 'lohri songs', 'punjab tracks',
                'punjab da swag', 'punjabi drop', 'punjabi melody'
            ],
            'tamil': [
                'tamil', 'kollywood', 'ilayaraja', 'vijay', 'arr', 'rajini', 'south india', 'tamizh',
                'tamil beat', 'mass songs', 'tamil melody', 'tamil pop', 'sivakarthikeyan', 'tamil kuthu',
                'surya songs', 'thalaiva', 'vijay anthems', 'kollywood love', 'anirudh', 'tamil rap',
                'gaana', 'folk tamil', 'chennai vibe', 'south mass', 'tamil classic',
                'tamil hitlist', 'tamil banger', 'tamil weekend', 'tamil songs mix', 'arr hits',
                'rajinikanth bgm', 'tamil chill', 'tamil energetic'
            ],
            'telugu': [
                'telugu', 'tollywood', 'ntr', 'mahesh', 'allu', 'south hits', 'andhra', 'telangana',
                'telugu beat', 'devi sri prasad', 'telugu mass', 'telugu romantic', 'manam', 'rrr',
                'telugu mix', 'telugu vibes', 'tollywood dance', 'south pop', 'telegu songs',
                'vijay devarakonda', 'pawankalyan', 'bunny', 'south melodies', 'blockbuster songs',
                'chiranjeevi', 'saaho', 'telugu chill', 'bahubali', 'tollywood bangers', 'telugu rap',
                'andhra songs', 'south bgm', 'telugu club', 'nani songs'
            ]
        }

        self.genre_keywords = {
            'pop': [
                'pop', 'top 40', 'mainstream', 'catchy', 'radio', 'billboard', 'dance pop',
                'synth pop', 'teen pop', 'vocal pop', 'pop hits', 'pop chart', 'pop classics',
                'bubblegum pop', 'electro pop', 'idol pop', 'pop vocal', 'new pop', 'alt pop',
                'girl pop', 'boy band', 'commercial pop', 'radio edit', 'summer pop',
                'melodic pop', 'pop r&b', 'pop groove', 'pop anthem', 'modern pop', 'chill pop',
                'global pop', 'euro pop', 'pop banger', 'pop fusion', 'fresh pop', 'trending pop'
            ],
            'rock': [
                'rock', 'alternative', 'indie', 'punk', 'metal', 'classic rock', 'hard rock',
                'grunge', 'garage rock', 'rock and roll', 'band', 'guitar', 'drums',
                'stadium rock', 'emo rock', 'punk rock', 'folk rock', 'soft rock', 'psych rock',
                'vintage rock', 'alt rock', 'pop rock', 'rock legend', 'metalcore', 'industrial',
                'thrash', 'rock riff', 'rock solo', 'acoustic rock', 'headbang', 'garage jam',
                'mosh pit', 'guitar solo', 'indie garage', 'rock banger'
            ],
            'jazz': [
                'jazz', 'blues', 'saxophone', 'trumpet', 'swing', 'smooth jazz', 'bebop', 'fusion',
                'improv', 'jazz band', 'jazz night', 'cool jazz', 'modal jazz', 'piano jazz',
                'jazz vocal', 'bossa nova', 'nu jazz', 'jazz trio', 'brass section',
                'late night jazz', 'jazz lounge', 'instrumental jazz', 'jazz cafe', 'freestyle',
                'new orleans', 'jazz fusion', 'latin jazz', 'jazz ambient', 'jazz slow',
                'contemporary jazz', 'jazzy beats', 'jazz piano', 'jazz guitar'
            ],
            'electronic': [
                'electronic', 'edm', 'techno', 'house', 'electro', 'trance', 'dubstep', 'synth',
                'beats', 'rave', 'club music', 'drum and bass', 'future bass', 'bass drop',
                'hardstyle', 'melodic edm', 'ambient edm', 'festival set', 'progressive house',
                'deep house', 'lofi electronic', 'dancefloor', 'nightclub music', 'electro swing',
                'edm pop', 'big room', 'electronic mix', 'glitch', 'stepper', 'edm vibes',
                'plur', 'vibe drop', 'trap edm', 'tech house', 'edm classics'
            ],
            'hiphop': [
                'hip hop', 'rap', 'r&b', 'trap', 'drill', 'bars', 'freestyle', 'beats', 'rhymes',
                'old school', 'flow', 'urban', 'gangsta rap', 'east coast', 'west coast',
                'hip hop mix', 'trap soul', 'hiphop club', 'rap anthem', 'underground rap',
                'boom bap', 'battle rap', 'spit fire', 'verse', 'trap beat', 'lyrical rap',
                'street vibe', 'new wave rap', 'rap drill', 'rap melody', 'dirty south',
                '808', 'rap freestyle', 'hiphop banger', 'club rap', 'trap banger'
            ],
            'hindi': self.activity_keywords['hindi'],
            'punjabi': self.activity_keywords['punjabi'],
            'tamil': self.activity_keywords['tamil'],
            'telugu': self.activity_keywords['telugu']
        }

        self.activity_labels = list(self.activity_keywords.keys())
        self.activity_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.activity_embeddings = self.activity_embedding_model.encode(self.activity_labels)

    def _best_activity_from_embedding(self, user_text: str) -> str:
        user_embedding = self.activity_embedding_model.encode([user_text])
        sims = cosine_similarity(user_embedding, self.activity_embeddings)[0]
        best_idx = np.argmax(sims)
        if sims[best_idx] > 0.4:  # You can tune this threshold
            return self.activity_labels[best_idx]
        return "unknown"


    async def analyze_text(self, text: str) -> Dict:
        try:
            language = detect(text)
            if language == 'hi':
                self.genre_keywords['hindi'].extend(['गाना', 'संगीत'])
        except:
            pass

        doc = nlp(text.lower())
        embedding = sentence_model.encode(text)
        sentiment = sentiment_analyzer.polarity_scores(text)
        activities = self._detect_activities(doc)
        genres = self._detect_genres(doc)
        temporal_context = self._detect_temporal_context(doc)
        metaphors = self._detect_metaphors(doc)
        key_phrases = self._extract_key_phrases(doc)

        return {
            "embedding": embedding.tolist(),
            "sentiment": sentiment,
            "activities": activities,
            "genres": genres,
            "temporal_context": temporal_context,
            "metaphors": metaphors,
            "key_phrases": key_phrases,
            "audio_features": self._predict_audio_features(text, sentiment, activities)
        }

    def _detect_activities(self, doc) -> List[str]:
        matched = [activity for activity, keywords in self.activity_keywords.items()
            if any(token.text in keywords or token.lemma_ in keywords for token in doc)]
        if not matched:
            fallback_text = doc.text if hasattr(doc, 'text') else " ".join([t.text for t in doc])
            fallback = self._best_activity_from_embedding(fallback_text)
            if fallback != "unknown":
                matched.append(fallback)
    
        return matched

    def _detect_genres(self, doc) -> List[str]:
        return [genre for genre, keywords in self.genre_keywords.items()
                if any(token.text in keywords for token in doc)]

    def _detect_temporal_context(self, doc) -> Optional[str]:
        for token in doc:
            if token.text in ['morning', 'afternoon', 'evening', 'night', 'dawn']:
                return token.text
        return None

    def _detect_metaphors(self, doc) -> List[str]:
        return [doc[max(token.i - 2, 0):min(token.i + 3, len(doc))].text
                for token in doc if token.text.lower() in ['like', 'as']]

    def _extract_key_phrases(self, doc) -> List[str]:
        return [
            " ".join([token.text for token in chunk if token.pos_ != "DET"])
            for chunk in doc.noun_chunks if len(chunk.text.split()) > 1
        ]

    def _predict_audio_features(self, text: str, sentiment: Dict, activities: List[str]) -> Dict:
        features = {
            'valence': max(0, min(1, 0.5 + sentiment['compound'] * 0.5)),
            'energy': 0.5,
            'danceability': 0.5,
            'tempo': 100,
            'acousticness': 0.5
        }

        if 'exercising' in activities:
            features.update({'energy': 0.9, 'tempo': 130, 'danceability': 0.8})
        elif 'relaxing' in activities:
            features.update({'energy': 0.3, 'tempo': 80, 'acousticness': 0.8})
        elif 'partying' in activities:
            features.update({'energy': 0.95, 'danceability': 0.95, 'tempo': 125})

        return features

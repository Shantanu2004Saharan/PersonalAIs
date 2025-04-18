from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_user_conversation_history, update_user_preferences
from nlp_module import TextAnalyzer
from spotify import SpotifyClient
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer, util
import faiss
import json

from models import TrackRecommendation 

from sentence_transformers import SentenceTransformer, util

# Load vector data and metadata

VECTOR_PATH = r"C:\Users\hello\music_recommender\backend\data\track_embeddings.npy"
INDEX_PATH = r"C:\Users\hello\music_recommender\backend\data\faiss_index.bin"
METADATA_PATH = r"C:\Users\hello\music_recommender\backend\data\track_metadata.json"

def load_track_embeddings():
    # Load precomputed numpy embeddings
    embeddings = np.load(VECTOR_PATH)

    # Load faiss index
    index = faiss.read_index(INDEX_PATH)

    return embeddings, index

def load_track_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings, index = load_track_embeddings()
metadata = load_track_metadata()

async def generate_dynamic_recommendations(user_message: str) -> Tuple[str, List[TrackRecommendation]]:
    query_embedding = model.encode(user_message)
    query_embedding = np.array([query_embedding]).astype("float32")

    D, I = index.search(query_embedding, k=12)  # Top 12 matches

    matched_tracks = []
    for idx in I[0]:
        track = metadata[idx]
        matched_tracks.append(TrackRecommendation(
            id=track["id"],
            name=track["name"],
            artists=track["artists"],
            preview_url=track.get("preview_url"),
            external_url=track["external_url"],
            features=track["features"]
        ))

    # Naive mood detection (can improve later)
    mood = "chill" if "relax" in user_message.lower() else "energetic"

    return mood, matched_tracks # adjust this if your TrackRecommendation class is elsewhere

def get_recommendations_with_filters(
    spotify: SpotifyClient, 
    tracks: List[dict], 
    mood_filters: dict
) -> List[dict]:
    """
    Filter a list of tracks based on mood filters like valence, energy, etc.
    """
    def matches(track, filters):
        for key, value in filters.items():
            if key not in track['features']:
                return False
            # Allowing a small margin of ±0.1
            if abs(track['features'][key] - value) > 0.1:
                return False
        return True

    return [track for track in tracks if matches(track, mood_filters)]


SPOTIFY_CLIENT_ID = "5b42d00bcf0a42de83f9bd8855c5d629"
SPOTIFY_CLIENT_SECRET = "246b8f0425dc464ba1758a1e2ab4fa72"
DEBUG_MODE = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicRecommender:
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.spotify = SpotifyClient()
        self.conversation_memory = {}
    
    async def recommend_songs(
        self, 
        db: AsyncSession,
        user_id: str, 
        user_input: str
    ) -> Dict[str, Any]:
        """
        Generate personalized song recommendations with context
        
        Args:
            db: Async database session
            user_id: Unique user identifier
            user_input: Current user message/input
            
        Returns:
            Dictionary containing:
            - analysis: Mood/activity/genre analysis
            - recommendations: Sorted list of tracks
            - explanation: Human-readable justification
        """
        try:
            # Get conversation history for context
            history = await get_user_conversation_history(db, user_id)
            context = " ".join([msg.message for msg in history[-5:]])  # Last 5 messages
            
            # Analyze text with NLP
            full_text = f"{context} {user_input}".strip()
            analysis = await self.text_analyzer.analyze_text(full_text)
            logger.info(f"Analysis completed for user {user_id}: {analysis}")
            
            # Prepare dynamic audio features for filtering
            audio_features = {
                f"target_{k}": v for k, v in analysis['audio_features'].items() if v is not None
            }

            spotify_query = {
                'seed_genres': analysis['genres'][:2],
                'limit': 100,
                **audio_features
            }

            # Get recommendations from Spotify
            recommendations = await self.spotify.get_recommendations(spotify_query)
            
            if not recommendations:
                raise ValueError("No recommendations found from Spotify API")
            
            # Score, filter and sort recommendations
            scored_recommendations = self._score_recommendations(
                recommendations, 
                analysis
            )
            top_recommendations = scored_recommendations[:10]  # Return top 10
            
            # Update user preferences
            await self._update_user_profile(db, user_id, analysis)
            
            return {
                "analysis": {
                    "mood": self._describe_mood(analysis['sentiment']),
                    "activities": analysis['activities'],
                    "genres": analysis['genres'],
                    "key_phrases": analysis.get('key_phrases', [])
                },
                "recommendations": top_recommendations,
                "explanation": self._generate_explanation(analysis, top_recommendations[0])
            }
            
        except Exception as e:
            logger.error(f"Recommendation failed for user {user_id}: {str(e)}", exc_info=True)
            raise RecommendationError(f"Recommendation failed: {str(e)}") from e

    def _score_recommendations(
        self, 
        recommendations: List[Dict], 
        analysis: Dict
    ) -> List[Dict]:
        """
        Score recommendations based on multiple factors:
        - Audio feature matching (50%)
        - Genre relevance (30%) 
        - Activity matching (20%)
        """
        scored = []

        for track in recommendations:
            # Dynamic feature similarity
            feature_keys = ['valence', 'energy', 'danceability']  # Extend as needed

            track_vec = []
            target_vec = []

            for key in feature_keys:
                track_val = track['features'].get(key)
                target_val = analysis['audio_features'].get(key)
                if track_val is not None and target_val is not None:
                    track_vec.append(track_val)
                    target_vec.append(target_val)

            if track_vec and target_vec:
                track_features = np.array(track_vec).reshape(1, -1)
                target_features = np.array(target_vec).reshape(1, -1)
                feature_score = 0.5 * cosine_similarity(target_features, track_features)[0][0]
            else:
                feature_score = 0

            # Genre matching (Jaccard similarity)
            track_genres = set(track.get('genres', []))
            target_genres = set(analysis['genres'])
            genre_score = 0.3 * len(track_genres & target_genres) / len(track_genres | target_genres) if track_genres else 0

            # Activity matching
            activity_score = 0.2 if any(
                activity.lower() in [a.lower() for a in track.get('activity_tags', [])]
                for activity in analysis['activities']
            ) else 0

            total_score = feature_score + genre_score + activity_score

            scored.append({
                **track,
                "match_score": round(total_score, 4),
                "feature_breakdown": {
                    "audio_similarity": round(feature_score, 4),
                    "genre_match": round(genre_score, 4),
                    "activity_match": round(activity_score, 4)
                }
            })

        # Sort by score and remove very low matches
        return sorted(
            [t for t in scored if t['match_score'] > 0.2],  # Lower threshold
            key=lambda x: x['match_score'],
            reverse=True
        )[:12]

    async def _update_user_profile(
        self,
        db: AsyncSession,
        user_id: str,
        analysis: Dict
    ) -> None:
        """Update user preferences in database based on current interaction"""
        try:
            preferences = {
                'last_updated': datetime.now().isoformat(),
                'genres': analysis['genres'],
                'activities': analysis['activities'],
                'mood_profile': {
                    'valence': analysis['audio_features']['valence'],
                    'energy': analysis['audio_features']['energy'],
                    'sentiment': analysis['sentiment']
                },
                'recent_keywords': analysis.get('key_phrases', [])
            }
            
            await update_user_preferences(db, user_id, preferences)
            logger.info(f"Updated preferences for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update preferences for user {user_id}: {str(e)}")
            raise

    def _describe_mood(self, sentiment: Dict) -> str:
        """Convert sentiment analysis to descriptive mood"""
        compound = sentiment['compound']
        if compound > 0.6:
            return "upbeat and positive"
        elif compound > 0.2:
            return "positive"
        elif compound > -0.2:
            return "neutral"
        elif compound > -0.6:
            return "somber"
        else:
            return "melancholic"

    def _generate_explanation(
        self, 
        analysis: Dict, 
        top_track: Dict
    ) -> str:
        """Generate natural language explanation for recommendations"""
        explanations = []

        # Mood context
        mood = self._describe_mood(analysis['sentiment'])
        explanations.append(f"Based on your {mood} mood,")

        # Activity context
        if analysis['activities']:
            activities = ", ".join(analysis['activities'][:2])
            explanations.append(f"perfect for {activities},")

        # Track justification
        explanations.append(
            f"we recommend '{top_track['name']}' by {', '.join(top_track['artists'])} "
            f"with a {top_track['match_score']*100:.0f}% match "
            f"(similar mood and {top_track['features']['energy']*100:.0f}% energy)."
        )

        # Genre context
        if analysis['genres']:
            genres = ", ".join(analysis['genres'][:2])
            explanations.append(f"Enjoy these {genres} vibes!")

        return " ".join(explanations)


class RecommendationError(Exception):
    """Custom exception for recommendation failures"""
    pass


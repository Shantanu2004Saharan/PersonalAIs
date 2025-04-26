from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_user_conversation_history, update_user_preferences
from model_matcher_nlp import TextAnalyzer
from spotify_client import SpotifyClient
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer, util
import faiss
import json
from model_matcher_nlp import TrackRecommendation 
from spotify_client import SpotifyClient, generate_dynamic_recommendations


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

    seen = set()
    matched_tracks = []

    for idx in I[0]:
        track = metadata[idx]
        if track["id"] not in seen:
            seen.add(track["id"])
            matched_tracks.append(track)

    spotify = SpotifyClient()
    track_ids = [track["id"] for track in matched_tracks]

    # 🆕 Improved fetching: batch call
    audio_features_list = spotify.get_audio_features_batch(track_ids) or []

    features_map = {}
    if audio_features_list:
        for track_id, features in zip(track_ids, audio_features_list):
            if features:
                features_map[track_id] = features

    final_tracks = []
    for track in matched_tracks:
        features = features_map.get(track["id"], {})
        final_tracks.append(TrackRecommendation(
            id=track["id"],
            name=track["name"],
            artists=track["artists"],
            preview_url=track.get("preview_url") or f"https://p.scdn.co/mp3-preview/{track['id']}",
            external_url=track.get("external_url") or track.get("external_urls", {}).get("spotify") or f"https://open.spotify.com/track/{track['id']}",
            features=features
            ))


    mood = "chill" if "relax" in user_message.lower() else "energetic"

    return mood, final_tracks


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
        try:
            history = await get_user_conversation_history(db, user_id)
            context = " ".join([msg.message for msg in history[-5:]])
            full_text = f"{context} {user_input}".strip()

            analysis = await self.text_analyzer.analyze_text(full_text)
            logger.info(f"Analysis completed for user {user_id}: {analysis}")

            spotify_query = user_input
            recommendations = self.spotify.search_tracks(spotify_query, limit=30)
            if not recommendations:
                raise ValueError("No recommendations found from Spotify API")

            track_ids = [track['id'] for track in recommendations if 'id' in track]

            try:
                audio_features_list = self.spotify.get_audio_features_batch(track_ids)
                if not audio_features_list:
                    logger.warning("No audio features returned. Using default backup tracks.")
                    audio_features_list = [{} for _ in track_ids]
            except Exception as e:
                logger.error(f"Spotify audio feature fetch failed: {e}")
                audio_features_list = [{} for _ in track_ids]

            features_map = {
                track_id: af for track_id, af in zip(track_ids, audio_features_list) if af
            }

            for track in recommendations:
                track_id = track.get('id')
                track['features'] = features_map.get(track_id, {})

            scored_recommendations = self._score_recommendations(recommendations, analysis)
            top_recommendations = scored_recommendations[:10]

            MIN_RECOMMENDATIONS = 10
            if len(top_recommendations) < MIN_RECOMMENDATIONS:
                logger.warning(f"Only {len(top_recommendations)} songs found. Adding fallback songs...")

                existing_ids = {t["id"] for t in top_recommendations}
                extra_needed = MIN_RECOMMENDATIONS - len(top_recommendations)

                for track in scored_recommendations[len(top_recommendations):]:
                    if track["id"] not in existing_ids:
                        top_recommendations.append(track)
                        existing_ids.add(track["id"])
                    if len(top_recommendations) >= MIN_RECOMMENDATIONS:
                        break

                if not top_recommendations:
                    logger.warning("All scoring failed. Returning raw Spotify search results.")
                    for track in recommendations[:MIN_RECOMMENDATIONS]:
                        track.setdefault("features", {})
                        track["match_score"] = 0.0
                        track["feature_breakdown"] = {}
                        top_recommendations.append(track)

            await self._update_user_profile(db, user_id, analysis)

            return {
                "analysis": {
                    "mood": self._describe_mood(analysis['sentiment']),
                    "activities": analysis['activities'],
                    "genres": analysis['genres'],
                    "key_phrases": analysis.get('key_phrases', [])
                },
                "recommendations": top_recommendations,
                "explanation": self._generate_explanation(analysis, top_recommendations[0]) if top_recommendations else "No recommendations found."
            }

        except Exception as e:
            logger.error(f"Recommendation failed for user {user_id}: {str(e)}", exc_info=True)
            raise RecommendationError(f"Recommendation failed: {str(e)}") from e

    async def _update_user_profile(self, db, user_id, analysis):
        logger.info(f"Updating user profile for user {user_id}... [SKIPPED]")

    def _describe_mood(self, sentiment: dict) -> str:
        compound = sentiment.get("compound", 0)
        if compound >= 0.5:
            return "upbeat and energetic"
        elif compound >= 0:
            return "positive and cheerful"
        else:
            return "calm or reflective"

    def _generate_explanation(self, analysis: dict, top_track: dict) -> str:
        return f"Because you mentioned activities like {', '.join(analysis.get('activities', []))}, this song matches your energy and vibe!"

    def _score_recommendations(
        self, 
        recommendations: List[Dict], 
        analysis: Dict
    ) -> List[Dict]:
        scored = []
        for track in recommendations:
            feature_keys = ['valence', 'energy', 'danceability']
            track_vec = []
            target_vec = []

            for key in feature_keys:
                track_val = track['features'].get(key)
                target_val = analysis['audio_features'].get(key)
                if track_val is not None and target_val is not None:
                    track_vec.append(track_val)
                    target_vec.append(target_val)

            feature_score = 0.0
            if track_vec and target_vec:
                track_features = np.array(track_vec).reshape(1, -1)
                target_features = np.array(target_vec).reshape(1, -1)
                feature_score = 0.5 * cosine_similarity(target_features, track_features)[0][0]

            track_genres = set(track.get('genres', []))
            target_genres = set(analysis['genres'])
            genre_score = 0.3 * len(track_genres & target_genres) / len(track_genres | target_genres) if track_genres else 0

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

        return sorted(
            [t for t in scored if t['match_score'] > 0.2],
            key=lambda x: x['match_score'],
            reverse=True
        )[:12]

class RecommendationError(Exception):
    """Custom exception for recommendation failures"""
    pass


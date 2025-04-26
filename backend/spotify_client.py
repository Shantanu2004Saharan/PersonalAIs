import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import List, Dict, Optional
import logging
from sentence_transformers import SentenceTransformer
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss
import json
import os
from model_matcher_nlp import TrackRecommendation
import json

# ===================== Logging Setup =====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== Spotify Client =====================
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials

class SpotifyClient:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        token_info: Optional[Dict] = None
    ):
        self.client_id = client_id or "736bb144677e448dad56d2fe2ab70cd0"
        self.client_secret = client_secret or "d7beffe6e8d740deb7e1ddd9a111c88f"
        self.redirect_uri = redirect_uri or "http://127.0.0.1:8000/callback"

        if token_info:
            # Initialize with token from OAuth callback
            self.sp = spotipy.Spotify(auth=token_info["access_token"])
            try:
                self.user_id = self.sp.current_user()["id"]
            except Exception as e:
                logger.error(f"Failed to get current user with token: {e}")
                self.user_id = None
        elif redirect_uri:
            self.auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope="user-library-read user-top-read playlist-modify-public"
            )
            self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
            try:
                self.user_id = self.sp.current_user()["id"]
            except Exception:
                self.user_id = None
        else:
            self.auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
            self.user_id = None

        logger.info("✅ Spotify Client initialized successfully.")

    def refresh_client(self):
        if isinstance(self.auth_manager, SpotifyClientCredentials):
            self.auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
        logger.info("🔄 Spotify Client session refreshed.")

    def search_tracks(self, query: str, limit: int = 20) -> List[Dict]:
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            return results['tracks']['items']
        except Exception as e:
            logger.error(f"Error searching tracks: {str(e)}")
            self.refresh_client()
            return []

    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        try:
            features = self.sp.audio_features([track_id])[0]
            if features:
                return self.clean_features(features)
            logger.warning(f"No audio features found for track: {track_id}")
            return None
        except Exception as e:
            logger.warning(f"Couldn't fetch audio features for track {track_id}: {e}")
            self.refresh_client()
            return None

    def get_audio_features_batch(self, track_ids: List[str]) -> List[Optional[Dict]]:
        try:
            track_ids = track_ids[:100]
            features_list = self.sp.audio_features(track_ids)
            if not features_list:
                logger.warning("No features found for the given batch.")
                return []
            return [self.clean_features(f) for f in features_list if f]
        except Exception as e:
            logger.error(f"Error fetching batch audio features: {e}")
            self.refresh_client()
            return []

    def get_recommendations_by_genre(self, genre: str, limit: int = 10) -> List[Dict]:
        try:
            recs = self.sp.recommendations(seed_genres=[genre], limit=limit)
            return recs['tracks']
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            self.refresh_client()
            return []

    def create_playlist(self, name: str, description: str = "", public: bool = True) -> Dict:
        if not self.user_id:
            raise RuntimeError("Cannot create playlist without a valid OAuth user.")

        playlist = self.sp.user_playlist_create(
            user=self.user_id,
            name=name,
            public=public,
            description=description
        )
        return playlist

    def add_to_playlist(self, playlist_id: str, track_ids: List[str]):
        if not track_ids:
            logger.warning("No tracks provided to add to playlist.")
            return
        self.sp.playlist_add_items(playlist_id, track_ids)

    @staticmethod
    def clean_features(features: Dict) -> Dict:
        return {
            "danceability": features.get("danceability", 0.0),
            "energy": features.get("energy", 0.0),
            "valence": features.get("valence", 0.0),
            "tempo": features.get("tempo", 0.0),
            "acousticness": features.get("acousticness", 0.0),
            "instrumentalness": features.get("instrumentalness", 0.0),
            "liveness": features.get("liveness", 0.0),
            "speechiness": features.get("speechiness", 0.0),
            "duration_ms": features.get("duration_ms", 0)
        }


class EmbeddingUtils:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)


# ===================== Music Recommender =====================
class MusicRecommender:
    def __init__(self):
        self.spotify = SpotifyClient()
        self.embedder = EmbeddingUtils()

    def get_songs(self, query: str) -> List[Dict]:
        return self.spotify.search_tracks(query)

    def recommend(self, query: str, mood: Dict[str, float], top_k: int = 10) -> List[Dict]:
        songs = self.get_songs(query)
        query_embed = self.embedder.embed_text(query)
        ranked = []

        for song in songs:
            text = f"{song['name']} {' '.join(a['name'] for a in song['artists'])}"
            embed = self.embedder.embed_text(text)
            score = cosine_similarity([query_embed], [embed])[0][0]

            features = self.spotify.get_audio_features(song['id']) or {}
            song['score'] = score
            song['features'] = features

            if features:
                match = all(
                    mood[k] * 0.8 <= features.get(k, 0) <= mood[k] * 1.2
                    for k in mood if k in features
                )
                if match:
                    ranked.append((score, song))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in ranked[:top_k]]

# ===================== Dynamic Recommendation (FAISS-based) =====================
VECTOR_PATH = r"C:\\Users\\hello\\music_recommender\\backend\\data\\track_embeddings.npy"
INDEX_PATH = r"C:\\Users\\hello\\music_recommender\\backend\\data\\faiss_index.bin"
METADATA_PATH = r"C:\\Users\\hello\\music_recommender\\backend\\data\\track_metadata.json"

_model = SentenceTransformer("all-MiniLM-L6-v2")
_embeddings = np.load(VECTOR_PATH)
_index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    _metadata = json.load(f)

async def generate_dynamic_recommendations(user_message: str) -> tuple[str, list[TrackRecommendation]]:
    query_embedding = _model.encode(user_message)
    query_embedding = np.array([query_embedding]).astype("float32")

    D, I = _index.search(query_embedding, k=12)

    seen = set()
    matched_tracks = []

    for idx in I[0]:
        track = _metadata[idx]
        if track["id"] not in seen:
            seen.add(track["id"])
            matched_tracks.append(track)

    spotify = SpotifyClient()
    track_ids = [track["id"] for track in matched_tracks]
    audio_features_list = spotify.get_audio_features_batch(track_ids) or []

    features_map = {
        tid: af for tid, af in zip(track_ids, audio_features_list) if af
    }

    final_tracks = []
    for track in matched_tracks:
        features = features_map.get(track["id"], {})
        final_tracks.append(TrackRecommendation(
            id=track["id"],
            name=track["name"],
            artists=track["artists"],
            preview_url=track.get("preview_url") or f"https://p.scdn.co/mp3-preview/{track['id']}",
            external_url=track.get("external_url") or f"https://open.spotify.com/track/{track['id']}",
            features=features
        ))

    mood = "chill" if "relax" in user_message.lower() else "energetic"
    return mood, final_tracks

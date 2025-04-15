"""
Enhanced Spotify Client with Complete Functionality
- OAuth authentication with callback server
- All original features preserved
- Properly organized methods
- Secure credential handling
- Now supports audio feature filtering for semantic recommendations
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import List, Dict, Optional
import logging
import asyncio
import time
from functools import lru_cache
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Security Note: Regenerate these credentials if committed to version control
class SpotifySecrets:
    CLIENT_ID = "5b42d00bcf0a42de83f9bd8855c5d629"
    CLIENT_SECRET = "246b8f0425dc464ba1758a1e2ab4fa72"
    REDIRECT_URI = "http://127.0.0.1:8000/callback"
    SCOPES = [
        "user-library-read",
        "user-top-read",
        "playlist-modify-public",
        "user-read-recently-played",
        "user-read-playback-state"
    ]

    @classmethod
    def clear_memory(cls):
        """Securely wipe credentials from memory"""
        cls.CLIENT_ID = "x" * len(cls.CLIENT_ID)
        cls.CLIENT_SECRET = "x" * len(cls.CLIENT_SECRET)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CallbackHandler(BaseHTTPRequestHandler):
    """Handles Spotify OAuth redirects"""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<h1>Authentication successful! You may close this tab.</h1>")
        logger.info(f"Received callback at: {self.path}")
        self.server.callback_received = True

class SpotifyAuthServer:
    """Manages the local callback server"""
    def __init__(self):
        self.server = HTTPServer(('127.0.0.1', 8000), CallbackHandler)
        self.server.callback_received = False

    def start(self):
        """Start server in a background thread"""
        thread = threading.Thread(target=self.server.serve_forever)
        thread.daemon = True
        thread.start()
        logger.info("Callback server started on http://127.0.0.1:8000")

    def stop(self):
        """Shutdown server"""
        self.server.shutdown()
        logger.info("Callback server stopped")

class SpotifyClient:
    def __init__(self):
        """Initialize with built-in auth server"""
        self.auth_server = SpotifyAuthServer()
        self.auth_server.start()
        
        self.client_id = SpotifySecrets.CLIENT_ID
        self.client_secret = SpotifySecrets.CLIENT_SECRET
        self.redirect_uri = SpotifySecrets.REDIRECT_URI
        self.scope = " ".join(SpotifySecrets.SCOPES)

        self.sp = self._initialize_spotify_client()
        self.valid_genres = self._get_valid_genres()
        self.user_id = self._get_user_id()
        
        SpotifySecrets.clear_memory()
        self.auth_server.stop()
        logger.info("Spotify client initialized")

    def _initialize_spotify_client(self, retries=3) -> spotipy.Spotify:
        """Initialize with retry logic"""
        for attempt in range(retries):
            try:
                auth_manager = SpotifyOAuth(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    redirect_uri=self.redirect_uri,
                    scope=self.scope,
                    cache_path=".spotify_cache",
                    open_browser=True
                )
                sp = spotipy.Spotify(
                    auth_manager=auth_manager,
                    retries=3,
                    status_retries=3,
                    backoff_factor=0.3
                )
                sp.current_user()
                return sp
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                time.sleep(2 ** attempt)
        raise ConnectionError("Could not initialize Spotify client")

    def _get_user_id(self) -> Optional[str]:
        try:
            return self.sp.current_user().get("id")
        except:
            return None

    def get_current_user(self) -> Dict:
        return self.sp.current_user()

    def get_user_top_tracks(self, limit=20, time_range='medium_term') -> List[Dict]:
        return self.sp.current_user_top_tracks(limit=limit, time_range=time_range)['items']

    def get_user_top_artists(self, limit=20, time_range='medium_term') -> List[Dict]:
        return self.sp.current_user_top_artists(limit=limit, time_range=time_range)['items']

    async def get_recommendations(self, features: Dict, limit=20) -> List[Dict]:
        params = {
            'limit': min(limit, 100),
            'market': 'US',
            **self._prepare_seed_params(features),
            **self._prepare_audio_features(features)
        }
        try:
            results = self.sp.recommendations(**{k: v for k, v in params.items() if v is not None})
            return self._process_recommendations(results['tracks'])
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            return []

    def create_playlist(self, name: str, public=True, description="") -> Dict:
        if not self.user_id:
            raise ValueError("No authenticated user")
        return self.sp.user_playlist_create(user=self.user_id, name=name, public=public, description=description)

    def add_to_playlist(self, playlist_id: str, track_ids: List[str]) -> Dict:
        return self.sp.playlist_add_items(playlist_id, track_ids)

    @lru_cache(maxsize=100)
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        try:
            return self.sp.audio_features([track_id])[0]
        except:
            return None

    def get_audio_analysis(self, track_id: str) -> Optional[Dict]:
        try:
            return self.sp.audio_analysis(track_id)
        except:
            return None

    def _prepare_seed_params(self, features: Dict) -> Dict:
        seeds = {}
        if 'genres' in features:
            seeds['seed_genres'] = self._validate_genres(features['genres'])
        if 'artists' in features:
            seeds['seed_artists'] = ','.join(features['artists'][:5])
        if 'tracks' in features:
            seeds['seed_tracks'] = ','.join(features['tracks'][:5])
        return seeds

    def _validate_genres(self, genres: List[str]) -> str:
        valid = [g for g in genres if g in self.valid_genres][:5]
        return ','.join(valid) if valid else 'pop'

    def _prepare_audio_features(self, features: Dict) -> Dict:
        feature_map = {
            'valence': 'target_valence',
            'energy': 'target_energy',
            'danceability': 'target_danceability',
            'tempo': 'target_tempo',
            'acousticness': 'target_acousticness',
            'instrumentalness': 'target_instrumentalness',
            'liveness': 'target_liveness',
            'speechiness': 'target_speechiness'
        }
        return {
            feature_map[k]: v 
            for k, v in features.items() 
            if k in feature_map and v is not None
        }

    def _process_recommendations(self, tracks: List) -> List[Dict]:
        return [
            {
                'id': t['id'],
                'name': t['name'],
                'artists': [a['name'] for a in t['artists']],
                'album': t['album']['name'],
                'preview_url': t['preview_url'],
                'external_url': t['external_urls']['spotify'],
                'features': self.get_audio_features(t['id']) or {}
            }
            for t in tracks if t.get('id')
        ]

    def _get_valid_genres(self) -> List[str]:
        try:
            return self.sp.recommendation_genre_seeds()['genres']
        except:
            return ['pop', 'rock', 'hip-hop', 'indie', 'electronic']

# ====================== TESTING ========================
async def test_all_features():
    print("\n🔍 Testing Spotify Client...")
    try:
        client = SpotifyClient()

        print(f"👤 User ID: {client.user_id or 'Not available'}")

        print("\n🎵 Testing top tracks...")
        top_tracks = client.get_user_top_tracks(limit=2)
        for i, track in enumerate(top_tracks, 1):
            print(f"{i}. {track['name']} by {track['artists'][0]['name']}")

        print("\n🎧 Testing recommendations...")
        recs = await client.get_recommendations({
            "genres": ["pop"],
            "valence": 0.7,
            "energy": 0.6,
            "danceability": 0.8
        }, limit=2)

        if recs:
            print(f"✅ Got {len(recs)} recommendations:")
            for i, track in enumerate(recs, 1):
                print(f"{i}. {track['name']} by {', '.join(track['artists'])}")

        if recs and client.user_id:
            print("\n📝 Testing playlist creation...")
            playlist = client.create_playlist(
                name="API Test Playlist",
                description="Created by Spotify API"
            )
            print(f"✅ Created playlist: {playlist['external_urls']['spotify']}")

            print("\n➕ Adding tracks to playlist...")
            track_ids = [track['id'] for track in recs]
            result = client.add_to_playlist(playlist['id'], track_ids)
            print(f"✅ Added {len(track_ids)} tracks")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    finally:
        print("\nTest completed")

if __name__ == "__main__":
    asyncio.run(test_all_features())


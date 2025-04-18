"""
Enhanced Spotify Client with Complete Functionality
- Direct credential initialization (no .env required)
- All original features preserved
- Improved error handling
- Better type hints
- Optimized performance
- Added missing functions for imports
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import List, Dict, Optional, Union, Any
import logging
import asyncio
import time
from functools import lru_cache
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import webbrowser


# ========================== LOGGING ==========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== AUTH CALLBACK SERVER ====================
class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<h1>Authentication successful! You may close this tab.</h1>")
        logger.info(f"Received callback at: {self.path}")
        self.server.callback_received = True

class SpotifyAuthServer:
    def __init__(self, port: int = 8000):
        self.server = HTTPServer(('127.0.0.1', port), CallbackHandler)
        self.server.callback_received = False

    def start(self):
        thread = threading.Thread(target=self.server.serve_forever)
        thread.daemon = True
        thread.start()
        logger.info(f"Callback server started on http://127.0.0.1:{self.server.server_port}")

    def stop(self):
        self.server.shutdown()
        logger.info("Callback server stopped")

# ========================= MAIN CLIENT ==========================
class SpotifyClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "http://127.0.0.1:8000/callback",
        scopes: List[str] = None
    ):
        """
        Initialize Spotify client with direct credentials
        
        Args:
            client_id: Your Spotify app client ID
            client_secret: Your Spotify app client secret
            redirect_uri: Your Spotify app redirect URI (default: http://127.0.0.1:8000/callback)
            scopes: List of Spotify API scopes (default: common music recommendation scopes)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes or [
            "user-library-read",
            "user-top-read",
            "playlist-modify-public",
            "user-read-recently-played",
            "user-read-playback-state"
        ]

        # Parse port from redirect URI
        try:
            port = int(self.redirect_uri.split(':')[-1].split('/')[0])
        except:
            port = 8000

        self.auth_server = SpotifyAuthServer(port)
        self.auth_server.start()

        self.sp = self._initialize_spotify_client()
        self.valid_genres = self._get_valid_genres()
        self.user_id = self._get_user_id()

        self.auth_server.stop()
        logger.info("Spotify client initialized")

    def _initialize_spotify_client(self, retries: int = 3) -> spotipy.Spotify:
        """Initialize and authenticate Spotify client"""
        auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=" ".join(self.scopes),
            cache_path=".spotify_cache",
            open_browser=True
    )
    
    # Try to get cached token first
        token_info = auth_manager.get_cached_token()
        if not token_info:
        # If no cached token, do the auth flow
            auth_url = auth_manager.get_authorize_url()
            print(f"Please authorize here: {auth_url}")
            webbrowser.open(auth_url)
        
        # Wait for user to paste the redirect URL
            redirect_response = input("Paste the redirect URL here: ")
            code = auth_manager.parse_response_code(redirect_response)
            token_info = auth_manager.get_access_token(code)
    
        return spotipy.Spotify(auth_manager=auth_manager)

    def _get_user_id(self) -> Optional[str]:
        """Get current authenticated user ID"""
        try:
            return self.sp.current_user().get("id")
        except Exception as e:
            logger.warning(f"Could not get user ID: {str(e)}")
            return None

    def get_current_user(self) -> Dict:
        """Get current authenticated user profile"""
        return self.sp.current_user()

    def get_user_top_tracks(self, limit: int = 20, time_range: str = 'medium_term') -> List[Dict]:
        """Get user's top tracks"""
        return self.sp.current_user_top_tracks(limit=limit, time_range=time_range)['items']

    def get_user_top_artists(self, limit: int = 20, time_range: str = 'medium_term') -> List[Dict]:
        """Get user's top artists"""
        return self.sp.current_user_top_artists(limit=limit, time_range=time_range)['items']

    async def get_recommendations(self, features: Dict, limit: int = 20) -> List[Dict]:
        """Get track recommendations based on audio features"""
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

    def create_playlist(self, name: str, public: bool = True, description: str = "") -> Dict:
        """Create a new playlist"""
        if not self.user_id:
            raise ValueError("No authenticated user")
        return self.sp.user_playlist_create(
            user=self.user_id,
            name=name,
            public=public,
            description=description
        )

    def add_to_playlist(self, playlist_id: str, track_ids: List[str]) -> Dict:
        """Add tracks to an existing playlist"""
        return self.sp.playlist_add_items(playlist_id, track_ids)

    @lru_cache(maxsize=100)
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """Get audio features for a track (cached)"""
        try:
            return self.sp.audio_features([track_id])[0]
        except Exception as e:
            logger.warning(f"Could not get audio features: {str(e)}")
            return None

    def get_audio_analysis(self, track_id: str) -> Optional[Dict]:
        """Get detailed audio analysis for a track"""
        try:
            return self.sp.audio_analysis(track_id)
        except Exception as e:
            logger.warning(f"Could not get audio analysis: {str(e)}")
            return None

    def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for tracks on Spotify"""
        try:
            results = self.sp.search(q=query, type="track", limit=limit)
            tracks = results.get('tracks', {}).get('items', [])
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
                for t in tracks
            ]
        except Exception as e:
            logger.error(f"Track search failed: {str(e)}")
            return []

    # ======== Functions needed for imports from other modules ========
    def search_spotify_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for tracks on Spotify (alias for search_tracks)"""
        return self.search_tracks(query, limit)

    def get_track_features(self, track_id: str) -> Dict[str, Any]:
        """Get audio features for a track (alias for get_audio_features)"""
        features = self.get_audio_features(track_id)
        return features or {}

    def get_track_info(self, track_id: str) -> Dict[str, Any]:
        """Get detailed track information"""
        try:
            track = self.sp.track(track_id)
            return {
                'id': track['id'],
                'name': track['name'],
                'artists': [a['name'] for a in track['artists']],
                'album': track['album']['name'],
                'preview_url': track['preview_url'],
                'external_url': track['external_urls']['spotify'],
                'features': self.get_audio_features(track_id) or {}
            }
        except Exception as e:
            logger.error(f"Could not get track info: {str(e)}")
            return {}

    # ======== Helper Methods ========
    def _prepare_seed_params(self, features: Dict) -> Dict:
        """Prepare seed parameters for recommendations"""
        seeds = {}
        if 'genres' in features:
            seeds['seed_genres'] = self._validate_genres(features['genres'])
        if 'artists' in features:
            seeds['seed_artists'] = ','.join(features['artists'][:5])
        if 'tracks' in features:
            seeds['seed_tracks'] = ','.join(features['tracks'][:5])
        return seeds

    def _validate_genres(self, genres: List[str]) -> str:
        """Validate and format genres for recommendations"""
        valid = [g for g in genres if g in self.valid_genres][:5]
        return ','.join(valid) if valid else 'pop'

    def _prepare_audio_features(self, features: Dict) -> Dict:
        """Prepare audio feature parameters for recommendations"""
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
        """Process raw recommendation results into standardized format"""
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
        """Get list of valid genre seeds from Spotify"""
        try:
            return self.sp.recommendation_genre_seeds()['genres']
        except Exception as e:
            logger.warning(f"Could not get genre seeds: {str(e)}")
            return ['pop', 'rock', 'hip-hop', 'indie', 'electronic']

# ===================== HELPER FUNCTIONS =======================
def search_spotify_tracks(query: str, limit: int = 10) -> List[Dict]:
    """Standalone function to search tracks (for backward compatibility)"""
    client = SpotifyClient(
        client_id="736bb144677e448dad56d2fe2ab70cd0",
        client_secret="d7beffe6e8d740deb7e1ddd9a111c88f"
    )
    return client.search_tracks(query, limit)

def get_audio_features(track_id: str) -> Dict[str, Any]:
    """Standalone function to get audio features (for backward compatibility)"""
    client = SpotifyClient(
        client_id="736bb144677e448dad56d2fe2ab70cd0",
        client_secret="d7beffe6e8d740deb7e1ddd9a111c88f"
    )
    return client.get_audio_features(track_id) or {}

# ===================== TESTING UTIL =======================
async def test_all_features():
    print("\n🔍 Testing Spotify Client...")
    
    # Initialize with your credentials directly
    client = SpotifyClient(
        client_id="736bb144677e448dad56d2fe2ab70cd0",
        client_secret="d7beffe6e8d740deb7e1ddd9a111c88f",
        redirect_uri="http://127.0.0.1:8000/callback"
    )

    try:
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

        print("\n🔍 Testing search...")
        results = client.search_tracks("Imagine Dragons Believer", limit=1)
        for track in results:
            print(f"Found: {track['name']} by {', '.join(track['artists'])}")

        print("\n🔍 Testing standalone functions...")
        print("Search results:", len(search_spotify_tracks("test", 1)))
        print("Audio features:", get_audio_features(results[0]['id'] if results else ""))

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    finally:
        print("\nTest completed")

if __name__ == "__main__":
    asyncio.run(test_all_features())
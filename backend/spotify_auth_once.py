import spotipy
from spotipy.oauth2 import SpotifyOAuth
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("spotify")
logger.setLevel(logging.INFO)

class SpotifyClient:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, token_info: Optional[Dict] = None):
        try:
            if token_info:
                self.sp = spotipy.Spotify(auth=token_info["access_token"])
                self.user_id = self.sp.current_user()["id"]
            else:
                self.auth_manager = SpotifyOAuth(
                    client_id=client_id,
                    client_secret=client_secret,
                    redirect_uri=redirect_uri,
                    scope="user-library-read user-top-read playlist-modify-public"
                )
                self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
                self.user_id = self.sp.current_user()["id"]
            logger.info("✅ Spotify Client initialized with OAuth")
        except Exception as e:
            logger.error(f"❌ Spotify OAuth initialization failed: {e}")
            raise

    def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            return results['tracks']['items']
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return []

    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        try:
            return self.sp.audio_features(track_ids)
        except Exception as e:
            logger.error(f"Error fetching audio features: {e}")
            return []

    def create_playlist(self, name: str, description: str = "") -> Dict:
        if not self.user_id:
            raise RuntimeError("Cannot create playlist without valid OAuth user")
        return self.sp.user_playlist_create(
            user=self.user_id,
            name=name,
            public=True,
            description=description
        )

    def add_to_playlist(self, playlist_id: str, track_ids: List[str]):
        if not track_ids:
            logger.warning("No tracks provided to add to playlist")
            return
        self.sp.playlist_add_items(playlist_id, track_ids)

    def get_auth_url(self) -> str:
        return self.auth_manager.get_authorize_url()

    def get_access_token(self, code: str) -> Dict:
        return self.auth_manager.get_access_token(code)

import os
import logging
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import MemoryCacheHandler
from typing import List, Dict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyClient:
    def __init__(self):
        self.client_id = "5b42d00bcf0a42de83f9bd8855c5d629"
        self.client_secret = "246b8f0425dc464ba1758a1e2ab4fa72"
        self.redirect_uri = "http://127.0.0.1:8000/callback"
        self.scope = (
            "user-library-read user-top-read playlist-modify-private "
            "playlist-modify-public user-read-recently-played user-read-private"
        )

        self.auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=self.scope,
            cache_handler=MemoryCacheHandler(),
            show_dialog=True
        )

        self.client = spotipy.Spotify(auth_manager=self.auth_manager)
        self.current_user = None

    async def authenticate_user(self, code: str = None) -> Dict:
        try:
            self.current_user = self.client.current_user()
            return {
                "id": self.current_user["id"],
                "display_name": self.current_user.get("display_name"),
                "email": self.current_user.get("email"),
                "image": self.current_user["images"][0]["url"] if self.current_user.get("images") else None
            }
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise Exception("Spotify authentication failed")

    async def get_recommendations(self, mood_vector: Dict, limit: int = 10) -> List[Dict]:
        try:
            features = mood_vector['audio_profile']
            results = self.client.recommendations(
                limit=limit,
                seed_genres=["pop", "rock", "electronic"],
                target_valence=features['valence'],
                target_energy=features['energy'],
                target_danceability=features['danceability'],
                target_tempo=features['tempo'],
                target_acousticness=features['acousticness']
            )
            return await self._format_recommendations(results['tracks'])
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            return []

    async def get_user_top_tracks(self, limit: int = 20, time_range: str = "medium_term") -> List[Dict]:
        try:
            results = self.client.current_user_top_tracks(
                limit=limit,
                time_range=time_range
            )
            return await self._format_tracks(results['items'])
        except Exception as e:
            logger.error(f"Failed to get top tracks: {str(e)}")
            return []

    async def create_playlist(self, user_id: str, name: str, description: str = "") -> Dict:
        try:
            playlist = self.client.user_playlist_create(
                user=user_id,
                name=name,
                description=description,
                public=False
            )
            return {
                "id": playlist["id"],
                "name": playlist["name"],
                "url": playlist["external_urls"]["spotify"],
                "tracks": []
            }
        except Exception as e:
            logger.error(f"Playlist creation failed: {str(e)}")
            raise Exception("Failed to create playlist")

    async def add_to_playlist(self, playlist_id: str, track_uris: List[str]) -> bool:
        try:
            self.client.playlist_add_items(playlist_id, track_uris)
            return True
        except Exception as e:
            logger.error(f"Failed to add tracks: {str(e)}")
            return False

    async def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        try:
            self._refresh_token_if_expired()
            features = self.client.audio_features(track_ids)
            return [f for f in features if f is not None]
        except Exception as e:
            logger.error(f"Failed to get audio features: {str(e)}")
            return []

    async def _format_tracks(self, tracks: List[Dict]) -> List[Dict]:
        formatted = []
        for track in tracks:
            features = await self.get_audio_features([track['id']])
            formatted.append({
                "id": track["id"],
                "name": track["name"],
                "artists": [{"id": a["id"], "name": a["name"]} for a in track["artists"]],
                "album": {
                    "id": track["album"]["id"],
                    "name": track["album"]["name"],
                    "image": track["album"]["images"][0]["url"] if track["album"].get("images") else None
                },
                "duration_ms": track["duration_ms"],
                "popularity": track["popularity"],
                "preview_url": track.get("preview_url"),
                "uri": track["uri"],
                "features": features[0] if features else {}
            })
        return formatted

    async def _format_recommendations(self, tracks: List[Dict]) -> List[Dict]:
        return await self._format_tracks(tracks)

    async def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        try:
            results = self.client.search(q=query, limit=limit, type="track")
            return await self._format_tracks(results['tracks']['items'])
        except Exception as e:
            logger.error(f"Track search failed: {str(e)}")
            return []

    def _refresh_token_if_expired(self):
        token_info = self.auth_manager.get_cached_token()
        if self.auth_manager.is_token_expired(token_info):
            new_token = self.auth_manager.refresh_access_token(token_info['refresh_token'])
            self.client = spotipy.Spotify(auth=new_token['access_token'])

if __name__ == "__main__":
    import asyncio

    async def test():
        client = SpotifyClient()
        await client.authenticate_user()
        tracks = await client.get_user_top_tracks(limit=5)
        print(json.dumps(tracks, indent=2))

    asyncio.run(test())


'''import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import RedisCacheHandler
import os
from typing import List, Dict, Optional
import logging
import redis
from datetime import datetime, timedelta
from fastapi import HTTPException
#from backend.spotify_module import SpotifyClient

logger = logging.getLogger(__name__)

class SpotifyClient:
    def __init__(self):
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        
        self.cache_handler = RedisCacheHandler(self.redis)
        self.scope = "user-library-read user-top-read playlist-modify-public playlist-modify-private user-read-recently-played"
        
        self.auth_manager = SpotifyOAuth(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
            scope=self.scope,
            cache_handler=self.cache_handler,
            show_dialog=True
        )
        
        self.client = spotipy.Spotify(auth_manager=self.auth_manager)
    
    async def authenticate_user(self, code: str) -> Dict[str, str]:
        """Complete OAuth flow and return user info."""
        try:
            token = self.auth_manager.get_access_token(code, as_dict=False)
            self.client = spotipy.Spotify(auth=token)
            user = self.client.current_user()
            
            return {
                "id": user["id"],
                "display_name": user.get("display_name"),
                "email": user.get("email"),
                "image": user["images"][0]["url"] if user.get("images") else None
            }
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Spotify authentication failed")
    
    async def get_user_top_items(self, user_id: str, type: str = "tracks", limit: int = 10) -> List[Dict]:
        """Get user's top tracks or artists."""
        try:
            results = self.client.current_user_top_tracks(limit=limit) if type == "tracks" \
                else self.client.current_user_top_artists(limit=limit)
            return results["items"]
        except Exception as e:
            logger.error(f"Failed to get top items: {str(e)}")
            return []
    
    async def get_recently_played(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get user's recently played tracks."""
        try:
            results = self.client.current_user_recently_played(limit=limit)
            return [item["track"] for item in results["items"]]
        except Exception as e:
            logger.error(f"Failed to get recently played: {str(e)}")
            return []
    
    async def search_tracks(self, query: str, limit: int = 10, market: str = None) -> List[Dict]:
        """Search for tracks on Spotify."""
        try:
            results = self.client.search(q=query, limit=limit, type="track", market=market)
            return self._format_track_results(results["tracks"]["items"])
        except Exception as e:
            logger.error(f"Track search failed: {str(e)}")
            return []
    
    async def get_track(self, track_id: str) -> Optional[Dict]:
        """Get track details by ID."""
        try:
            track = self.client.track(track_id)
            features = self.client.audio_features([track_id])[0]
            return self._format_track(track, features)
        except Exception as e:
            logger.error(f"Failed to get track: {str(e)}")
            return None
    
    async def get_artist_top_tracks(self, artist_id: str) -> List[Dict]:
        """Get artist's top tracks."""
        try:
            results = self.client.artist_top_tracks(artist_id)
            return self._format_track_results(results["tracks"])
        except Exception as e:
            logger.error(f"Failed to get artist tracks: {str(e)}")
            return []
    
    async def get_genre_recommendations(self, genre: str, limit: int = 10) -> List[Dict]:
        """Get recommendations based on genre."""
        try:
            results = self.client.recommendations(seed_genres=[genre], limit=limit)
            return self._format_track_results(results["tracks"])
        except Exception as e:
            logger.error(f"Genre recommendations failed: {str(e)}")
            return []
    
    async def create_playlist(self, user_id: str, name: str, description: str = "") -> Dict:
        """Create a new playlist for the user."""
        try:
            playlist = self.client.user_playlist_create(
                user=user_id,
                name=name,
                description=description,
                public=True
            )
            return {
                "id": playlist["id"],
                "name": playlist["name"],
                "url": playlist["external_urls"]["spotify"],
                "snapshot_id": playlist["snapshot_id"]
            }
        except Exception as e:
            logger.error(f"Playlist creation failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Playlist creation failed")
    
    async def add_to_playlist(self, playlist_id: str, track_uris: List[str]) -> bool:
        """Add tracks to an existing playlist."""
        try:
            self.client.playlist_add_items(playlist_id, track_uris)
            return True
        except Exception as e:
            logger.error(f"Failed to add tracks: {str(e)}")
            return False
    
    async def get_playlist(self, playlist_id: str) -> Optional[Dict]:
        """Get playlist details."""
        try:
            playlist = self.client.playlist(playlist_id)
            return {
                "id": playlist["id"],
                "name": playlist["name"],
                "description": playlist["description"],
                "tracks": [self._format_track(item["track"]) for item in playlist["tracks"]["items"]],
                "image": playlist["images"][0]["url"] if playlist.get("images") else None
            }
        except Exception as e:
            logger.error(f"Failed to get playlist: {str(e)}")
            return None
    
    def _format_track_results(self, tracks: List[Dict]) -> List[Dict]:
        """Format raw track results from Spotify."""
        return [self._format_track(track) for track in tracks]
    
    def _format_track(self, track: Dict, features: Dict = None) -> Dict:
        """Format a single track with optional audio features."""
        return {
            "id": track["id"],
            "uri": track["uri"],
            "name": track["name"],
            "artists": [{"id": a["id"], "name": a["name"]} for a in track["artists"]],
            "album": {
                "id": track["album"]["id"],
                "name": track["album"]["name"],
                "image": track["album"]["images"][0]["url"] if track["album"].get("images") else None
            },
            "duration_ms": track["duration_ms"],
            "popularity": track["popularity"],
            "preview_url": track.get("preview_url"),
            "features": features or {},
            "genres": []  # Will be populated separately
        }
    
    async def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """Get audio features for multiple tracks."""
        try:
            features = self.client.audio_features(track_ids)
            return [f for f in features if f is not None]
        except Exception as e:
            logger.error(f"Failed to get audio features: {str(e)}")
            return []
    
    async def get_artist_genres(self, artist_ids: List[str]) -> Dict[str, List[str]]:
        """Get genres for multiple artists."""
        try:
            artists = self.client.artists(artist_ids)["artists"]
            return {a["id"]: a["genres"] for a in artists}
        except Exception as e:
            logger.error(f"Failed to get artist genres: {str(e)}")
            return {}

            '''
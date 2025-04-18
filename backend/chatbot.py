from spotify import SpotifyClient
from recommendation import get_recommendations_with_filters
from song_index import SongIndex
from typing import List, Dict

class ChatbotController:
    def __init__(self):
        """Initialize the chatbot with hardcoded Spotify credentials (for testing only)"""
        # WARNING: In production, use environment variables instead of hardcoded credentials
        self.spotify = SpotifyClient(
            client_id="736bb144677e448dad56d2fe2ab70cd0",  # Replace with your actual client ID
            client_secret="d7beffe6e8d740deb7e1ddd9a111c88f",  # Replace with your actual client secret
            redirect_uri="http://127.0.0.1:8000/callback"  # Default callback URL
        )
        self.indexer = SongIndex()

    def handle_input(self, user_query: str, mood: Dict = None) -> List[Dict]:
        """
        Process user input and return recommended tracks
        
        Args:
            user_query: Text description of desired music
            mood: Dictionary with mood parameters (valence, energy, danceability)
            
        Returns:
            List of recommended track dictionaries
        """
        if mood is None:
            mood = {}
            
        try:
            # Get base tracks for recommendations
            base_tracks = self.spotify.get_user_top_tracks(limit=50)
            if not base_tracks:
                return []
                
            # Process and rank tracks
            embedded = self.indexer.embed_tracks(base_tracks)
            ranked = self.indexer.rank_tracks(user_query, embedded)

            # Apply mood filters if provided
            return (
                get_recommendations_with_filters(self.spotify, ranked, mood)
                if mood 
                else ranked
            )
            
        except Exception as e:
            print(f"Error in handle_input: {str(e)}")
            return []

    def create_playlist_from_results(self, name: str, results: List[Dict]) -> str:
        """
        Create a Spotify playlist from recommendation results
        
        Args:
            name: Playlist name
            results: List of track dictionaries
            
        Returns:
            URL of the created playlist or empty string on failure
        """
        if not results:
            return ""
            
        try:
            playlist = self.spotify.create_playlist(
                name=name,
                description="Created by PersonalAIs Music Companion"
            )
            track_ids = [t["id"] for t in results if t.get("id")]
            if track_ids:
                self.spotify.add_to_playlist(playlist["id"], track_ids)
                return playlist["external_urls"]["spotify"]
            return ""
        except Exception as e:
            print(f"Error creating playlist: {str(e)}")
            return ""


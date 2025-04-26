from typing import Dict, List, Optional
import logging
import json
from database import save_user_message, get_user_conversation_history
from recommendation import MusicRecommender, get_recommendations_with_filters
from spotify_client import SpotifyClient
from embeddings_emotion import SongIndex

#logging to display logs for debugging and tracing activity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedChatbot:
    def __init__(self, token_info=None):
        self.responses = {
            "welcome": (
                "Hi! I'm your music recommendation assistant. "
                "Tell me how you're feeling or what kind of music you'd like to hear."
            ),
            "no_input": "I didn't quite get that. Could you describe your mood or what you're doing?",
            "recommendation_intro": "Based on what you told me, here are some recommendations:",
            "follow_up": "Would you like more recommendations or something different?"
        }


        #  Initialize a Spotify client depending on whether user authentication is provided, these client id and all I have taken 
        #  from the spotify dashboard of my app
        if token_info:
            self.spotify = SpotifyClient(
                client_id="736bb144677e448dad56d2fe2ab70cd0",
                client_secret="d7beffe6e8d740deb7e1ddd9a111c88f",
                redirect_uri="http://127.0.0.1:8000/callback",
                token_info=token_info
            )
        else:
            self.spotify = SpotifyClient(
                client_id="736bb144677e448dad56d2fe2ab70cd0",
                client_secret="d7beffe6e8d740deb7e1ddd9a111c88f"
            )

# Here I have initialized the embedding-based song indexer for our requirement of emotion and semantic similarity
        self.indexer = SongIndex()

    async def handle_message(
        self,
        user_id: str,
        message: str,
        mood: Dict = None
    ) -> Dict:
        """Handles user input and returns a structured response with recommendations."""
        try:
            # Storing the user’s message for contextual tracking
            await save_user_message(user_id, message)

            # Checking for any empty or invalid message
            if not message.strip():
                return self._format_response(self.responses["no_input"])

            # This get the user's top tracks from Spotify to use as the recommendation model and its recommendations base
            base_tracks = self.spotify.get_user_top_tracks(limit=50)
            if not base_tracks:
                return self._format_response(
                    "Sorry, couldn't retrieve top tracks."
                )

            # Embedding all the tracks and rank them based on the similarity to the user mood description message
            embedded = self.indexer.embed_tracks(base_tracks)
            ranked_tracks = self.indexer.rank_tracks(message, embedded)

            # Optionally filter ranked results by mood dimensions
            recommendations = (
                get_recommendations_with_filters(
                    self.spotify,
                    ranked_tracks,
                    mood
                ) if mood else ranked_tracks
            )

            # this will construct the final response payload for the system
            explanation = self.responses['recommendation_intro']
            response = {
                "text": f"{explanation}\n\n{self.responses['follow_up']}",
                "recommendations": recommendations,
                "analysis": mood or {}
            }

            # It Logs out chatbot's response to maintain conversation history which is shown in the interface 
            await save_user_message(
                user_id,
                response["text"],
                is_bot=True
            )
            return response

        except Exception as e:
            logger.error(f"Chat handling failed: {e}")
            return self._format_response(
                "Sorry, I couldn't process your request. Please try again."
            )

    def create_playlist_from_results(
        self,
        name: str,
        results: List[Dict]
    ) -> str:
        """
        Creates and returns a Spotify playlist URL from recommendation results.
        """
        try:
            if not results:
                return ""
            # This creates a plylist in the user spotify account where the songs could be added by the chatbot automatically 
            playlist = self.spotify.create_playlist(
                name=name,
                description="Created by Music Recommender"
            )

            # This part extracts track IDs and add them to the newly created playlist
            track_ids = [t["id"] for t in results if t.get("id")]
            self.spotify.add_to_playlist(
                playlist_id=playlist["id"],
                track_ids=track_ids
            )
            return playlist["external_urls"]["spotify"]
        except Exception as e:
            logger.error(f"Playlist creation failed: {e}")
            return ""

    async def get_conversation_history(
        self,
        user_id: str
    ) -> List[Dict]:
        """Returns the user's recent mobile chat history."""
        history = await get_user_conversation_history(
            user_id=user_id,
            limit=10
        )
        return [
            {
                "message": msg.message,
                "is_bot": msg.is_bot,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in history
        ]

    def _format_response(self, text: str) -> Dict:
        """Returns a default response dictionary with no recommendations."""
        return {
            "text": text,
            "recommendations": [],
            "analysis": {}
        }

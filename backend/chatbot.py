from spotify import SpotifyClient
from recommendation import get_recommendations_with_filters
from song_index import SongIndexer
from typing import List

class ChatbotController:
    def __init__(self):
        self.spotify = SpotifyClient()
        self.indexer = SongIndexer()

    def handle_input(self, user_query: str, mood: dict = {}) -> list:
        """Main chatbot interface: get query, mood, return ranked songs"""
        base_tracks = self.spotify.get_user_top_tracks(limit=50)
        embedded = self.indexer.embed_tracks(base_tracks)
        ranked = self.indexer.rank_tracks(user_query, embedded)

        if mood:
            filtered = get_recommendations_with_filters(self.spotify, ranked, mood)
            return filtered
        return ranked

    def create_playlist_from_results(self, name: str, results: List[dict]) -> str:
        playlist = self.spotify.create_playlist(name)
        track_ids = [t["id"] for t in results]
        self.spotify.add_to_playlist(playlist["id"], track_ids)
        return playlist["external_urls"]["spotify"]


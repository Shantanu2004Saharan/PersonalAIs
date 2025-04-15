from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MusicMatcher:
    def __init__(self, spotify_client):
        self.spotify = spotify_client
        self.emotion_cache = {}  # Stores recent emotion mappings

    async def find_recommendations(self, emotion_profile: Dict) -> List[Dict]:
        """Find songs matching emotional profile"""
        params = self._create_spotify_params(emotion_profile)
        tracks = await self.spotify.get_recommendations(**params)
        return self._rerank_tracks(tracks, emotion_profile)

    def _create_spotify_params(self, emotion_profile) -> Dict:
        """Convert emotion profile to Spotify API parameters"""
        music_profile = emotion_profile['music_profile']

        return {
            'limit': 15,
            'target_valence': music_profile['valence'],
            'target_energy': music_profile['energy'],
            'target_danceability': music_profile['danceability'],
            'min_acousticness': max(0, 1 - music_profile['complexity'] - 0.2),
            'max_acousticness': min(1, 1 - music_profile['complexity'] + 0.2),
            'seed_genres': self._select_genres(emotion_profile)
        }

    def _select_genres(self, emotion_profile) -> str:
        """Select genres based on emotion profile"""
        primary = emotion_profile['primary_emotion']

        genre_map = {
            'joy': 'pop,dance,disco',
            'sadness': 'blues,jazz,soul',
            'anger': 'rock,metal,punk',
            'love': 'r-n-b,pop,soul',
            'fear': 'ambient,classical,new-age',
            'surprise': 'edm,indie,alternative'
        }
        return genre_map.get(primary, 'pop')

    def _rerank_tracks(self, tracks, emotion_profile) -> List[Dict]:
        """Re-order tracks by emotional fit"""
        for track in tracks:
            track['emotional_fit'] = self._calculate_fit_score(track, emotion_profile)
        return sorted(tracks, key=lambda x: x['emotional_fit'], reverse=True)

    def _calculate_fit_score(self, track, emotion_profile) -> float:
        """Calculate emotional fit score (0-1)"""
        valence_score = 1 - abs(track['valence'] - emotion_profile['music_profile']['valence'])
        energy_score = 1 - abs(track['energy'] - emotion_profile['music_profile']['energy'])
        danceability_score = 1 - abs(track['danceability'] - emotion_profile['music_profile']['danceability'])
        return (valence_score + energy_score + danceability_score) / 3
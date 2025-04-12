import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.nlp_module import MoodVector
from backend.spotify_module import SpotifyClient
from backend.database import save_interaction, get_user_history

import logging
import numpy as np

logger = logging.getLogger(__name__)

class MusicRecommender:
    def __init__(self, user_id):
        self.user_id = user_id
        self.history = get_user_history(user_id)
        self.current_mood = MoodVector()

    def process_description(self, description):
        self.current_mood.process_description(description)
        logger.info(f"Processed description. Mood vector: {self.current_mood.get_vector()}")

    def recommend(self, description=None):
        if description:
            self.process_description(description)

        # Generate playlist using Spotify API
        mood_vector = self.current_mood.get_vector()
        recommendations = SpotifyClient(mood_vector)

        # Store this interaction
        save_interaction(self.user_id, description, mood_vector, recommendations)

        return recommendations

    def refine_with_feedback(self, new_description):
        self.process_description(new_description)
        return self.recommend()







'''import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from database import get_user_profile, get_song_features

class RecommendationEngine:
    def __init__(self):
        self.audio_feature_weights = {
            'valence': 0.25,
            'energy': 0.2,
            'tempo': 0.15,
            'danceability': 0.1,
            'acousticness': 0.1,
            'semantic': 0.2
        }
    
    async def recommend_songs(self, user_id: str, mood_vector: Dict) -> List[Dict]:
        """Generate personalized recommendations based on mood vector"""
        # 1. Get user profile and candidate songs
        user_profile = await get_user_profile(user_id)
        all_songs = await get_song_features()
        
        # 2. Score each song
        scored_songs = []
        for song in all_songs:
            score = self.calculate_match_score(song, mood_vector, user_profile)
            scored_songs.append((score, song))
        
        # 3. Sort and apply diversity
        scored_songs.sort(key=lambda x: x[0], reverse=True)
        return self.apply_diversity(scored_songs)
    
    def calculate_match_score(self, song: Dict, mood: Dict, user: Dict) -> float:
        """Calculate personalized match score (0-1)"""
        # 1. Audio feature similarity
        audio_score = 0
        for feature, weight in self.audio_feature_weights.items():
            if feature in mood['audio_profile'] and feature in song['features']:
                audio_score += weight * (1 - abs(mood['audio_profile'][feature] - song['features'][feature]))
        
        # 2. Semantic similarity
        semantic_sim = cosine_similarity(
            [mood['semantic_embed']],
            [song['semantic_embed']]
        )[0][0]
        semantic_score = self.audio_feature_weights['semantic'] * semantic_sim
        
        # 3. Personalization factor
        personal_score = 0
        if user['preferred_artists']:
            artist_match = any(artist['id'] in user['preferred_artists'] for artist in song['artists'])
            personal_score += 0.1 if artist_match else 0
        
        # 4. Contextual bonuses
        context_bonus = self.calculate_context_bonus(song, mood, user)
        
        return audio_score + semantic_score + personal_score + context_bonus
    
    def calculate_context_bonus(self, song: Dict, mood: Dict, user: Dict) -> float:
        """Calculate bonuses based on context"""
        bonus = 0
        
        # Activity matching
        if mood['activities'] and song['activity_tags']:
            matched_activities = set(mood['activities']) & set(song['activity_tags'])
            bonus += 0.05 * len(matched_activities)
        
        # Temporal context
        if mood['temporal_context'] and mood['temporal_context'] in song['time_tags']:
            bonus += 0.03
        
        # Metaphor alignment
        if mood['metaphors'] and song['mood_tags']:
            for _, metaphor in mood['metaphors']:
                if metaphor in song['mood_tags']:
                    bonus += 0.02
        
        return min(bonus, 0.1)  # Cap at 0.1
    
    def apply_diversity(self, scored_songs: List, top_n: int = 25) -> List[Dict]:
        """Ensure diverse recommendations"""
        # Simple diversity sampling - could be enhanced
        return [song for _, song in scored_songs[:top_n]]
    
'''
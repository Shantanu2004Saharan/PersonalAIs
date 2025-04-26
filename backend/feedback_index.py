# feedback_index.py
import logging
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from database import update_user_preferences, Feedback, UserPreference
from model_matcher_nlp import TrackRecommendation
from recommendation import MusicRecommender
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import select

logger = logging.getLogger(__name__)

class FeedbackIndex:
    def __init__(self, db: AsyncSession, recommender: MusicRecommender):
        self.db = db
        self.recommender = recommender
        self.feedback_weights = {
            'like': 1.2,
            'dislike': 0.8
        }
        self.feature_weights = {
            'valence': 0.6,
            'energy': 0.7,
            'danceability': 0.5
        }

    async def process_feedback(self, user_id: str, track_id: str, liked: bool):
        """Process feedback and update user preferences/track indices"""
        try:
            # 1. Save explicit feedback
            await self._save_feedback_record(user_id, track_id, liked)
            
            # 2. Update user preferences
            await self._update_user_preferences(user_id, track_id, liked)
            
            # 3. Adjust track weights in recommendation index
            await self._update_track_weights(track_id, liked)
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Feedback processing failed: {str(e)}")
            raise

    async def _save_feedback_record(self, user_id: str, track_id: str, liked: bool):
        """Store feedback in database"""
        feedback = Feedback(
            user_id=user_id,
            track_id=track_id,
            liked=liked,
            feedback_type="explicit"
        )
        self.db.add(feedback)
        await self.db.commit()

    async def _update_user_preferences(self, user_id: str, track_id: str, liked: bool):
        """Update user's musical preferences based on feedback"""
        # Get track features from recommendation system
        track_features = await self.recommender.get_track_features(track_id)
        
        if not track_features:
            logger.warning(f"No features found for track {track_id}")
            return

        # Calculate preference updates with decay factor
        weight = self.feedback_weights['like' if liked else 'dislike']
        preference_update = {
            'genres': {genre: weight * 0.9 for genre in track_features.get('genres', [])},
            'audio_features': {k: v * weight * self.feature_weights.get(k, 0.5) 
                            for k, v in track_features.items() 
                            if k in self.feature_weights}
        }
        
        # Merge with existing preferences
        existing_prefs = await self._get_existing_preferences(user_id)
        merged_prefs = self._merge_preferences(existing_prefs, preference_update)
        
        await update_user_preferences(self.db, user_id, merged_prefs)

    async def _get_existing_preferences(self, user_id: str) -> Dict:
        """Retrieve current user preferences"""
        result = await self.db.execute(
            select(UserPreference).where(UserPreference.user_id == user_id)
        )
        prefs = result.scalars().first()
        return prefs.preferences if prefs else {}

    def _merge_preferences(self, existing: Dict, new: Dict) -> Dict:
        """Merge new preferences with existing ones"""
        merged = existing.copy()
        
        # Merge genres
        for genre, weight in new.get('genres', {}).items():
            merged['genres'] = merged.get('genres', {})
            merged['genres'][genre] = merged['genres'].get(genre, 0) + weight
            
        # Merge audio features
        for feature, value in new.get('audio_features', {}).items():
            merged['audio_features'] = merged.get('audio_features', {})
            merged['audio_features'][feature] = merged['audio_features'].get(feature, 0) + value
            
        return merged

    async def _update_track_weights(self, track_id: str, liked: bool):
        """Adjust track's position in recommendation indices"""
        adjustment = 0.1 if liked else -0.1
        try:
            # Example implementation for FAISS index adjustment
            if hasattr(self.recommender, 'update_index_weights'):
                await self.recommender.update_index_weights(track_id, adjustment)
                
            # Example implementation for Spotify API boost
            if hasattr(self.recommender.spotify, 'boost_track'):
                await self.recommender.spotify.boost_track(track_id, adjustment)
                
        except Exception as e:
            logger.error(f"Index update failed: {str(e)}")

class FeedbackLearner(FeedbackIndex):
    """Backward-compatible alias for existing code references"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
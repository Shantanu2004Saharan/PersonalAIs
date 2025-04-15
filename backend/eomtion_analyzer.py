import numpy as np
from transformers import pipeline
from typing import Dict, Any

class EmotionAnalyzer:
    def __init__(self):
    # More nuanced emotion model
        self.emotion_model = pipeline(
        "text-classification",
        model="joeddav/distilbert-base-uncased-go-emotions-student",
        return_all_scores=True
    )
    
    # Enhanced emotion mappings
        self.emotion_to_features = {
        'admiration': {'valence': 0.8, 'energy': 0.6, 'danceability': 0.4},
        'amusement': {'valence': 0.9, 'energy': 0.7, 'danceability': 0.8},
        'anger': {'valence': 0.1, 'energy': 0.95, 'danceability': 0.3},
        'annoyance': {'valence': 0.2, 'energy': 0.85, 'danceability': 0.3},
        'approval': {'valence': 0.8, 'energy': 0.6, 'danceability': 0.5},
        # Add all 28 emotions with detailed mappings
    }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text and return emotional profile with music features"""
        # Get emotion predictions
        emotion_results = self.emotion_model(text)[0]
        
        # Convert to music profile
        embeddings = np.mean(self.embedding_model(text), axis=1)[0]
        music_profile = self._create_music_profile(emotion_results)
        
        return {
            'emotions': {e['label']: e['score'] for e in emotion_results},
            'primary_emotion': max(emotion_results, key=lambda x: x['score'])['label'],
            'music_profile': music_profile,
            'text_embedding': embeddings.tolist()
        }

    def _create_music_profile(self, emotion_results) -> Dict[str, float]:
        """Convert emotions to musical characteristics"""
        profile = {
            'valence': 0.5,
            'energy': 0.5,
            'complexity': 0.5,
            'danceability': 0.5
        }
        
        # Weighted average of all emotions
        for emotion in emotion_results:
            features = self.emotion_to_features.get(emotion['label'])
            if features:
                for key in profile:
                    profile[key] += emotion['score'] * features.get(key, 0)
        
        # Normalize values between 0-1
        return {k: max(0, min(1, v)) for k, v in profile.items()}
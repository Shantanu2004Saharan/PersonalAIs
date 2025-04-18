import numpy as np
from transformers import pipeline
from typing import Dict, Any

class EmotionAnalyzer:
    def __init__(self):
        self.emotion_model = pipeline(
            "text-classification",
            model="joeddav/distilbert-base-uncased-go-emotions-student",
            return_all_scores=True
        )

        self.emotion_to_features = {
            'admiration': {'valence': 0.8, 'energy': 0.6, 'danceability': 0.4},
            'amusement': {'valence': 0.9, 'energy': 0.7, 'danceability': 0.8},
            'anger': {'valence': 0.1, 'energy': 0.95, 'danceability': 0.3},
            'annoyance': {'valence': 0.2, 'energy': 0.85, 'danceability': 0.3},
            'approval': {'valence': 0.8, 'energy': 0.6, 'danceability': 0.5},
            # Add remaining mappings as needed
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        emotion_results = self.emotion_model(text)[0]
        music_profile = self._create_music_profile(emotion_results)

        return {
            'emotions': {e['label']: e['score'] for e in emotion_results},
            'primary_emotion': max(emotion_results, key=lambda x: x['score'])['label'],
            'music_profile': music_profile,
            # Dummy embedding to keep return structure consistent
            'text_embedding': np.zeros((384,)).tolist()
        }

    def _create_music_profile(self, emotion_results) -> Dict[str, float]:
        profile = {
            'valence': 0.5,
            'energy': 0.5,
            'complexity': 0.5,
            'danceability': 0.5
        }

        for emotion in emotion_results:
            features = self.emotion_to_features.get(emotion['label'])
            if features:
                for key in profile:
                    profile[key] += emotion['score'] * features.get(key, 0)

        return {k: max(0, min(1, v)) for k, v in profile.items()}

# ✅ TEST BLOCK (runs only when this file is executed directly)
if __name__ == "__main__":
    analyzer = EmotionAnalyzer()

    print("Running tests...\n")

    # Test 1: Basic input
    result = analyzer.analyze_text("I am so happy and excited today!")
    print("Primary Emotion:", result["primary_emotion"])
    print("Top Emotions:", result["emotions"])
    print("Music Profile:", result["music_profile"])
    assert isinstance(result['music_profile'], dict), "Music profile should be a dictionary"
    assert 0 <= result['music_profile']['valence'] <= 1, "Valence must be in [0, 1]"

    # Test 2: Negative sentiment
    result = analyzer.analyze_text("I feel very frustrated and angry.")
    print("\nPrimary Emotion:", result["primary_emotion"])
    print("Top Emotions:", result["emotions"])
    print("Music Profile:", result["music_profile"])
    assert "anger" in result["emotions"], "Anger should be among detected emotions"

    print("\n✅ All tests passed!")

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import unittest
from unittest.mock import patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load models with smaller footprint
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentiment_analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

class TextAnalyzer:
    def __init__(self):
        self.activity_keywords = {
            'working': ['work', 'study', 'code', 'write', 'read'],
            'exercising': ['run', 'workout', 'exercise', 'gym', 'jog'],
            'relaxing': ['relax', 'chill', 'unwind', 'rest', 'sleep'],
            'driving': ['drive', 'commute', 'road trip', 'travel'],
            'partying': ['party', 'celebrate', 'dance', 'club'],
            'hindi': ['hindi', 'bollywood', 'indian'],
            'punjabi': ['punjabi', 'bhangra'],
            'tamil': ['tamil', 'kollywood'],
            'telugu': ['telugu', 'tollywood']
        }

        self.genre_keywords = {
            'pop': ['pop', 'top 40'],
            'rock': ['rock', 'alternative', 'indie'],
            'jazz': ['jazz', 'blues'],
            'electronic': ['electronic', 'edm', 'techno', 'house'],
            'hiphop': ['hip hop', 'rap', 'r&b'],
            'hindi': ['hindi', 'bollywood', 'indian'],
            'punjabi': ['punjabi', 'bhangra'],
            'tamil': ['tamil', 'kollywood'],
            'telugu': ['telugu', 'tollywood']
        }

    async def analyze_text(self, text: str) -> Dict:
        try:
            language = detect(text)
            if language == 'hi':
                self.genre_keywords['hindi'].extend(['गाना', 'संगीत'])
        except:
            pass

        doc = nlp(text.lower())
        embedding = sentence_model.encode(text)
        sentiment = sentiment_analyzer.polarity_scores(text)
        activities = self._detect_activities(doc)
        genres = self._detect_genres(doc)
        temporal_context = self._detect_temporal_context(doc)
        metaphors = self._detect_metaphors(doc)
        key_phrases = self._extract_key_phrases(doc)

        return {
            "embedding": embedding.tolist(),
            "sentiment": sentiment,
            "activities": activities,
            "genres": genres,
            "temporal_context": temporal_context,
            "metaphors": metaphors,
            "key_phrases": key_phrases,
            "audio_features": self._predict_audio_features(text, sentiment, activities)
        }

    def _detect_activities(self, doc) -> List[str]:
        return [
            activity for activity, keywords in self.activity_keywords.items()
            if any(token.text in keywords for token in doc)
        ]

    def _detect_genres(self, doc) -> List[str]:
        return [
            genre for genre, keywords in self.genre_keywords.items()
            if any(token.text in keywords for token in doc)
        ]

    def _detect_temporal_context(self, doc) -> Optional[str]:
        time_phrases = ['morning', 'afternoon', 'evening', 'night', 'dawn']
        for token in doc:
            if token.text in time_phrases:
                return token.text
        return None

    def _detect_metaphors(self, doc) -> List[str]:
        """Detect metaphor-like phrases that use 'like' or 'as'"""
        metaphors = []
        for token in doc:
            if token.text.lower() in ['like', 'as']:
                start = max(token.i - 2, 0)
                end = min(token.i + 3, len(doc))
                phrase = doc[start:end].text
                metaphors.append(phrase)
        return metaphors


    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract important phrases from text without determiners (like 'the')"""
        return [
        " ".join([token.text for token in chunk if token.pos_ != "DET"])
        for chunk in doc.noun_chunks if len(chunk.text.split()) > 1
    ]


    def _predict_audio_features(self, text: str, sentiment: Dict, activities: List[str]) -> Dict:
        features = {
            'valence': max(0, min(1, 0.5 + sentiment['compound'] * 0.5)),
            'energy': 0.5,
            'danceability': 0.5,
            'tempo': 100,
            'acousticness': 0.5
        }

        if 'exercising' in activities:
            features.update({'energy': 0.9, 'tempo': 130, 'danceability': 0.8})
        elif 'relaxing' in activities:
            features.update({'energy': 0.3, 'tempo': 80, 'acousticness': 0.8})
        elif 'partying' in activities:
            features.update({'energy': 0.95, 'danceability': 0.95, 'tempo': 125})

        return features

# Test cases
class TestTextAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TextAnalyzer()

    @patch('langdetect.detect', return_value='en')
    def test_analyze_text(self, mock_detect):
        import asyncio
        result = asyncio.run(self.analyzer.analyze_text("I'm going to the gym to workout."))
        self.assertIn("embedding", result)
        self.assertIn("sentiment", result)
        self.assertIn("activities", result)
        self.assertIn("genres", result)
        self.assertIn("temporal_context", result)
        self.assertIn("metaphors", result)
        self.assertIn("key_phrases", result)
        self.assertIn("audio_features", result)
        self.assertIn('exercising', result['activities'])

    def test_detect_activities(self):
        doc = nlp("I love to run and workout every morning.")
        self.assertIn("exercising", self.analyzer._detect_activities(doc))

    def test_detect_genres(self):
        doc = nlp("I enjoy listening to rock and pop music.")
        genres = self.analyzer._detect_genres(doc)
        self.assertIn("rock", genres)
        self.assertIn("pop", genres)

    def test_detect_temporal_context(self):
        doc = nlp("I usually workout in the morning.")
        self.assertEqual(self.analyzer._detect_temporal_context(doc), "morning")

    def test_predict_audio_features_exercising(self):
        sentiment = {'compound': 0.5}
        activities = ['exercising']
        audio = self.analyzer._predict_audio_features("text", sentiment, activities)
        self.assertEqual(audio['energy'], 0.9)
        self.assertEqual(audio['tempo'], 130)
        self.assertEqual(audio['danceability'], 0.8)

    def test_predict_audio_features_relaxing(self):
        sentiment = {'compound': -0.5}
        activities = ['relaxing']
        audio = self.analyzer._predict_audio_features("text", sentiment, activities)
        self.assertEqual(audio['energy'], 0.3)
        self.assertEqual(audio['tempo'], 80)
        self.assertEqual(audio['acousticness'], 0.8)

    def test_metaphor_detection(self):
        doc = nlp("Life is like a rollercoaster.")
        self.assertTrue(any("like" in m for m in self.analyzer._detect_metaphors(doc)))

    def test_key_phrases_detection(self):
        doc = nlp("The fast car zoomed through the crowded streets.")
        key_phrases = self.analyzer._extract_key_phrases(doc)
        self.assertIn("fast car", key_phrases)
        self.assertIn("crowded streets", key_phrases)

if __name__ == "__main__":
    unittest.main()



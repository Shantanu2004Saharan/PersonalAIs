from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

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
        """Comprehensive text analysis for music recommendation"""

        try:
            language = detect(text)
            if language == 'hi':
        # Adjust weights for Hindi content
                self.genre_keywords['hindi'].extend(['गाना', 'संगीत'])
        except:
            pass

        doc = nlp(text.lower())
        
        # Semantic embedding
        embedding = sentence_model.encode(text)
        
        # Sentiment analysis
        sentiment = sentiment_analyzer.polarity_scores(text)
        
        # Extract features
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
        """Detect activities from text"""
        return [
            activity for activity, keywords in self.activity_keywords.items()
            if any(token.text in keywords for token in doc)
        ]
    
    def _detect_genres(self, doc) -> List[str]:
        """Detect music genres from text"""
        return [
            genre for genre, keywords in self.genre_keywords.items()
            if any(token.text in keywords for token in doc)
        ]
    
    def _detect_temporal_context(self, doc) -> Optional[str]:
        """Detect time references in text"""
        time_phrases = ['morning', 'afternoon', 'evening', 'night', 'dawn']
        for token in doc:
            if token.text in time_phrases:
                return token.text
        return None
    
    def _detect_metaphors(self, doc) -> List[str]:
        """Detect metaphorical language"""
        return [
            chunk.text for chunk in doc.noun_chunks 
            if any(tok.text in ['like', 'as'] for tok in chunk)
        ]
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract important phrases from text"""
        return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    
    def _predict_audio_features(self, text: str, sentiment: Dict, activities: List[str]) -> Dict:
        """Predict ideal audio features based on text analysis"""
        features = {
            'valence': max(0, min(1, 0.5 + sentiment['compound'] * 0.5)),
            'energy': 0.5,
            'danceability': 0.5,
            'tempo': 100,
            'acousticness': 0.5
        }
        
        # Adjust based on activities
        if 'exercising' in activities:
            features.update({'energy': 0.9, 'tempo': 130, 'danceability': 0.8})
        elif 'relaxing' in activities:
            features.update({'energy': 0.3, 'tempo': 80, 'acousticness': 0.8})
        elif 'partying' in activities:
            features.update({'energy': 0.95, 'danceability': 0.95, 'tempo': 125})
        
        return features






'''from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import numpy as np
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load models with error handling
    try:
        # Using a more reliable model
        sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        logger.info("Sentence transformer model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load sentence transformer: {e}")
        raise

    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    try:
        nlp = spacy.load("en_core_web_sm")  # Using smaller model
        logger.info("spaCy model loaded successfully")
    except OSError:
        logger.warning("spaCy model not found, downloading...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

except ImportError as e:
    logger.error(f"Missing required library: {e}")
    raise
except Exception as e:
    logger.error(f"Initialization error: {e}")
    raise

class MoodVector:
    def __init__(self):
        self.semantic_embed = None
        self.audio_profile = {
            'valence': 0.5,
            'energy': 0.5,
            'danceability': 0.5,
            'tempo': 120,
            'acousticness': 0.5
        }
        self.activities = []
        self.metaphors = []
        self.temporal_context = None

async def analyze_description(text: str) -> Dict:
    """Deep analysis of user's description"""
    try:
        doc = nlp(text)
        mv = MoodVector()
        
        # 1. Semantic embedding
        mv.semantic_embed = sentence_model.encode(text)
        
        # 2. Enhanced sentiment analysis
        sentiment = sentiment_analyzer.polarity_scores(text)
        mv.audio_profile['valence'] = normalize(sentiment['compound'], -1, 1, 0, 1)
        
        # 3. Activity detection
        mv.activities = detect_activities(doc)
        
        # 4. Metaphor detection
        mv.metaphors = detect_metaphors(doc)
        
        # 5. Temporal context
        mv.temporal_context = detect_temporal_context(doc)
        
        # 6. Predict audio features
        mv.audio_profile.update(predict_audio_features(text))
        
        return mv.__dict__
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

def normalize(value, old_min, old_max, new_min, new_max):
    """Normalize value to new range"""
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

def detect_activities(doc) -> List[str]:
    """Detect activities like studying, working, chilling, etc."""
    activities = []
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ in {"study", "work", "relax", "exercise", "dance", "run", "walk", "drive"}:
            activities.append(token.lemma_)
    return activities

def detect_metaphors(doc) -> List[str]:
    """Placeholder for metaphor detection (to be improved)."""
    metaphors = []
    for sent in doc.sents:
        if "like" in sent.text or "as" in sent.text:
            metaphors.append(sent.text.strip())
    return metaphors

def detect_temporal_context(doc) -> str:
    """Detect whether the text refers to morning/evening, etc."""
    for token in doc:
        if token.lower_ in {"morning", "evening", "night", "afternoon", "midnight"}:
            return token.lower_
    return "unspecified"

def predict_audio_features(text: str) -> Dict[str, float]:
    """Heuristic-based audio profile prediction"""
    lowered = text.lower()
    if any(word in lowered for word in ["chill", "relax", "calm", "slow"]):
        return {
            "energy": 0.2,
            "danceability": 0.3,
            "tempo": 80,
            "acousticness": 0.8
        }
    elif any(word in lowered for word in ["party", "dance", "club", "hype", "excited"]):
        return {
            "energy": 0.9,
            "danceability": 0.95,
            "tempo": 130,
            "acousticness": 0.1
        }
    elif any(word in lowered for word in ["drive", "focus", "work"]):
        return {
            "energy": 0.6,
            "danceability": 0.5,
            "tempo": 110,
            "acousticness": 0.3
        }
    else:
        return {}  # Fallback — keep default

'''
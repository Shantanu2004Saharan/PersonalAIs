from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import numpy as np
from typing import Dict, List
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    sentiment_analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    
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

async def analyze_full_conversation(messages: List[str]) -> Dict:
    """Analyze full conversation history (last few messages)"""
    try:
        full_text = " ".join(messages)
        doc = nlp(full_text)
        mv = MoodVector()

        # Semantic embedding based on last message
        mv.semantic_embed = sentence_model.encode(full_text)

        # Sentiment (based on full conversation tone)
        sentiment = sentiment_analyzer.polarity_scores(full_text)
        mv.audio_profile['valence'] = normalize(sentiment['compound'], -1, 1, 0, 1)

        # Activity and metaphor analysis
        mv.activities = detect_activities(doc)
        mv.metaphors = detect_metaphors(doc)
        mv.temporal_context = detect_temporal_context(doc)

        # Additional audio prediction
        mv.audio_profile.update(predict_audio_features(full_text))

        return mv.__dict__
    except Exception as e:
        logger.error(f"Conversation analysis failed: {e}")
        raise

def normalize(value, old_min, old_max, new_min, new_max):
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

def detect_activities(doc) -> List[str]:
    verbs = {"study", "work", "relax", "exercise", "dance", "run", "walk", "drive", "party"}
    return list({token.lemma_ for token in doc if token.pos_ == "VERB" and token.lemma_ in verbs})

def detect_metaphors(doc) -> List[str]:
    return [sent.text.strip() for sent in doc.sents if "like" in sent.text or "as" in sent.text]

def detect_temporal_context(doc) -> str:
    for token in doc:
        if token.lower_ in {"morning", "evening", "night", "afternoon", "midnight"}:
            return token.lower_
    return "unspecified"

def predict_audio_features(text: str) -> Dict[str, float]:
    lowered = text.lower()
    if any(w in lowered for w in ["chill", "relax", "calm", "slow"]):
        return {"energy": 0.2, "danceability": 0.3, "tempo": 80, "acousticness": 0.8}
    if any(w in lowered for w in ["party", "dance", "club", "excited", "hype"]):
        return {"energy": 0.9, "danceability": 0.95, "tempo": 130, "acousticness": 0.1}
    if any(w in lowered for w in ["drive", "focus", "work", "productive"]):
        return {"energy": 0.6, "danceability": 0.5, "tempo": 110, "acousticness": 0.3}
    if any(w in lowered for w in ["angry", "rage", "fired up"]):
        return {"energy": 0.95, "danceability": 0.7, "tempo": 140, "acousticness": 0.2}
    return {}






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
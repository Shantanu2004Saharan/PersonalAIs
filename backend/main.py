'''@app.post("/recommend")
async def recommend_songs(request: Request, db: AsyncSession = Depends(get_async_session)):
    """
    Generate personalized song recommendations based on detailed text analysis.
    Uses semantic understanding rather than simple emotion classification.
    """
    try:
        data = await request.json()
        user_id = data['user_id']
        user_text = data['text']
        context = data.get('context', {})
        
        # 1. Perform deep linguistic analysis
        analysis = await analyze_text_comprehensively(user_text, db)
        
        # 2. Retrieve user's musical preferences and history
        user_profile = await get_user_profile_with_context(user_id, context, db)
        
        # 3. Generate personalized recommendations
        recommendations = await generate_personalized_recommendations(
            user_id=user_id,
            text_analysis=analysis,
            user_profile=user_profile,
            db=db
        )
        
        # 4. Prepare explainable insights
        insights = await generate_recommendation_insights(
            recommendations=recommendations,
            analysis=analysis,
            user_profile=user_profile
        )
        
        return {
            "status": "success",
            "analysis": insights['analysis'],
            "recommendations": insights['recommendations'],
            "explanation": insights['explanation']
        }
        
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Recommendation generation failed")

async def analyze_text_comprehensively(text: str, db: AsyncSession) -> Dict:
    """
    Perform multi-faceted text analysis including:
    - Semantic meaning (using sentence transformers)
    - Emotional tone (without rigid classification)
    - Activity detection
    - Temporal context
    - Metaphorical language
    """
    doc = nlp(text)
    
    # Semantic embedding (768-dimensional vector)
    semantic_embedding = sentence_model.encode(text)
    
    # Sentiment analysis (continuous scale)
    sentiment = sentiment_analyzer.polarity_scores(text)
    
    # Detect activities and contexts
    activities = detect_activities(doc)
    temporal_context = detect_temporal_context(doc)
    metaphors = detect_metaphors(doc)
    
    # Extract key musical aspects
    musical_prefs = extract_musical_preferences(doc)
    
    return {
        "semantic_embedding": semantic_embedding.tolist(),
        "sentiment": sentiment,
        "activities": activities,
        "temporal_context": temporal_context,
        "metaphors": metaphors,
        "musical_preferences": musical_prefs,
        "key_phrases": extract_key_phrases(doc)
    }

async def generate_personalized_recommendations(
    user_id: str,
    text_analysis: Dict,
    user_profile: Dict,
    db: AsyncSession
) -> List[Dict]:
    """
    Generate recommendations based on multiple factors:
    1. Semantic similarity to song lyrics/descriptions
    2. Audio feature matching
    3. Personal taste alignment
    4. Contextual relevance
    """
    # Get candidate songs from multiple sources
    candidates = await get_candidate_songs(
        user_id=user_id,
        text_analysis=text_analysis,
        user_profile=user_profile,
        db=db
    )
    
    # Score each candidate song
    scored_songs = []
    for song in candidates:
        score = calculate_song_score(
            song=song,
            text_analysis=text_analysis,
            user_profile=user_profile
        )
        scored_songs.append((score, song))
    
    # Sort by score and apply diversity
    scored_songs.sort(key=lambda x: x[0], reverse=True)
    return apply_diversity(scored_songs)

def calculate_song_score(song: Dict, text_analysis: Dict, user_profile: Dict) -> float:
    """
    Calculate personalized match score (0-1) considering:
    - Semantic similarity (40%)
    - Audio feature match (30%)
    - Personal preference (20%)
    - Contextual relevance (10%)
    """
    # 1. Semantic similarity to song lyrics/description
    semantic_sim = cosine_similarity(
        [text_analysis['semantic_embedding']],
        [song['semantic_embedding']]
    )[0][0] * 0.4
    
    # 2. Audio feature matching
    audio_score = 0
    target_features = predict_ideal_audio_features(text_analysis)
    for feature in ['valence', 'energy', 'tempo', 'danceability']:
        audio_score += 0.075 * (1 - abs(target_features[feature] - song['features'][feature]))
    
    # 3. Personal preference alignment
    personal_score = 0
    if user_profile['preferred_artists']:
        artist_match = any(a in user_profile['preferred_artists'] for a in song['artists'])
        personal_score += 0.1 if artist_match else 0
    
    if user_profile['preferred_genres']:
        genre_match = any(g in user_profile['preferred_genres'] for g in song['genres'])
        personal_score += 0.1 if genre_match else 0
    
    # 4. Contextual relevance
    context_score = calculate_context_score(song, text_analysis, user_profile)
    
    return semantic_sim + audio_score + personal_score + context_score

async def generate_recommendation_insights(
    recommendations: List[Dict],
    analysis: Dict,
    user_profile: Dict
) -> Dict:
    """
    Generate human-readable explanations for recommendations
    """
    # Analyze why top recommendations were selected
    top_song = recommendations[0]
    reasons = []
    
    # Semantic reason
    similar_phrases = find_semantic_overlap(
        analysis['key_phrases'],
        top_song['lyric_snippets']
    )
    if similar_phrases:
        reasons.append(f"Matches your mention of '{similar_phrases[0]}'")
    
    # Activity reason
    if analysis['activities'] and top_song['activity_tags']:
        matched_activities = set(analysis['activities']) & set(top_song['activity_tags'])
        if matched_activities:
            reasons.append(f"Great for {list(matched_activities)[0]}")
    
    # Audio feature reason
    feature_explanations = []
    if analysis['sentiment']['compound'] > 0.5 and top_song['features']['valence'] > 0.7:
        feature_explanations.append("upbeat mood")
    if analysis['sentiment']['compound'] < -0.5 and top_song['features']['valence'] < 0.3:
        feature_explanations.append("emotional depth")
    
    if feature_explanations:
        reasons.append(f"Has the right {', '.join(feature_explanations)}")
    
    # Personalization reason
    if any(a in user_profile['preferred_artists'] for a in top_song['artists']):
        reasons.append("From artists you love")
    
    return {
        "analysis": {
            "mood_characteristics": {
                "positivity": analysis['sentiment']['compound'],
                "energy_level": predict_energy_level(analysis),
                "temporal_context": analysis['temporal_context']
            },
            "detected_themes": analysis['key_phrases']
        },
        "recommendations": format_recommendations(recommendations),
        "explanation": {
            "main_reason": reasons[0] if reasons else "We think you'll enjoy this",
            "additional_factors": reasons[1:] if len(reasons) > 1 else []
        }
    }

'''


'''
# Fixed version:
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
sentence_model = SentenceTransformer('all-mpnet-base-v2')
sentiment_analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_lg")

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

@app.post("/recommend")
async def recommend_songs(request: Request, db: AsyncSession = Depends(get_async_session)):
    """
    Generate personalized song recommendations based on detailed text analysis.
    Uses semantic understanding rather than simple emotion classification.
    """
    try:
        data = await request.json()
        user_id = data['user_id']
        user_text = data['text']
        context = data.get('context', {})
        
        # 1. Perform deep linguistic analysis
        analysis = await analyze_text_comprehensively(user_text, db)
        
        # 2. Retrieve user's musical preferences and history
        user_profile = await get_user_profile_with_context(user_id, context, db)
        
        # 3. Generate personalized recommendations
        recommendations = await generate_personalized_recommendations(
            user_id=user_id,
            text_analysis=analysis,
            user_profile=user_profile,
            db=db
        )
        
        # 4. Prepare explainable insights
        insights = await generate_recommendation_insights(
            recommendations=recommendations,
            analysis=analysis,
            user_profile=user_profile
        )
        
        return {
            "status": "success",
            "analysis": insights['analysis'],
            "recommendations": insights['recommendations'],
            "explanation": insights['explanation']
        }
        
    except Exception as e:
        logger.error(f"Recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Recommendation generation failed")

async def analyze_text_comprehensively(text: str, db: AsyncSession) -> Dict:
    """
    Perform multi-faceted text analysis including:
    - Semantic meaning (using sentence transformers)
    - Emotional tone (without rigid classification)
    - Activity detection
    - Temporal context
    - Metaphorical language
    """
    doc = nlp(text)
    
    # Semantic embedding (768-dimensional vector)
    semantic_embedding = sentence_model.encode(text)
    
    # Sentiment analysis (continuous scale)
    sentiment = sentiment_analyzer.polarity_scores(text)
    
    # Detect activities and contexts
    activities = detect_activities(doc)
    temporal_context = detect_temporal_context(doc)
    metaphors = detect_metaphors(doc)
    
    # Extract key musical aspects
    musical_prefs = extract_musical_preferences(doc)
    
    return {
        "semantic_embedding": semantic_embedding.tolist(),
        "sentiment": sentiment,
        "activities": activities,
        "temporal_context": temporal_context,
        "metaphors": metaphors,
        "musical_preferences": musical_prefs,
        "key_phrases": extract_key_phrases(doc)
    }

def detect_activities(doc):
    """Detect activities from text"""
    activities = []
    activity_keywords = {
        'working': ['work', 'code', 'program', 'write'],
        'exercising': ['run', 'workout', 'exercise', 'gym'],
        'relaxing': ['relax', 'chill', 'unwind', 'rest'],
        'driving': ['drive', 'commute', 'road trip'],
        'partying': ['party', 'celebrate', 'dance']
    }
    
    for token in doc:
        for activity, keywords in activity_keywords.items():
            if token.lemma_ in keywords and activity not in activities:
                activities.append(activity)
    return activities

def detect_temporal_context(doc):
    """Detect temporal references"""
    time_phrases = ['morning', 'afternoon', 'evening', 'night', 
                    'today', 'tonight', 'weekend', 'summer', 'winter']
    for token in doc:
        if token.lemma_ in time_phrases:
            return token.lemma_
    return None

def detect_metaphors(doc):
    """Detect metaphorical language"""
    metaphors = {
        'blue': 'sadness',
        'cloud nine': 'happiness',
        'heavy heart': 'sadness',
        'light': 'happiness',
        'storm': 'turmoil'
    }
    detected = []
    for chunk in doc.noun_chunks:
        if chunk.text.lower() in metaphors:
            detected.append((chunk.text, metaphors[chunk.text.lower()]))
    return detected

def extract_musical_preferences(doc):
    """Extract musical preferences from text"""
    preferences = {
        'genres': [],
        'instruments': [],
        'characteristics': []
    }
    
    genre_keywords = {
        'jazz': ['jazz', 'blues'],
        'rock': ['rock', 'metal'],
        'electronic': ['electronic', 'edm', 'techno']
    }
    
    for token in doc:
        for genre, keywords in genre_keywords.items():
            if token.lemma_ in keywords and genre not in preferences['genres']:
                preferences['genres'].append(genre)
                
        if token.lemma_ in ['piano', 'guitar', 'violin']:
            if token.lemma_ not in preferences['instruments']:
                preferences['instruments'].append(token.lemma_)
                
        if token.lemma_ in ['fast', 'slow', 'loud', 'quiet']:
            preferences['characteristics'].append(token.lemma_)
    
    return preferences

def extract_key_phrases(doc):
    """Extract important phrases from text"""
    return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

async def get_user_profile_with_context(user_id: str, context: Dict, db: AsyncSession):
    """Retrieve user profile with contextual preferences"""
    # In a real implementation, this would query your database
    # This is a simplified version
    return {
        'preferred_artists': ['artist1', 'artist2'],
        'preferred_genres': ['rock', 'jazz'],
        'recently_played': ['song1', 'song2'],
        'contextual_preferences': context
    }

async def generate_personalized_recommendations(
    user_id: str,
    text_analysis: Dict,
    user_profile: Dict,
    db: AsyncSession
) -> List[Dict]:
    """
    Generate recommendations based on multiple factors:
    1. Semantic similarity to song lyrics/descriptions
    2. Audio feature matching
    3. Personal taste alignment
    4. Contextual relevance
    """
    # Get candidate songs from multiple sources
    candidates = await get_candidate_songs(
        user_id=user_id,
        text_analysis=text_analysis,
        user_profile=user_profile,
        db=db
    )
    
    # Score each candidate song
    scored_songs = []
    for song in candidates:
        score = calculate_song_score(
            song=song,
            text_analysis=text_analysis,
            user_profile=user_profile
        )
        scored_songs.append((score, song))
    
    # Sort by score and apply diversity
    scored_songs.sort(key=lambda x: x[0], reverse=True)
    return apply_diversity(scored_songs)

async def get_candidate_songs(user_id, text_analysis, user_profile, db):
    """Get candidate songs from various sources"""
    # In a real implementation, this would query your database and external APIs
    # Here's a simplified version with mock data
    return [
        {
            'id': 'song1',
            'name': 'Example Song 1',
            'artists': ['artist1'],
            'genres': ['rock'],
            'features': {
                'valence': 0.8,
                'energy': 0.7,
                'tempo': 120,
                'danceability': 0.6
            },
            'lyric_snippets': ['feeling good', 'happy day'],
            'activity_tags': ['working', 'driving'],
            'semantic_embedding': np.random.rand(768).tolist()
        },
        {
            'id': 'song2',
            'name': 'Example Song 2',
            'artists': ['artist2'],
            'genres': ['jazz'],
            'features': {
                'valence': 0.4,
                'energy': 0.5,
                'tempo': 90,
                'danceability': 0.4
            },
            'lyric_snippets': ['rainy day', 'feeling blue'],
            'activity_tags': ['relaxing'],
            'semantic_embedding': np.random.rand(768).tolist()
        }
    ]

def calculate_song_score(song: Dict, text_analysis: Dict, user_profile: Dict) -> float:
    """
    Calculate personalized match score (0-1) considering:
    - Semantic similarity (40%)
    - Audio feature match (30%)
    - Personal preference (20%)
    - Contextual relevance (10%)
    """
    # 1. Semantic similarity to song lyrics/description
    semantic_sim = cosine_similarity(
        [text_analysis['semantic_embedding']],
        [song['semantic_embedding']]
    )[0][0] * 0.4
    
    # 2. Audio feature matching
    audio_score = 0
    target_features = predict_ideal_audio_features(text_analysis)
    for feature in ['valence', 'energy', 'tempo', 'danceability']:
        audio_score += 0.075 * (1 - abs(target_features[feature] - song['features'][feature]))
    
    # 3. Personal preference alignment
    personal_score = 0
    if user_profile['preferred_artists']:
        artist_match = any(a in user_profile['preferred_artists'] for a in song['artists'])
        personal_score += 0.1 if artist_match else 0
    
    if user_profile['preferred_genres']:
        genre_match = any(g in user_profile['preferred_genres'] for g in song['genres'])
        personal_score += 0.1 if genre_match else 0
    
    # 4. Contextual relevance
    context_score = calculate_context_score(song, text_analysis, user_profile)
    
    return semantic_sim + audio_score + personal_score + context_score

def predict_ideal_audio_features(text_analysis):
    """Predict ideal audio features based on text analysis"""
    # Simplified version - in reality you'd use a more sophisticated model
    sentiment = text_analysis['sentiment']['compound']
    return {
        'valence': max(0, min(1, 0.5 + sentiment * 0.5)),
        'energy': 0.7 if 'exercising' in text_analysis['activities'] else 0.5,
        'tempo': 120 if 'exercising' in text_analysis['activities'] else 90,
        'danceability': 0.6 if 'partying' in text_analysis['activities'] else 0.4
    }

def calculate_context_score(song, text_analysis, user_profile):
    """Calculate contextual relevance score"""
    score = 0
    
    # Activity matching
    if text_analysis['activities'] and song['activity_tags']:
        matched_activities = set(text_analysis['activities']) & set(song['activity_tags'])
        score += 0.05 * len(matched_activities)
    
    # Temporal context
    if text_analysis['temporal_context'] == 'morning' and 'morning' in song.get('tags', []):
        score += 0.03
    elif text_analysis['temporal_context'] == 'night' and 'night' in song.get('tags', []):
        score += 0.03
    
    return min(score, 0.1)  # Cap at 0.1

def apply_diversity(scored_songs, top_n=5):
    """Ensure diverse recommendations"""
    # Simplified diversity implementation
    return [song for score, song in scored_songs[:top_n]]

async def generate_recommendation_insights(recommendations, analysis, user_profile):
    """
    Generate human-readable explanations for recommendations
    """
    if not recommendations:
        return {
            "analysis": analysis,
            "recommendations": [],
            "explanation": {
                "main_reason": "No recommendations found",
                "additional_factors": []
            }
        }
    
    top_song = recommendations[0]
    reasons = []
    
    # Semantic reason
    if analysis['key_phrases']:
        reasons.append(f"Matches your mention of '{analysis['key_phrases'][0]}'")
    
    # Activity reason
    if analysis['activities'] and top_song.get('activity_tags'):
        reasons.append(f"Great for {analysis['activities'][0]}")
    
    # Sentiment reason
    sentiment = analysis['sentiment']['compound']
    if sentiment > 0.6 and top_song['features']['valence'] > 0.7:
        reasons.append("Matches your positive mood")
    elif sentiment < -0.6 and top_song['features']['valence'] < 0.3:
        reasons.append("Fits your current mood")
    
    # Personalization reason
    if any(a in user_profile['preferred_artists'] for a in top_song['artists']):
        reasons.append("From artists you like")
    
    return {
        "analysis": {
            "mood_characteristics": {
                "positivity": analysis['sentiment']['compound'],
                "energy_level": "high" if analysis['sentiment']['compound'] > 0.5 else "low",
                "temporal_context": analysis['temporal_context']
            },
            "detected_themes": analysis['key_phrases']
        },
        "recommendations": [{
            'id': song['id'],
            'name': song['name'],
            'artists': song['artists'],
            'features': song['features']
        } for song in recommendations],
        "explanation": {
            "main_reason": reasons[0] if reasons else "Recommended based on your preferences",
            "additional_factors": reasons[1:] if len(reasons) > 1 else []
        }
    }

# Add this if you want to run the app directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


    '''

'''

import os
import logging
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from typing import Dict, List, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Music Recommendation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize ML models
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller model
    sentiment_analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise

# Pydantic models for request/response validation
class RecommendationRequest(BaseModel):
    user_id: str
    text: str
    context: Optional[Dict[str, Any]] = None

class SongRecommendation(BaseModel):
    song_id: str
    title: str
    artist: str
    match_score: float
    features: Dict[str, float]

class RecommendationResponse(BaseModel):
    status: str
    recommendations: List[SongRecommendation]
    analysis: Dict[str, Any]

# Database dependency
async def get_async_session() -> AsyncSession:
    """Override this with your actual database session setup"""
    raise NotImplementedError("Database session factory not implemented")

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_songs(request: RecommendationRequest, db: AsyncSession = Depends(get_async_session)):
    """
    Generate personalized song recommendations based on:
    - Text semantic analysis
    - User preferences
    - Contextual information
    """
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")

        # Perform text analysis
        analysis = await analyze_text(request.text)
        
        # Generate recommendations (mock implementation - replace with your actual logic)
        recommendations = [
            {
                "song_id": "123",
                "title": "Happy Days",
                "artist": "Sunshine Band",
                "match_score": 0.87,
                "features": {
                    "valence": 0.8,
                    "energy": 0.7,
                    "tempo": 120
                }
            }
        ]
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "analysis": {
                "sentiment": analysis["sentiment"],
                "key_phrases": analysis["key_phrases"],
                "detected_activities": analysis["activities"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Recommendation service error")

async def analyze_text(text: str) -> Dict[str, Any]:
    """Perform comprehensive text analysis"""
    doc = nlp(text)
    
    # Get semantic embedding
    embedding = sentence_model.encode(text)
    
    # Analyze sentiment
    sentiment = sentiment_analyzer.polarity_scores(text)
    
    # Extract key information
    activities = detect_activities(doc)
    temporal_context = detect_temporal_context(doc)
    key_phrases = extract_key_phrases(doc)
    
    return {
        "embedding": embedding.tolist(),
        "sentiment": sentiment,
        "activities": activities,
        "temporal_context": temporal_context,
        "key_phrases": key_phrases
    }

def detect_activities(doc) -> List[str]:
    """Detect activities from text"""
    activity_keywords = {
        'working': ['work', 'code', 'program'],
        'exercising': ['run', 'workout', 'gym'],
        'relaxing': ['relax', 'chill', 'rest'],
        'driving': ['drive', 'commute'],
        'partying': ['party', 'celebrate']
    }
    return [
        activity for activity, keywords in activity_keywords.items()
        if any(token.lemma_ in keywords for token in doc)
    ]

def detect_temporal_context(doc) -> Optional[str]:
    """Detect time references in text"""
    time_phrases = ['morning', 'afternoon', 'evening', 'night']
    for token in doc:
        if token.lemma_ in time_phrases:
            return token.lemma_
    return None

def extract_key_phrases(doc) -> List[str]:
    """Extract meaningful phrases from text"""
    return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

def calculate_song_score(song: Dict, analysis: Dict) -> float:
    """Calculate recommendation score (0-1)"""
    # Semantic similarity (40%)
    semantic_sim = cosine_similarity(
        [analysis["embedding"]],
        [song["embedding"]]
    )[0][0] * 0.4
    
    # Feature matching (30%)
    target_valence = 0.5 + analysis["sentiment"]["compound"] * 0.5
    feature_score = 0.3 * (1 - abs(target_valence - song["features"]["valence"]))
    
    # Context matching (30%)
    context_score = 0.3 * (0.5 if "happy" in song["tags"] and analysis["sentiment"]["pos"] > 0.5 else 0.2)
    
    return semantic_sim + feature_score + context_score

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

'''

'''

import os
import logging
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from typing import Dict, List, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Music Recommendation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize ML models
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller model
    sentiment_analyzer = SentimentIntensityAnalyzer()
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise

# Pydantic models for request/response validation
class RecommendationRequest(BaseModel):
    user_id: str
    text: str
    context: Optional[Dict[str, Any]] = None

class SongRecommendation(BaseModel):
    song_id: str
    title: str
    artist: str
    match_score: float
    features: Dict[str, float]

class RecommendationResponse(BaseModel):
    status: str
    recommendations: List[SongRecommendation]
    analysis: Dict[str, Any]

# Database dependency
async def get_async_session() -> AsyncSession:
    """Override this with your actual database session setup"""
    raise NotImplementedError("Database session factory not implemented")

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_songs(request: RecommendationRequest, db: AsyncSession = Depends(get_async_session)):
    """
    Generate personalized song recommendations based on:
    - Text semantic analysis
    - User preferences
    - Contextual information
    """
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")

        # Perform text analysis
        analysis = await analyze_text(request.text)
        
        # Generate recommendations (mock implementation - replace with your actual logic)
        recommendations = [
            {
                "song_id": "123",
                "title": "Happy Days",
                "artist": "Sunshine Band",
                "match_score": 0.87,
                "features": {
                    "valence": 0.8,
                    "energy": 0.7,
                    "tempo": 120
                }
            }
        ]
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "analysis": {
                "sentiment": analysis["sentiment"],
                "key_phrases": analysis["key_phrases"],
                "detected_activities": analysis["activities"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Recommendation service error")

async def analyze_text(text: str) -> Dict[str, Any]:
    """Perform comprehensive text analysis"""
    doc = nlp(text)
    
    # Get semantic embedding
    embedding = sentence_model.encode(text)
    
    # Analyze sentiment
    sentiment = sentiment_analyzer.polarity_scores(text)
    
    # Extract key information
    activities = detect_activities(doc)
    temporal_context = detect_temporal_context(doc)
    key_phrases = extract_key_phrases(doc)
    
    return {
        "embedding": embedding.tolist(),
        "sentiment": sentiment,
        "activities": activities,
        "temporal_context": temporal_context,
        "key_phrases": key_phrases
    }

def detect_activities(doc) -> List[str]:
    """Detect activities from text"""
    activity_keywords = {
        'working': ['work', 'code', 'program'],
        'exercising': ['run', 'workout', 'gym'],
        'relaxing': ['relax', 'chill', 'rest'],
        'driving': ['drive', 'commute'],
        'partying': ['party', 'celebrate']
    }
    return [
        activity for activity, keywords in activity_keywords.items()
        if any(token.lemma_ in keywords for token in doc)
    ]

def detect_temporal_context(doc) -> Optional[str]:
    """Detect time references in text"""
    time_phrases = ['morning', 'afternoon', 'evening', 'night']
    for token in doc:
        if token.lemma_ in time_phrases:
            return token.lemma_
    return None

def extract_key_phrases(doc) -> List[str]:
    """Extract meaningful phrases from text"""
    return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

def calculate_song_score(song: Dict, analysis: Dict) -> float:
    """Calculate recommendation score (0-1)"""
    # Semantic similarity (40%)
    semantic_sim = cosine_similarity(
        [analysis["embedding"]],
        [song["embedding"]]
    )[0][0] * 0.4
    
    # Feature matching (30%)
    target_valence = 0.5 + analysis["sentiment"]["compound"] * 0.5
    feature_score = 0.3 * (1 - abs(target_valence - song["features"]["valence"]))
    
    # Context matching (30%)
    context_score = 0.3 * (0.5 if "happy" in song["tags"] and analysis["sentiment"]["pos"] > 0.5 else 0.2)
    
    return semantic_sim + feature_score + context_score

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

'''












'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from fastapi.middleware.cors import CORSMiddleware
import re
from collections import defaultdict
from typing import Any

# Initialize FastAPI app
app = FastAPI(title="Simplified Music Recommendation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock database of songs
MUSIC_DATABASE = [
    {
        "song_id": "1",
        "title": "Happy Days",
        "artist": "Sunshine Band",
        "embedding": np.random.rand(384).tolist(),  # Mock embedding
        "features": {
            "valence": 0.8,
            "energy": 0.7,
            "tempo": 120
        },
        "tags": ["happy", "uplifting", "summer"]
    },
    {
        "song_id": "2",
        "title": "Relaxing Waves",
        "artist": "Ocean Sounds",
        "embedding": np.random.rand(384).tolist(),
        "features": {
            "valence": 0.4,
            "energy": 0.3,
            "tempo": 80
        },
        "tags": ["calm", "peaceful", "nature"]
    }
]

# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: str
    text: str
    context: Optional[Dict[str, Any]] = None

class SongRecommendation(BaseModel):
    song_id: str
    title: str
    artist: str
    match_score: float
    features: Dict[str, float]

class RecommendationResponse(BaseModel):
    status: str
    recommendations: List[SongRecommendation]
    analysis: Dict[str, Any]

# Simple text analysis functions
def analyze_sentiment(text: str) -> Dict[str, float]:
    """Simple sentiment analysis without external dependencies"""
    positive_words = {"happy", "joy", "love", "great", "awesome"}
    negative_words = {"sad", "angry", "hate", "terrible", "awful"}
    
    words = re.findall(r'\w+', text.lower())
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    total = len(words)
    
    return {
        "positive": pos_count / total if total > 0 else 0,
        "negative": neg_count / total if total > 0 else 0,
        "compound": (pos_count - neg_count) / total if total > 0 else 0
    }

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords"""
    words = re.findall(r'\w+', text.lower())
    stopwords = {"i", "me", "my", "myself", "we", "our", "the", "a", "an"}
    return [word for word in words if word not in stopwords and len(word) > 2]

def detect_activities(text: str) -> List[str]:
    """Detect activities from text"""
    activity_map = {
        'working': ['work', 'code', 'program', 'job'],
        'exercising': ['run', 'workout', 'gym', 'exercise'],
        'relaxing': ['relax', 'chill', 'rest', 'sleep'],
        'driving': ['drive', 'commute', 'road'],
        'partying': ['party', 'celebrate', 'dance']
    }
    
    keywords = extract_keywords(text)
    return [
        activity for activity, terms in activity_map.items()
        if any(term in keywords for term in terms)
    ]

# Recommendation endpoint
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_songs(request: RecommendationRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")

        # Simple text analysis
        sentiment = analyze_sentiment(request.text)
        activities = detect_activities(request.text)
        keywords = extract_keywords(request.text)
        
        # Mock embedding for the input text (in a real app, you'd use a model)
        text_embedding = np.random.rand(384)
        
        # Generate recommendations by comparing with our mock database
        recommendations = []
        for song in MUSIC_DATABASE:
            # Calculate similarity score (simple version)
            similarity = cosine_similarity(
                [text_embedding],
                [np.array(song["embedding"])]
            )[0][0]
            
            # Adjust score based on sentiment match
            valence_diff = abs(sentiment["compound"] - song["features"]["valence"])
            score = similarity * 0.7 + (1 - valence_diff) * 0.3
            
            recommendations.append({
                "song_id": song["song_id"],
                "title": song["title"],
                "artist": song["artist"],
                "match_score": min(max(score, 0), 1),  # Ensure between 0-1
                "features": song["features"]
            })
        
        # Sort by best match
        recommendations.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {
            "status": "success",
            "recommendations": recommendations[:5],  # Return top 5
            "analysis": {
                "sentiment": sentiment,
                "keywords": keywords,
                "detected_activities": activities
            }
        }
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


'''





# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from recommendation_module import MusicRecommender

# FastAPI app instance
app = FastAPI(title="Chat-Based Music Recommender")

# CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: set to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input from frontend
class RecommendRequest(BaseModel):
    user_id: int
    text: str  # User's current message or description

# Song structure
class Track(BaseModel):
    id: str
    uri: str
    name: str
    artists: List[Dict]
    album: Dict
    duration_ms: int
    popularity: int
    preview_url: Optional[str] = None
    features: Optional[Dict] = {}
    genres: Optional[List[str]] = []

# Output to frontend
class RecommendResponse(BaseModel):
    status: str
    recommendations: List[Track]

@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendation(req: RecommendRequest):
    try:
        engine = MusicRecommender()
        tracks = await engine.recommend_songs(user_id=req.user_id, user_input=req.text)
        
        return {
            "status": "success",
            "recommendations": tracks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

# Optional: for running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

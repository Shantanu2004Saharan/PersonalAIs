import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Set up basic logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This class handles converting text to numbers (embeddings)
class TextToNumbers:
    def __init__(self):
        # Using a pre-trained model that converts sentences to number vectors
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Store embeddings we've already calculated to save time
        self.cache = {}
    
    def get_numbers(self, text):
        """Convert text to a list of numbers (vector)"""
        if not text:
            raise ValueError("Need some text to convert!")
        
        # Check if we've done this text before
        if text in self.cache:
            logger.info("Found this text in cache")
            return self.cache[text]
        
        try:
            # Actually convert the text to numbers
            numbers = self.model.encode(text)
            self.cache[text] = numbers
            logger.info(f"Converted text to numbers: {text[:30]}...")
            return numbers
        except Exception as e:
            logger.error(f"Oops, couldn't convert text: {e}")
            raise

# This class figures out emotions in text
class MoodDetector:
    def __init__(self):
        # Using a special model trained to detect emotions
        self.model = pipeline(
            "text-classification",
            model="joeddav/distilbert-base-uncased-go-emotions-student",
            return_all_scores=True
        )
        
        # How different emotions translate to music features
        self.emotion_map = {
            'happy': {'valence': 0.9, 'energy': 0.8, 'danceability': 0.7},
            'sad': {'valence': 0.2, 'energy': 0.4, 'danceability': 0.3},
            # ... (other emotions would go here)
        }
    
    def detect_mood(self, text):
        """Figure out what emotions are in the text"""
        if not text:
            return {'error': 'No text provided'}
            
        try:
            # Get all possible emotions and their scores
            emotion_results = self.model(text)[0]
            
            # Find the strongest emotion
            main_emotion = max(emotion_results, key=lambda x: x['score'])['label']
            
            # Convert emotions to music preferences
            music_profile = self._get_music_profile(emotion_results)
            
            return {
                'emotions': {e['label']: e['score'] for e in emotion_results},
                'main_emotion': main_emotion,
                'music_profile': music_profile
            }
        except Exception as e:
            logger.error(f"Couldn't detect mood: {e}")
            return {'error': str(e)}
    
    def _get_music_profile(self, emotions):
        """Convert emotions to music features like danceability"""
        profile = {
            'valence': 0.5,  # How positive/negative
            'energy': 0.5,   # How energetic
            'danceability': 0.5  # How danceable
        }
        
        # Adjust the profile based on detected emotions
        for emotion in emotions:
            if emotion['label'] in self.emotion_map:
                for feature in profile:
                    profile[feature] += emotion['score'] * self.emotion_map[emotion['label']].get(feature, 0)
        
        # Make sure values stay between 0 and 1
        return {k: max(0, min(1, v)) for k, v in profile.items()}

# Handles talking to Spotify
class SpotifyHelper:
    def __init__(self):
        # Set up connection to Spotify
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id="your-client-id",
            client_secret="your-client-secret",
            redirect_uri="http://localhost:8000/callback",
            scope="user-library-read playlist-modify-public"
        ))
        self.user_id = self.sp.current_user()["id"]
    
    def find_songs(self, search_term, limit=10):
        """Search for songs on Spotify"""
        results = self.sp.search(q=search_term, type="track", limit=limit)
        songs = []
        
        for track in results.get('tracks', {}).get('items', []):
            songs.append({
                'id': track['id'],
                'name': track['name'],
                'artists': [a['name'] for a in track['artists']],
                'preview_url': track['preview_url'],
                'spotify_link': track['external_urls']['spotify']
            })
        
        return songs
    
    def create_playlist(self, name, track_ids):
        """Make a new playlist with given songs"""
        playlist = self.sp.user_playlist_create(
            user=self.user_id,
            name=name,
            public=True
        )
        self.sp.playlist_add_items(playlist['id'], track_ids)
        return playlist['external_urls']['spotify']
    
class SongIndex:
    def __init__(self):
        pass

    def search(self, query, top_k=10):
        return []

# The main brain of the app
class MusicRecommender:
    def __init__(self):
        self.spotify = SpotifyHelper()
        self.text_converter = TextToNumbers()
        self.mood_detector = MoodDetector()
    
    def get_recommendations(self, search_term, mood=None):
        """Get song recommendations based on search and mood"""
        # First get songs from Spotify
        songs = self.spotify.find_songs(search_term, limit=50)
        
        # If we have mood preferences, filter songs
        if mood:
            songs = [s for s in songs if self._matches_mood(s['id'], mood)]
        
        # Convert song info to numbers for comparison
        song_numbers = {
            s['id']: self.text_converter.get_numbers(f"{s['name']} {', '.join(s['artists'])}")
            for s in songs
        }
        
        # Convert search term to numbers
        search_numbers = self.text_converter.get_numbers(search_term)
        
        # Calculate how similar each song is to the search
        similarities = []
        for song in songs:
            similarity = cosine_similarity(
                [search_numbers],
                [song_numbers[song['id']]]
            )[0][0]
            similarities.append((song, similarity))
        
        # Sort by similarity and return top 10
        return [song for song, _ in sorted(similarities, key=lambda x: x[1], reverse=True)[:10]]
    
    def _matches_mood(self, song_id, mood):
        """Check if a song matches the desired mood"""
        features = self.spotify.get_audio_features(song_id)
        return all(
            abs(features.get(k, 0.5) - v) <= 0.2
            for k, v in mood.items()
        )

# The web interface using Streamlit
def run_app():
    st.set_page_config(page_title="🎵 Music Buddy", layout="centered")
    st.title("🎧 Your Personal Music Assistant")
    st.write("Find perfect songs for your mood!")
    
    recommender = MusicRecommender()
    
    # Create tabs for different features
    tab1, tab2 = st.tabs(["Search Songs", "Mood Analysis"])
    
    with tab1:
        search = st.text_input("What kind of music do you want?")
        
        st.write("Adjust these to match your mood:")
        mood = {
            "valence": st.slider("Positivity", 0.0, 1.0, 0.5),
            "energy": st.slider("Energy", 0.0, 1.0, 0.5),
            "danceability": st.slider("Danceability", 0.0, 1.0, 0.5),
        }
        
        if st.button("Find Songs"):
            with st.spinner("Searching for perfect songs..."):
                songs = recommender.get_recommendations(search, mood)
                show_songs(songs, recommender)
    
    with tab2:
        feelings = st.text_area("How are you feeling today?")
        
        if st.button("Analyze My Mood"):
            with st.spinner("Understanding your mood..."):
                mood_result = recommender.mood_detector.detect_mood(feelings)
                
                st.subheader("Your Mood")
                st.write(f"Main emotion: **{mood_result['main_emotion']}**")
                
                st.subheader("Suggested Music Settings")
                for name, value in mood_result['music_profile'].items():
                    st.write(f"{name}: {value:.2f}")
                
                st.subheader("Recommended Songs")
                songs = recommender.get_recommendations(
                    mood_result['main_emotion'],
                    mood_result['music_profile']
                )
                show_songs(songs, recommender)

# Helper function to display songs
def show_songs(songs, recommender):
    if songs:
        st.success("Here are your recommendations!")
        
        for i, song in enumerate(songs, 1):
            st.markdown(f"**{i}. {song['name']}** — {', '.join(song['artists'])}")
            
            if song.get('preview_url'):
                st.audio(song['preview_url'], format='audio/mp3')
            
            st.markdown(f"[🔗 Open in Spotify]({song['spotify_link']})")
        
        if st.button("💿 Save as Playlist"):
            playlist_url = recommender.spotify.create_playlist(
                "My Music Buddy Playlist",
                [s['id'] for s in songs]
            )
            st.markdown(f"✅ Playlist created! [Open it here]({playlist_url})")
    else:
        st.error("Couldn't find any matching songs. Try different search terms!")

# Run the app
if __name__ == "__main__":
    run_app()

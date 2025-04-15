import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from joblib import Memory
import os

# Setup Spotify authentication
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="your_client_id", client_secret="your_client_secret"))

# Setup caching
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@memory.cache
def get_tracks_by_query(query):
    """
    Fetch tracks based on a query string.
    """
    try:
        results = sp.search(q=query, limit=10, type='track')
        tracks = results['tracks']['items']
        logging.info(f"Fetched {len(tracks)} tracks for query: {query}")
        return tracks
    except Exception as e:
        logging.error(f"Error fetching tracks for query '{query}': {e}")
        return []

@memory.cache
def get_audio_features(track_ids):
    """
    Fetch audio features for a list of track IDs (e.g., energy, valence, tempo).
    """
    try:
        features = sp.audio_features(track_ids)
        logging.info(f"Fetched audio features for {len(features)} tracks.")
        return features
    except Exception as e:
        logging.error(f"Error fetching audio features: {e}")
        return []

def filter_tracks_by_audio_features(tracks, features, mood_preferences):
    """
    Filter tracks based on audio features like 'energy' and 'valence'.
    mood_preferences example: {'energy': 0.7, 'valence': 0.6}
    """
    filtered_tracks = []
    for track, feature in zip(tracks, features):
        if feature:
            energy_match = mood_preferences.get('energy', 0.5)
            valence_match = mood_preferences.get('valence', 0.5)
            if (feature['energy'] >= energy_match) and (feature['valence'] >= valence_match):
                filtered_tracks.append(track)
    logging.info(f"Filtered down to {len(filtered_tracks)} tracks based on audio features.")
    return filtered_tracks

def get_track_details(track_id):
    """
    Fetch detailed metadata for a given track using its Spotify ID.
    """
    try:
        track = sp.track(track_id)
        logging.info(f"Fetched details for track: {track['name']}")
        return track
    except Exception as e:
        logging.error(f"Error fetching track details for ID '{track_id}': {e}")
        return {}

# Example usage:
# 1. Query for tracks
query = "chill instrumental"
tracks = get_tracks_by_query(query)

# 2. Get audio features for tracks
track_ids = [track['id'] for track in tracks]
features = get_audio_features(track_ids)

# 3. Filter tracks based on mood preferences (e.g., energy and valence)
mood_preferences = {'energy': 0.6, 'valence': 0.7}
filtered_tracks = filter_tracks_by_audio_features(tracks, features, mood_preferences)

# 4. Retrieve full track details if needed
for track in filtered_tracks:
    details = get_track_details(track['id'])
    print(details)

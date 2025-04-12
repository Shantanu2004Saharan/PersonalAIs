import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API credentials - REPLACE THESE WITH YOUR ACTUAL CREDENTIALS!
SPOTIPY_CLIENT_ID = "your_client_id_here"  # Replace with your actual client ID
SPOTIPY_CLIENT_SECRET = "your_client_secret_here"  # Replace with your actual client secret
SPOTIPY_REDIRECT_URI = "http://localhost:8888/callback"  # Can keep this default

# Get version
try:
    from importlib.metadata import version
    print(f"Spotipy version: {version('spotipy')}")
except:
    import pkg_resources
    print(f"Spotipy version: {pkg_resources.get_distribution('spotipy').version}")

# Test API connection
try:
    auth_manager = SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)

    results = sp.search(q='artist:Radiohead track:Karma Police', type='track')
    track = results['tracks']['items'][0]
    print(f"\nTest successful! Found: {track['name']} by {track['artists'][0]['name']}")
except Exception as e:
    print(f"\nError: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Make sure you've replaced the placeholder credentials with your actual Spotify API credentials")
    print("2. Verify your credentials in the Spotify Developer Dashboard")
    print("3. Check that your Spotify app has the correct redirect URI set")


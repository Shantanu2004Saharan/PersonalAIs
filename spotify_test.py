import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Get version
try:
    from importlib.metadata import version
    print(f"Spotipy version: {version('spotipy')}")
except:
    import pkg_resources
    print(f"Spotipy version: {pkg_resources.get_distribution('spotipy').version}")

# Test API connection
auth_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(auth_manager=auth_manager)

results = sp.search(q='artist:Radiohead track:Karma Police', type='track')
track = results['tracks']['items'][0]
print(f"\nTest successful! Found: {track['name']} by {track['artists'][0]['name']}")

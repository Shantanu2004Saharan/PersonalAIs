import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = "736bb144677e448dad56d2fe2ab70cd0"
client_secret = "d7beffe6e8d740deb7e1ddd9a111c88f"

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

track_id = "3n3Ppam7vgaVa1iaRUc9Lp"  # a known working ID (Eminem – Without Me)

try:
    features = sp.audio_features([track_id])
    print("✅ Spotify access success!")
    print(features)
except Exception as e:
    print("❌ Spotify auth failed:", e)

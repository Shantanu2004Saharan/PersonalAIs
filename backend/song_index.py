from spotify import SpotifyClient, search_spotify_tracks, get_audio_features
from embedding_utils import get_embedding as embed_text
from spotify import SpotifyClient
from embedding_utils import load_embedder

embedder = load_embedder()
embedding = embedder.encode("chill beats to relax/study to")

# Optional: FAISS for persistent vector search
try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False

class SongIndexer:
    def __init__(self):
        self.client = SpotifyClient()  # instantiate SpotifyClient
        self.songs = []

class SongIndex:  # Make sure class name matches in both file and import
    def __init__(self):
        self.client = SpotifyClient()  # ✅ Add this line
        self.songs = []
        self.index = None
        self.embedder = load_embedder() # SBERT embeddings are 384 dim

    def build_index(self, query, mood_filter=None):
        """
        Build the song index using Spotify search results and embeddings.
        Optional mood filter applies mood-based audio features.
        """
        self.songs = search_spotify_tracks(self.client, query)
        self.embeddings = []

        for song in self.songs:
            text = f"{song['name']} {song['artist']}"
            embedding = embed_text(text)
            self.embeddings.append(embedding)

        if USE_FAISS:
            import numpy as np
            self.index.add(np.array(self.embeddings).astype('float32'))

        if mood_filter:
            self.songs = self.filter_by_mood(self.songs, mood_filter)

    def filter_by_mood(self, songs, mood):
        """
        Filter songs by mood based on energy and valence ranges.
        """
        energy_range, valence_range = {
            'happy':   ((0.6, 1.0), (0.6, 1.0)),
            'sad':     ((0.0, 0.4), (0.0, 0.4)),
            'energetic': ((0.7, 1.0), (0.4, 1.0)),
            'calm':    ((0.0, 0.4), (0.4, 1.0))
        }.get(mood, ((0.0, 1.0), (0.0, 1.0)))  # Default ranges for any mood

        filtered = []
        for song in songs:
            features = get_audio_features(song['id'])
            if features and energy_range[0] <= features['energy'] <= energy_range[1] \
                and valence_range[0] <= features['valence'] <= valence_range[1]:
                filtered.append(song)

        return filtered

    def get_top_k(self, user_embedding, k=10):
        """
        Return the top K most similar songs based on user embedding.
        Uses FAISS for efficient search or cosine similarity if FAISS is unavailable.
        """
        import numpy as np
        if USE_FAISS:
            D, I = self.index.search(np.array([user_embedding]).astype('float32'), k)
            return [self.songs[i] for i in I[0]]
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity([user_embedding], self.embeddings)[0]
            ranked = sorted(zip(self.songs, scores), key=lambda x: x[1], reverse=True)
            return [s for s, _ in ranked[:k]]

if __name__ == "__main__":
    print("Building song index...")
    si = SongIndex()
    si.build_index("lofi", mood_filter="calm")
    
    user_query = "chill relaxing beats"
    from embedding_utils import embed_text
    user_embedding = embed_text(user_query)

    print("\nTop 5 recommended tracks:")
    for i, song in enumerate(si.get_top_k(user_embedding, k=5), 1):
        print(f"{i}. {song['name']} by {song['artist']}")

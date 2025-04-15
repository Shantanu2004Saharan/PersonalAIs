# song_index.py

from embedding_utils import embed_text
from spotify import search_spotify_tracks, get_audio_features

# Optional: FAISS for persistent vector search
try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False

class SongIndex:
    def __init__(self):
        self.songs = []
        self.embeddings = []
        if USE_FAISS:
            self.index = faiss.IndexFlatL2(384)  # SBERT embeddings are 384 dim

    def build_index(self, query, mood_filter=None):
        self.songs = search_spotify_tracks(query)
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
        energy_range, valence_range = {
            'happy':   ((0.6, 1.0), (0.6, 1.0)),
            'sad':     ((0.0, 0.4), (0.0, 0.4)),
            'energetic': ((0.7, 1.0), (0.4, 1.0)),
            'calm':    ((0.0, 0.4), (0.4, 1.0))
        }.get(mood, ((0.0, 1.0), (0.0, 1.0)))

        filtered = []
        for song in songs:
            features = get_audio_features(song['id'])
            if features and energy_range[0] <= features['energy'] <= energy_range[1] \
                and valence_range[0] <= features['valence'] <= valence_range[1]:
                filtered.append(song)

        return filtered

    def get_top_k(self, user_embedding, k=10):
        import numpy as np
        if USE_FAISS:
            D, I = self.index.search(np.array([user_embedding]).astype('float32'), k)
            return [self.songs[i] for i in I[0]]
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity([user_embedding], self.embeddings)[0]
            ranked = sorted(zip(self.songs, scores), key=lambda x: x[1], reverse=True)
            return [s for s, _ in ranked[:k]]
                
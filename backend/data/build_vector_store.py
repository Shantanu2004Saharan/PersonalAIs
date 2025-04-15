import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# File paths
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(DATA_DIR, "track_embeddings.npy")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "track_metadata.json")

# Dummy track data: replace this with real Spotify data
track_metadata = [
    {
        "id": "1",
        "name": "Relaxing Piano",
        "artists": ["Calm Artist"],
        "preview_url": None,
        "external_url": "https://open.spotify.com/track/1",
        "features": {"energy": 0.2, "valence": 0.4}
    },
    {
        "id": "2",
        "name": "Energetic EDM",
        "artists": ["DJ Hype"],
        "preview_url": None,
        "external_url": "https://open.spotify.com/track/2",
        "features": {"energy": 0.9, "valence": 0.8}
    },
    {
        "id": "3",
        "name": "Melancholic Guitar",
        "artists": ["Sad Strings"],
        "preview_url": None,
        "external_url": "https://open.spotify.com/track/3",
        "features": {"energy": 0.3, "valence": 0.2}
    }
]

# Prepare text for embedding
texts = [
    f"{track['name']} by {', '.join(track['artists'])}. Mood: energy {track['features']['energy']}, valence {track['features']['valence']}"
    for track in track_metadata
]

# Load model and generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Save embeddings
np.save(EMBEDDINGS_PATH, embeddings)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, INDEX_PATH)

# Save metadata
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(track_metadata, f, indent=2)

print("✅ Vector store built and saved to /data.")

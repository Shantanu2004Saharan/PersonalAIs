# emotion_embedding.py

from sentence_transformers import SentenceTransformer
import numpy as np

class EmotionEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_emotions(self, emotion_phrases):
        """
        Given a list of phrases (e.g., ["I'm feeling happy", "I am so angry"]),
        return their vector embeddings.
        """
        return self.model.encode(emotion_phrases, show_progress_bar=False)

    def similarity(self, vec1, vec2):
        """
        Cosine similarity between two embeddings.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Example usage
if __name__ == "__main__":
    embedder = EmotionEmbedder()
    phrases = ["I feel joyful", "I am very sad"]
    embeddings = embedder.encode_emotions(phrases)

    sim = embedder.similarity(embeddings[0], embeddings[1])
    print(f"Similarity between phrases: {sim:.4f}")

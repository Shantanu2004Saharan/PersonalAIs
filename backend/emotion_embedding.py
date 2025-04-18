from sentence_transformers import SentenceTransformer
import numpy as np

class EmotionEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode_emotions(self, emotion_phrases):
        return self.model.encode(emotion_phrases, show_progress_bar=False)

    def similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Example usage and simple tests
if __name__ == "__main__":
    embedder = EmotionEmbedder()

    # Test 1: Check embedding shape
    phrases = ["I feel joyful", "I am very sad"]
    embeddings = embedder.encode_emotions(phrases)
    assert len(embeddings) == 2, "Should return two embeddings"
    assert embeddings[0].shape == embeddings[1].shape, "Embeddings should have same shape"
    print("✅ Test 1 passed: Embedding shape is consistent.")

    # Test 2: Similarity of same phrase should be close to 1
    vec = embedder.encode_emotions(["I am happy", "I am happy"])
    sim = embedder.similarity(vec[0], vec[1])
    assert sim > 0.95, f"Expected high similarity, got {sim:.4f}"
    print(f"✅ Test 2 passed: Similarity of identical phrases is {sim:.4f}")

    # Test 3: Similarity of different emotions should be lower
    vec = embedder.encode_emotions(["I am extremely happy", "I am furious"])
    sim = embedder.similarity(vec[0], vec[1])
    assert sim < 0.7, f"Expected low similarity, got {sim:.4f}"
    print(f"✅ Test 3 passed: Dissimilar phrases similarity is {sim:.4f}")

    print("🎉 All tests passed!")

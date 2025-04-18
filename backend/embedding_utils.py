# backend/embedding_utils.py

import os
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding cache
embedding_cache = {}

# Embedder instance (lazy-loaded)
_embedder = None

def load_embedder(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """Load and return the SentenceTransformer embedder."""
    global _embedder
    if _embedder is None:
        try:
            _embedder = SentenceTransformer(model_name)
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return _embedder

def get_embedding(text: str) -> np.ndarray:
    """Generate embedding for given text using a local SentenceTransformer."""
    if not text:
        raise ValueError("Input text must not be empty.")

    if text in embedding_cache:
        logger.info("Using cached embedding.")
        return embedding_cache[text]

    try:
        embedder = load_embedder()
        embedding = embedder.encode(text)
        embedding_cache[text] = embedding
        logger.info(f"Embedding generated for: {text[:30]}...")
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed for '{text[:30]}...': {e}")
        return None

# Example usage or tests
if __name__ == "__main__":
    test_texts = ["Hello, this is a test sentence...", "Test text 1", "Test text 2", "Test text 3"]

    logger.info("Running embedding tests...")

    # Test 1: Single embedding
    embedding = get_embedding(test_texts[0])
    assert embedding is not None, "Test 1 failed: No embedding returned"
    logger.info("Test 1 passed: Embedding generated successfully.")

    # Test 2: Caching
    embedding2 = get_embedding(test_texts[0])
    assert np.array_equal(embedding, embedding2), "Test 2 failed: Embeddings differ"
    logger.info("Test 2 passed: Caching works correctly.")

    # Test 3: Multiple embeddings
    for text in test_texts:
        emb = get_embedding(text)
        assert emb is not None, f"Test 3 failed for text: {text}"
    logger.info("Test 3 passed: Multiple embeddings generated successfully.")

import os
import hashlib
import logging
import pickle
import openai
import numpy as np
from typing import List
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

CACHE_DIR = ".cache_embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(text: str) -> str:
    hash_key = hashlib.md5(text.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_key}.pkl")

def embed_text(text: str) -> np.ndarray:
    """
    Returns the OpenAI embedding of a given text with caching and logging.
    """
    cache_file = _cache_path(text)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            logger.info(f"Using cached embedding for text: {text[:30]}...")
            return pickle.load(f)

    try:
        logger.info(f"Generating embedding for text: {text[:30]}...")
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = np.array(response['data'][0]['embedding'])
        with open(cache_file, "wb") as f:
            pickle.dump(embedding, f)
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed for '{text[:30]}...': {str(e)}")
        return np.zeros(1536)

def embed_texts(texts: List[str]) -> List[np.ndarray]:
    return [embed_text(text) for text in texts]
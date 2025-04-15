import faiss
import numpy as np
import os
import pickle
from typing import List

class IndexStore:
    def __init__(self, dim: int, index_file="song_index.faiss", meta_file="metadata.pkl"):
        self.dim = dim
        self.index_file = index_file
        self.meta_file = meta_file
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

        if os.path.exists(index_file) and os.path.exists(meta_file):
            self.index = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)

    def add(self, vectors: List[np.ndarray], metas: List[dict]):
        self.index.add(np.array(vectors).astype("float32"))
        self.metadata.extend(metas)
        self.save()

    def search(self, vector: np.ndarray, k=10):
        D, I = self.index.search(np.array([vector]).astype("float32"), k)
        return [self.metadata[i] for i in I[0]]

    def save(self):
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadata, f)


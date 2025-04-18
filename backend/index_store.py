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
            print(f"Loading existing files: {index_file} and {meta_file}")
            self.index = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"Loaded metadata with {len(self.metadata)} entries.")

    def add(self, vectors: List[np.ndarray], metas: List[dict]):
        print(f"Adding {len(vectors)} vectors and {len(metas)} metadata entries")
        self.index.add(np.array(vectors).astype("float32"))
        print(f"Current metadata length before adding: {len(self.metadata)}")
        self.metadata.extend(metas)
        print(f"Current metadata length after adding: {len(self.metadata)}")
        self.save()

    def search(self, vector: np.ndarray, k=10):
        D, I = self.index.search(np.array([vector]).astype("float32"), k)
        return [self.metadata[i] for i in I[0]]

    def save(self):
        print("Saving index and metadata...")
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

        if os.path.exists(self.index_file):
            print(f"Index file saved: {self.index_file}")
        if os.path.exists(self.meta_file):
            print(f"Metadata file saved: {self.meta_file}")

    def run_tests(self):
        print("Resetting existing test files...")

        # Clean up any existing test files before running tests
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.meta_file):
            os.remove(self.meta_file)

        # Reinitialize a fresh index and metadata
        self.index = faiss.IndexFlatL2(self.dim)
        self.metadata = []

        print("Running test 1: Adding vectors and metadata...")

        vectors = [np.random.random(self.dim) for _ in range(5)]
        metadata = [{"track": f"track{i}", "artist": f"artist{i}"} for i in range(5)]

        self.add(vectors, metadata)

        print(f"Metadata length after adding: {len(self.metadata)}")
        assert len(self.metadata) == 5, f"Test 1 failed: Metadata length should be 5, but got {len(self.metadata)}"
        print("Test 1 passed.")

        print("Running test 2: Searching vectors...")

        search_vector = vectors[0]
        results = self.search(search_vector, k=3)

        assert len(results) == 3, f"Test 2 failed: Expected 3 results, but got {len(results)}"
        assert results[0]["track"] == "track0", "Test 2 failed: Incorrect metadata returned"
        print("Test 2 passed.")

        print("Running test 3: Saving and reloading index...")

        self.save()

        assert os.path.exists(self.index_file), f"Index file {self.index_file} does not exist"
        assert os.path.exists(self.meta_file), f"Metadata file {self.meta_file} does not exist"

        new_index_store = IndexStore(self.dim, self.index_file, self.meta_file)

        print(f"Metadata length after reloading: {len(new_index_store.metadata)}")
        assert len(new_index_store.metadata) == 5, f"Test 3 failed: Metadata length should be 5, but got {len(new_index_store.metadata)}"
        print("Test 3 passed.")

        os.remove(self.index_file)
        os.remove(self.meta_file)
        print("All tests passed!")


# Example usage
if __name__ == "__main__":
    index_store = IndexStore(dim=128, index_file="test_song_index.faiss", meta_file="test_metadata.pkl")
    index_store.run_tests()


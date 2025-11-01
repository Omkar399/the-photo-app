"""FAISS index management and utilities."""

import faiss
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FAISSManager:
    """Manager for FAISS index operations."""

    def __init__(
        self,
        index_dir: Path,
        nlist: int = 4096,
        m: int = 64,
        nbits: int = 8,
        nprobe: int = 8,
    ):
        """
        Initialize FAISS manager.

        Args:
            index_dir: Directory to store index files
            nlist: Number of centroids (coarse quantizer)
            m: Number of sub-quantizers
            nbits: Bits per sub-quantizer
            nprobe: Number of cells to probe during search
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe

        self.index_path = self.index_dir / "img.index"
        self.idmap_path = self.index_dir / "idmap.json"

        self.index = None
        self.id_map = {}  # index_row -> image_id

    def build_index(self, vectors: np.ndarray, embedding_dim: int) -> None:
        """
        Build FAISS index from vectors - adaptive based on dataset size.

        Args:
            vectors: Vector matrix (N x D, float32)
            embedding_dim: Embedding dimension (may be used as hint, actual dim from vectors takes precedence)
        """
        if len(vectors) == 0:
            logger.warning("No vectors to build index")
            return

        n_vectors = len(vectors)
        vectors = np.asarray(vectors, dtype=np.float32)
        
        # Get actual embedding dimension from vectors, not from parameter
        if vectors.ndim == 1:
            actual_dim = len(vectors)
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim == 2:
            actual_dim = vectors.shape[1]
        else:
            actual_dim = embedding_dim
            vectors = vectors.reshape(n_vectors, -1)
        
        logger.info(f"Building index: {n_vectors} vectors")
        logger.info(f"Vectors shape: {vectors.shape}, dtype: {vectors.dtype}")
        logger.info(f"Using actual embedding dim from vectors: {actual_dim}")

        # Adaptive index selection based on dataset size
        if n_vectors < 100000:
            # For small datasets, use simple flat L2 index (exact search)
            logger.info(f"Dataset size {n_vectors} < 100k, using IndexFlatL2 (exact search)")
            self.index = faiss.IndexFlatL2(actual_dim)
            logger.info(f"Index created with dimension: {self.index.d}")
            self.index.add(vectors)
        else:
            # For larger datasets, use IVF-PQ (approximate search)
            logger.info(f"Dataset size {n_vectors} >= 100k, using IVF-PQ")
            
            # Adaptive nlist based on dataset size
            nlist = max(16, min(self.nlist, n_vectors // 39))  # FAISS rule: nlist << N
            logger.info(f"Adjusted nlist to {nlist} for dataset size")

            quantizer = faiss.IndexFlatL2(actual_dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, actual_dim, nlist, self.m, self.nbits
            )

            # Train on sample
            train_size = min(100000, len(vectors))
            train_vectors = vectors[:train_size]
            logger.info(f"Training on {len(train_vectors)} vectors...")
            self.index.train(train_vectors)

            # Add all vectors
            logger.info("Adding vectors to index...")
            self.index.add(vectors)

            # Set search parameters
            self.index.nprobe = self.nprobe

        logger.info(f"Index built successfully ({self.index.ntotal} vectors, dim={self.index.d})")

    def save_index(self) -> None:
        """Save index and ID map to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return

        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"Saved index to {self.index_path}")

        # Save ID map
        with open(self.idmap_path, "w") as f:
            json.dump(self.id_map, f)
        logger.info(f"Saved ID map to {self.idmap_path}")

    def load_index(self) -> bool:
        """
        Load index and ID map from disk.

        Returns:
            True if successfully loaded, False otherwise
        """
        if not self.index_path.exists() or not self.idmap_path.exists():
            logger.warning(f"Index or ID map not found at {self.index_dir}")
            return False

        try:
            self.index = faiss.read_index(str(self.index_path))
            self.index.nprobe = self.nprobe

            with open(self.idmap_path, "r") as f:
                self.id_map = json.load(f)

            logger.info(f"Loaded index from {self.index_path} ({self.index.ntotal} vectors)")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def add_vectors(self, vectors: np.ndarray, start_id: int = 0) -> None:
        """
        Add vectors to existing index (incremental indexing).

        Args:
            vectors: New vectors (N x D, float32)
            start_id: Starting row ID for mapping
        """
        if self.index is None:
            logger.error("Index not initialized. Call build_index() first")
            return

        logger.info(f"Adding {len(vectors)} vectors to index...")
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            query_vector: Query vector (1 x D or D,, float32)
            k: Number of neighbors to return

        Returns:
            (distances, indices) - distances (1, k), indices (1, k)
        """
        if self.index is None:
            logger.error("Index not loaded")
            return np.array([]), np.array([])

        if query_vector.ndim == 1:
            query_vector = query_vector[np.newaxis, :]

        # Ensure vector is float32
        query_vector = query_vector.astype(np.float32)

        distances, indices = self.index.search(query_vector, k)
        return distances[0], indices[0]

    def batch_search(
        self, query_vectors: np.ndarray, k: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for k nearest neighbors.

        Args:
            query_vectors: Query vectors (N x D, float32)
            k: Number of neighbors per query

        Returns:
            (distances, indices) - distances (N, k), indices (N, k)
        """
        if self.index is None:
            logger.error("Index not loaded")
            return np.array([]), np.array([])

        query_vectors = query_vectors.astype(np.float32)
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices

    def map_indices_to_ids(self, indices: np.ndarray) -> List[Optional[str]]:
        """
        Map FAISS indices to image IDs.

        Args:
            indices: Array of FAISS indices

        Returns:
            List of image IDs
        """
        result = []
        for idx in indices.flat:
            if idx == -1:
                result.append(None)
            else:
                result.append(self.id_map.get(str(idx)))
        return result

    def register_image(self, image_id: str, row_index: int) -> None:
        """Register image ID for a row in the index."""
        self.id_map[str(row_index)] = image_id

    def get_index_size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal if self.index else 0

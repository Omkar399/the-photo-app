"""Vector normalization utilities for embeddings."""

import numpy as np
from typing import Union


def l2_normalize(vec: Union[np.ndarray, list]) -> np.ndarray:
    """
    L2-normalize a vector or batch of vectors.

    Args:
        vec: Vector (1-D) or batch (2-D) array

    Returns:
        L2-normalized vector/batch
    """
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)
    else:
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / (norms + 1e-8)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized vectors.

    For L2-normalized vectors: cosine_sim = dot_product

    Args:
        vec1: First vector (must be L2-normalized)
        vec2: Second vector (must be L2-normalized)

    Returns:
        Similarity score in [-1, 1]
    """
    return float(np.dot(vec1, vec2))


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine distance between two L2-normalized vectors.

    distance = 1 - similarity

    Args:
        vec1: First vector (must be L2-normalized)
        vec2: Second vector (must be L2-normalized)

    Returns:
        Distance in [0, 2]
    """
    return 1.0 - cosine_similarity(vec1, vec2)


def batch_cosine_similarity(
    vec1: np.ndarray, vec2: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarities between batch of vectors.

    Args:
        vec1: Query vector or batch (N x D or 1 x D)
        vec2: Database vectors batch (M x D)

    Returns:
        Similarity matrix (N x M or M,)
    """
    if vec1.ndim == 1:
        vec1 = vec1[np.newaxis, :]

    return np.dot(vec1, vec2.T)  # (N, M)


def batch_cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute cosine distances between batch of vectors.

    Args:
        vec1: Query vector or batch (N x D)
        vec2: Database vectors batch (M x D)

    Returns:
        Distance matrix (N x M)
    """
    return 1.0 - batch_cosine_similarity(vec1, vec2)

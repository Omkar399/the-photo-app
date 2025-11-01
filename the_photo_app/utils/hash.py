"""Image hashing and deduplication utilities."""

import hashlib
import xxhash
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_file_hash(path: Path, algorithm: str = "xxhash") -> str:
    """
    Compute hash of file for deduplication.

    Args:
        path: Path to image file
        algorithm: "xxhash" (fast) or "sha256" (strong)

    Returns:
        Hex hash string
    """
    if algorithm == "xxhash":
        h = xxhash.xxh64()
    elif algorithm == "sha256":
        h = hashlib.sha256()
    else:
        raise ValueError(f"Unknown hash algorithm: {algorithm}")

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def compute_quick_hash(path: Path) -> str:
    """Compute quick hash for initial deduplication (first 1MB)."""
    h = xxhash.xxh64()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))  # First 1MB
    return h.hexdigest()


def generate_image_id(path: Path, use_path: bool = False) -> str:
    """
    Generate unique image ID from path or file hash.

    Args:
        path: Path to image
        use_path: If True, use path-based ID; else use file hash

    Returns:
        Image ID string
    """
    if use_path:
        # Simple path-based ID (deterministic but path-dependent)
        return hashlib.md5(str(path).encode()).hexdigest()[:16]
    else:
        # Hash-based ID (stable across moves)
        return compute_file_hash(path, algorithm="xxhash")[:16]

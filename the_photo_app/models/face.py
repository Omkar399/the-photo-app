"""Face detection and embedding using InsightFace."""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError:
    logger.warning("insightface not installed. Install with: pip install insightface")


class FaceDetector:
    """Face detection and embedding using InsightFace."""

    def __init__(
        self,
        det_name: str = "scrfd_2.5g_bnkps",
        rec_name: str = "arcface_r50",
        device: str = "cuda:0",
    ):
        """
        Initialize face detector with InsightFace.

        Args:
            det_name: Detector model (e.g., "scrfd_2.5g_bnkps")
            rec_name: Recognition/embedding model (e.g., "arcface_r50")
            device: Device to use (e.g., "cuda:0" or "cpu")
        """
        self.det_name = det_name
        self.rec_name = rec_name
        self.device = device
        self.face_app = None
        self.embedding_dim = 512

        self._load_models()

    def _load_models(self) -> None:
        """Load InsightFace models."""
        try:
            self.face_app = FaceAnalysis(name="buffalo_l")
            self.face_app.prepare(ctx_id=0 if "cuda" in self.device else -1, det_size=(640, 640))
            logger.info("Loaded FaceAnalysis models (SCRFD + ArcFace)")
        except Exception as e:
            logger.error(f"Failed to load face models: {e}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image and extract embeddings.

        Args:
            image: RGB image array (HxWxC)

        Returns:
            List of detected faces with embeddings and bboxes
        """
        if image is None:
            return []

        try:
            faces = self.face_app.get(image)
            results = []

            for face in faces:
                result = {
                    "bbox": face.bbox.tolist(),  # [x1, y1, x2, y2]
                    "kps": face.kps.tolist(),  # Keypoints for alignment
                    "det_score": float(face.det_score),  # Detection confidence
                    "embedding": face.embedding.astype(np.float32),  # 512-D
                    "gender": face.gender if hasattr(face, "gender") else None,
                    "age": face.age if hasattr(face, "age") else None,
                }
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def detect_faces_from_path(self, image_path: Path) -> List[dict]:
        """
        Detect faces from image file.

        Args:
            image_path: Path to image file

        Returns:
            List of detected faces with embeddings
        """
        try:
            from PIL import Image

            img = np.array(Image.open(image_path).convert("RGB"))
            return self.detect_faces(img)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return []

    def get_face_embeddings(self, faces: List[dict]) -> np.ndarray:
        """
        Extract embeddings from detected faces.

        Args:
            faces: List of face dicts from detect_faces()

        Returns:
            Array of embeddings (N x 512)
        """
        embeddings = []
        for face in faces:
            embedding = face.get("embedding")
            if embedding is not None:
                # L2-normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32) if embeddings else np.array([])

    def recognize_faces(
        self, probe_embedding: np.ndarray, database_embeddings: np.ndarray, threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Recognize faces by matching probe against database.

        Args:
            probe_embedding: Query face embedding (512-D, L2-normalized)
            database_embeddings: Database face embeddings (N x 512, L2-normalized)
            threshold: Minimum similarity score to consider a match

        Returns:
            List of (index, similarity_score) for matches above threshold
        """
        if len(database_embeddings) == 0:
            return []

        probe_embedding = probe_embedding / (np.linalg.norm(probe_embedding) + 1e-8)
        similarities = np.dot(database_embeddings, probe_embedding)

        matches = [
            (int(idx), float(sim))
            for idx, sim in enumerate(similarities)
            if sim >= threshold
        ]

        return sorted(matches, key=lambda x: x[1], reverse=True)

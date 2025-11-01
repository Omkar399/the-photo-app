"""Model manager for loading and caching models."""

import logging
from typing import Optional
from pathlib import Path

from the_photo_app.config import settings
from the_photo_app.models.embeddings import EmbeddingModel
from the_photo_app.models.face import FaceDetector

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton for managing model instances."""

    _instance = None
    _embedding_model: Optional[EmbeddingModel] = None
    _face_detector: Optional[FaceDetector] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_embedding_model(self) -> EmbeddingModel:
        """Get or initialize embedding model."""
        if self._embedding_model is None:
            device = "cuda" if settings.use_gpu else "cpu"
            self._embedding_model = EmbeddingModel(
                model_name=settings.embedding_model, device=device
            )
            logger.info(
                f"Initialized embedding model: {settings.embedding_model} "
                f"(dim={self._embedding_model.embedding_dim})"
            )
        return self._embedding_model

    def get_face_detector(self) -> FaceDetector:
        """Get or initialize face detector."""
        if self._face_detector is None:
            device = "cuda:0" if settings.use_gpu else "cpu"
            self._face_detector = FaceDetector(
                det_name=settings.face_detector, device=device
            )
            logger.info(
                f"Initialized face detector: {settings.face_detector} "
                f"(embedding_dim={self._face_detector.embedding_dim})"
            )
        return self._face_detector

    def unload_all(self) -> None:
        """Unload all models."""
        self._embedding_model = None
        self._face_detector = None
        logger.info("Unloaded all models")


# Global singleton instance
model_manager = ModelManager()

"""Configuration management using Pydantic settings."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Data & Model directories
    data_dir: Path = Path("./data")
    image_dir: Path = Path("./data/images")
    models_dir: Path = Path("./models")
    db_path: Path = Path("./db/media.db")

    # Embedding model configuration
    embedding_model: str = "siglip_base_256"
    embedding_dim: int = 256

    # Face detection configuration
    face_detector: str = "scrfd_2.5g_bnkps"
    face_embedding_dim: int = 512

    # FAISS index hyperparameters
    faiss_nlist: int = 4096
    faiss_m: int = 64
    faiss_nbits: int = 8
    faiss_nprobe: int = 8

    # Thumbnail generation
    thumb_size: int = 320
    thumb_format: str = "webp"

    # Server configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_port: int = 8501

    # GPU/Compute settings
    use_gpu: bool = True
    batch_size: int = 32
    batch_size_embedding: int = 128
    batch_size_face: int = 16

    # Optional
    solana_keypair_path: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def model_post_init(self, __context):
        """Create necessary directories after initialization."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.data_dir / "thumbs").mkdir(exist_ok=True)
        (self.data_dir / "faiss").mkdir(exist_ok=True)


# Global settings instance
settings = Settings()

"""Main indexing pipeline for photos."""

import time
import json
from pathlib import Path
from typing import Optional, List, Set
import numpy as np
import logging
import argparse

from the_photo_app.config import settings
from the_photo_app.db.schema import Database
from the_photo_app.models.model_manager import model_manager
from the_photo_app.utils.hash import generate_image_id, compute_quick_hash
from the_photo_app.utils.image import load_image_rgb, get_image_dimensions
from the_photo_app.indexing.faiss_manager import FAISSManager
from the_photo_app.indexing.thumbnail_gen import generate_thumbnail

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class PhotoIndexer:
    """Main photo indexing pipeline."""

    def __init__(self):
        """Initialize indexer with models and database."""
        self.db = Database(settings.db_path)
        self.db.init_db()

        self.faiss_mgr = FAISSManager(
            settings.data_dir / "faiss",
            nlist=settings.faiss_nlist,
            m=settings.faiss_m,
            nbits=settings.faiss_nbits,
            nprobe=settings.faiss_nprobe,
        )

        self.embedding_model = model_manager.get_embedding_model()
        self.face_detector = model_manager.get_face_detector()

        self.indexed_hashes: Set[str] = set()

    def discover_images(self, image_dir: Path) -> List[Path]:
        """Discover all images in directory."""
        image_dir = Path(image_dir)
        images = []

        for ext in SUPPORTED_FORMATS:
            images.extend(image_dir.rglob(f"*{ext}"))
            images.extend(image_dir.rglob(f"*{ext.upper()}"))

        logger.info(f"Found {len(images)} images in {image_dir}")
        return sorted(images)

    def index_images(
        self,
        image_dir: Path,
        batch_size: int = 32,
        skip_existing: bool = True,
    ) -> None:
        """
        Index all images in directory.

        Args:
            image_dir: Directory containing images
            batch_size: Batch size for embedding generation
            skip_existing: Skip already indexed images
        """
        images = self.discover_images(image_dir)
        if not images:
            logger.warning("No images found")
            return

        # Load existing index if available
        has_existing_index = self.faiss_mgr.load_index()

        all_vectors = []
        all_image_ids = []

        try:
            for idx, img_path in enumerate(images):
                try:
                    # Generate image ID
                    image_id = generate_image_id(img_path)

                    # Check if already indexed
                    if skip_existing and self.db.image_exists(image_id):
                        logger.debug(f"Skipping already indexed: {img_path}")
                        continue

                    # Get image dimensions
                    dims = get_image_dimensions(img_path)
                    if dims is None:
                        logger.warning(f"Failed to get dimensions: {img_path}")
                        continue

                    width, height = dims

                    # Load image for embedding
                    img_rgb = load_image_rgb(img_path)
                    if img_rgb is None:
                        logger.warning(f"Failed to load image: {img_path}")
                        continue

                    # Generate embedding
                    embedding = self.embedding_model.encode_image(img_rgb, normalize=True)

                    # Generate thumbnail
                    generate_thumbnail(
                        img_path,
                        settings.data_dir / "thumbs",
                        thumb_size=settings.thumb_size,
                        thumb_format=settings.thumb_format,
                        image_id=image_id,
                    )

                    # Detect faces
                    faces = self.face_detector.detect_faces(img_rgb)

                    # Store image metadata
                    ts = int(time.time())
                    self.db.upsert_image(
                        image_id=image_id,
                        path=str(img_path),
                        width=width,
                        height=height,
                        indexed_ts=ts,
                    )

                    # Store image vector metadata
                    self.db.upsert_image_vector(
                        image_id=image_id,
                        dim=len(embedding),
                        norm=float(np.linalg.norm(embedding)),
                    )

                    # Store faces
                    for face_idx, face in enumerate(faces):
                        face_id = f"{image_id}_face_{face_idx}"
                        bbox_json = json.dumps(face["bbox"])
                        face_embedding = face.get("embedding")

                        if face_embedding is not None:
                            # L2-normalize
                            face_embedding = (
                                face_embedding
                                / (np.linalg.norm(face_embedding) + 1e-8)
                            )

                            self.db.insert_face(
                                face_id=face_id,
                                image_id=image_id,
                                bbox=bbox_json,
                                embedding=face_embedding.astype(np.float32).tobytes(),
                                confidence=face.get("det_score"),
                            )

                    # Accumulate vector
                    all_vectors.append(embedding)
                    all_image_ids.append(image_id)

                    if len(all_vectors) % batch_size == 0:
                        logger.info(f"Processed {len(all_vectors)} images...")

                except Exception as e:
                    logger.error(f"Error indexing {img_path}: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Indexing interrupted by user")

        # Build FAISS index
        if all_vectors:
            vectors_array = np.array(all_vectors, dtype=np.float32)

            if has_existing_index and self.faiss_mgr.index is not None:
                # Add to existing index
                logger.info(f"Adding {len(all_vectors)} vectors to existing index...")
                start_id = self.faiss_mgr.get_index_size()
                self.faiss_mgr.add_vectors(vectors_array, start_id=start_id)

                # Update ID map
                for local_idx, img_id in enumerate(all_image_ids):
                    self.faiss_mgr.register_image(img_id, start_id + local_idx)
            else:
                # Build new index
                logger.info(f"Building new index with {len(all_vectors)} vectors...")
                self.faiss_mgr.build_index(
                    vectors_array,
                    embedding_dim=self.embedding_model.embedding_dim,
                )

                # Register IDs
                for idx, img_id in enumerate(all_image_ids):
                    self.faiss_mgr.register_image(img_id, idx)

            # Save index
            self.faiss_mgr.save_index()

            logger.info(f"Successfully indexed {len(all_vectors)} images")
        else:
            logger.warning("No new images to index")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Index photos for semantic search")
    parser.add_argument("--image-dir", type=Path, default="data/images", help="Image directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--skip-existing", action="store_true", default=True)

    args = parser.parse_args()

    indexer = PhotoIndexer()
    indexer.index_images(args.image_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

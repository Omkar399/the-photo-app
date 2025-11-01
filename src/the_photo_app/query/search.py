"""Search query engine with semantic and face-aware filtering."""

import numpy as np
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from the_photo_app.config import settings
from the_photo_app.db.schema import Database
from the_photo_app.models.model_manager import model_manager
from the_photo_app.indexing.faiss_manager import FAISSManager
from the_photo_app.utils.normalization import l2_normalize, batch_cosine_similarity

logger = logging.getLogger(__name__)


class SearchEngine:
    """Semantic search engine with face awareness."""

    def __init__(self):
        """Initialize search engine."""
        self.db = Database(settings.db_path)
        self.faiss_mgr = FAISSManager(
            settings.data_dir / "faiss",
            nlist=settings.faiss_nlist,
            m=settings.faiss_m,
            nbits=settings.faiss_nbits,
            nprobe=settings.faiss_nprobe,
        )

        # Load index
        if not self.faiss_mgr.load_index():
            logger.warning("No FAISS index found. Run indexing first.")

        self.embedding_model = model_manager.get_embedding_model()
        self.face_detector = model_manager.get_face_detector()

    def search(
        self,
        query: str,
        topk: int = 200,
        alpha: float = 0.6,
        person_labels: Optional[List[str]] = None,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Search for images matching text query.

        Args:
            query: Text query
            topk: Number of results to return
            alpha: Weight for semantic score (1-alpha for face score)
            person_labels: Filter by person labels
            date_from: Filter by date (unix timestamp)
            date_to: Filter by date (unix timestamp)

        Returns:
            Dict with results list
        """
        try:
            logger.info(f"Starting search for query: {query}")
            
            # Encode query
            query_embedding = self.embedding_model.encode_text(query, normalize=True)
            query_embedding = query_embedding.astype(np.float32)
            logger.info(f"Query embedding shape: {query_embedding.shape}")

            # Search FAISS
            logger.info(f"Searching FAISS index for top {topk}...")
            distances, indices = self.faiss_mgr.search(query_embedding, k=topk)
            logger.info(f"FAISS returned distances: {distances.shape}, indices: {indices.shape}")

            # Convert distances to similarity scores (1 - distance for L2)
            scores = 1.0 - distances
            logger.info(f"Scores shape: {scores.shape}, scores: {scores}")

            # Map indices to image IDs
            logger.info(f"Mapping {len(indices)} indices to image IDs...")
            image_ids = self.faiss_mgr.map_indices_to_ids(indices)
            logger.info(f"Got {len(image_ids)} image IDs")

            # Filter and format results
            results = []
            for idx, (img_id, score) in enumerate(zip(image_ids, scores)):
                if img_id is None:
                    logger.debug(f"Skipping None image_id at index {idx}")
                    continue

                try:
                    # Build result
                    result = {
                        "image_id": img_id,
                        "score": float(score),
                        "faces": self._get_image_faces(img_id),
                    }

                    # Apply filters
                    if person_labels and not self._has_person_labels(img_id, person_labels):
                        continue

                    results.append(result)

                except Exception as e:
                    logger.warning(f"Error processing image {img_id}: {e}", exc_info=True)
                    continue

                if len(results) >= topk:
                    break

            logger.info(f"Returning {len(results)} results")
            return {
                "query": query,
                "results": results,
                "count": len(results),
            }

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return {"query": query, "results": [], "count": 0, "error": str(e)}

    def probe_face_search(
        self,
        query: str,
        probe_face: np.ndarray,
        topk: int = 200,
        alpha: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Search with face probe + optional text query.

        Args:
            query: Optional text query
            probe_face: Probe face image (HxWxC, RGB)
            topk: Number of results
            alpha: Weight for semantic score (1-alpha for face score)

        Returns:
            Dict with results
        """
        try:
            results = []

            # Get base results
            if query and len(query) > 0:
                base_results = self.search(query, topk=topk * 2)
                image_ids = [r["image_id"] for r in base_results.get("results", [])]
            else:
                # Return top images by arbitrary metric
                image_ids = list(self.faiss_mgr.id_map.values())[:topk * 2]

            # Encode probe face
            faces_probe = self.face_detector.detect_faces(probe_face)
            if not faces_probe:
                logger.warning("No face detected in probe image")
                return {
                    "query": query,
                    "results": [],
                    "count": 0,
                    "error": "No face in probe image",
                }

            probe_embedding = faces_probe[0].get("embedding")
            if probe_embedding is None:
                return {"query": query, "results": [], "count": 0, "error": "No embedding"}

            probe_embedding = probe_embedding / (np.linalg.norm(probe_embedding) + 1e-8)

            # Score each image by best face match
            for img_id in image_ids:
                faces = self._get_image_faces_with_embeddings(img_id)
                if not faces:
                    continue

                # Find best face match
                best_score = 0.0
                best_face = None
                for face in faces:
                    face_embedding = face.get("embedding")
                    if face_embedding is not None:
                        sim = np.dot(probe_embedding, face_embedding)
                        if sim > best_score:
                            best_score = sim
                            best_face = face

                if best_score > 0.0:
                    result = {
                        "image_id": img_id,
                        "face_score": float(best_score),
                        "best_face": best_face,
                        "faces": self._get_image_faces(img_id),
                    }
                    results.append(result)

            # Sort by face score
            results.sort(key=lambda x: x["face_score"], reverse=True)

            return {
                "query": query,
                "results": results[:topk],
                "count": len(results[:topk]),
            }

        except Exception as e:
            logger.error(f"Face search error: {e}")
            return {"query": query, "results": [], "count": 0, "error": str(e)}

    def _get_image_faces(self, image_id: str) -> List[Dict[str, Any]]:
        """Get faces for image (without embeddings)."""
        faces_rows = self.db.get_faces_for_image(image_id)
        faces = []

        for row in faces_rows:
            try:
                # Handle both tuple and Row objects
                if hasattr(row, 'keys'):  # sqlite3.Row object
                    bbox = json.loads(row['bbox'])
                    person_label = row['person_label']
                    confidence = row['confidence']
                else:  # tuple
                    bbox = json.loads(row[2])
                    person_label = row[4]
                    confidence = row[5]
                
                face = {
                    "bbox": bbox,
                    "person_label": person_label,
                    "confidence": confidence,
                }
                faces.append(face)
            except Exception as e:
                logger.warning(f"Error parsing face row: {e}")
                continue

        return faces

    def _get_image_faces_with_embeddings(self, image_id: str) -> List[Dict[str, Any]]:
        """Get faces for image (with embeddings)."""
        faces_rows = self.db.get_faces_for_image(image_id)
        faces = []

        for row in faces_rows:
            try:
                # Handle both tuple and Row objects
                if hasattr(row, 'keys'):  # sqlite3.Row object
                    bbox = json.loads(row['bbox'])
                    embedding_bytes = row['embedding']
                    person_label = row['person_label']
                    confidence = row['confidence']
                else:  # tuple
                    bbox = json.loads(row[2])
                    embedding_bytes = row[3]
                    person_label = row[4]
                    confidence = row[5]

                # Convert bytes back to array
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

                face = {
                    "bbox": bbox,
                    "embedding": embedding,
                    "person_label": person_label,
                    "confidence": confidence,
                }
                faces.append(face)
            except Exception as e:
                logger.warning(f"Error parsing face row with embeddings: {e}")
                continue

        return faces

    def _has_person_labels(self, image_id: str, person_labels: List[str]) -> bool:
        """Check if image has any of the specified person labels."""
        faces = self._get_image_faces(image_id)
        image_labels = {f.get("person_label") for f in faces if f.get("person_label")}
        return bool(image_labels & set(person_labels))


# Global search engine instance
search_engine = SearchEngine()

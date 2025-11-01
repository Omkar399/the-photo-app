"""FastAPI server for AFace photo search."""

import io
from typing import Optional, List
from pathlib import Path
import base64
import numpy as np
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from the_photo_app.config import settings
from the_photo_app.query.search import search_engine
from the_photo_app.utils.image import load_image_rgb
from PIL import Image

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AFace Photo Search",
    description="Local semantic search + face-aware photo indexing",
    version="0.1.0",
)


# Pydantic models
class SearchRequest(BaseModel):
    """Search request model."""

    q: str
    topk: int = 200
    alpha: float = 0.6
    person_labels: Optional[List[str]] = None
    date_from: Optional[int] = None
    date_to: Optional[int] = None


class ProbeSearchRequest(BaseModel):
    """Face probe search request."""

    q: Optional[str] = ""
    probe_face_b64: str
    topk: int = 200
    alpha: float = 0.6


class LabelPersonRequest(BaseModel):
    """Request to label a person."""

    cluster_id: str
    label: str
    note: Optional[str] = None


# API Routes
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "index_size": search_engine.faiss_mgr.get_index_size(),
    }


@app.post("/search")
async def search(request: SearchRequest):
    """
    Search for images by text query.

    Example:
    ```json
    {
      "q": "kids playing soccer",
      "topk": 50,
      "person_labels": ["Alice"]
    }
    ```
    """
    try:
        result = search_engine.search(
            query=request.q,
            topk=request.topk,
            alpha=request.alpha,
            person_labels=request.person_labels,
            date_from=request.date_from,
            date_to=request.date_to,
        )

        # Add thumbnail URLs
        for item in result.get("results", []):
            image_id = item["image_id"]
            item["thumb"] = f"/thumbs/{image_id}.webp"

        return result
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/probe-face-search")
async def probe_face_search(request: ProbeSearchRequest):
    """
    Search with a face probe image + optional text query.

    Send probe face as base64 encoded image.
    """
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.probe_face_b64)
            probe_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            probe_array = np.array(probe_image)
        except Exception as e:
            logger.error(f"Failed to decode probe image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image format")

        result = search_engine.probe_face_search(
            query=request.q or "",
            probe_face=probe_array,
            topk=request.topk,
            alpha=request.alpha,
        )

        # Add thumbnail URLs
        for item in result.get("results", []):
            image_id = item["image_id"]
            item["thumb"] = f"/thumbs/{image_id}.webp"

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Probe face search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/label-person")
async def label_person(request: LabelPersonRequest):
    """
    Assign a real name to a detected person/cluster.

    Example:
    ```json
    {
      "cluster_id": "person_001",
      "label": "Alice",
      "note": "My friend"
    }
    ```
    """
    try:
        search_engine.db.set_person_label(
            cluster_id=request.cluster_id,
            label=request.label,
            note=request.note,
        )

        # TODO: Update all faces with this cluster_id
        # For now, just store the mapping

        return {
            "status": "ok",
            "cluster_id": request.cluster_id,
            "label": request.label,
        }
    except Exception as e:
        logger.error(f"Label person error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thumbs/{image_id}")
async def get_thumbnail(image_id: str):
    """Get thumbnail for image."""
    try:
        thumb_path = settings.data_dir / "thumbs" / f"{image_id}.webp"

        if not thumb_path.exists():
            # Try other formats
            for ext in ["jpg", "png", "jpeg"]:
                alt_path = settings.data_dir / "thumbs" / f"{image_id}.{ext}"
                if alt_path.exists():
                    thumb_path = alt_path
                    break

        if thumb_path.exists():
            return FileResponse(thumb_path, media_type="image/webp")
        else:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thumbnail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get indexing statistics."""
    try:
        return {
            "total_images": search_engine.faiss_mgr.get_index_size(),
            "total_faces": 0,  # TODO: count from DB
            "embedding_dim": search_engine.embedding_model.embedding_dim,
            "face_embedding_dim": search_engine.face_detector.embedding_dim,
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run server."""
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

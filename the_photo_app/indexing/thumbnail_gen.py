"""Thumbnail generation utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def generate_thumbnail(
    image_path: Path,
    thumb_dir: Path,
    thumb_size: int = 320,
    thumb_format: str = "webp",
    image_id: Optional[str] = None,
) -> Optional[Path]:
    """
    Generate and save thumbnail for an image.

    Args:
        image_path: Path to original image
        thumb_dir: Directory to save thumbnails
        thumb_size: Short side length of thumbnail
        thumb_format: Format (webp, avif, jpg)
        image_id: Image ID for naming (if None, uses hash of path)

    Returns:
        Path to saved thumbnail, or None on error
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to read image: {image_path}")
            return None

        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        scale = thumb_size / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center crop to square
        h, w = thumb.shape[:2]
        y1 = max(0, (h - thumb_size) // 2)
        x1 = max(0, (w - thumb_size) // 2)
        thumb = thumb[y1 : y1 + thumb_size, x1 : x1 + thumb_size]

        # Pad if necessary (for very small images)
        if thumb.shape[0] < thumb_size or thumb.shape[1] < thumb_size:
            pad_h = thumb_size - thumb.shape[0]
            pad_w = thumb_size - thumb.shape[1]
            thumb = cv2.copyMakeBorder(
                thumb,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        # Generate thumbnail ID if not provided
        if image_id is None:
            import hashlib

            image_id = hashlib.md5(str(image_path).encode()).hexdigest()[:16]

        # Save thumbnail
        thumb_dir.mkdir(parents=True, exist_ok=True)
        thumb_ext = thumb_format.lower()
        thumb_path = thumb_dir / f"{image_id}.{thumb_ext}"

        # Encode options based on format
        if thumb_ext == "webp":
            cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_WEBP_QUALITY, 85])
        elif thumb_ext == "avif":
            # Note: OpenCV AVIF support requires build with libaom
            try:
                cv2.imwrite(str(thumb_path), thumb)
            except Exception:
                logger.warning(f"AVIF encoding not supported, falling back to JPEG")
                thumb_path = thumb_dir / f"{image_id}.jpg"
                cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        elif thumb_ext == "jpg" or thumb_ext == "jpeg":
            cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            logger.warning(f"Unknown format {thumb_ext}, using JPEG")
            thumb_path = thumb_dir / f"{image_id}.jpg"
            cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        logger.debug(f"Generated thumbnail: {thumb_path}")
        return thumb_path

    except Exception as e:
        logger.error(f"Error generating thumbnail for {image_path}: {e}")
        return None

"""Image I/O and preprocessing utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_image_cv2(path: Path) -> Optional[np.ndarray]:
    """Load image using OpenCV (BGR format)."""
    try:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to load image: {path}")
            return None
        return img
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None


def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Load image in RGB format."""
    img = load_image_cv2(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_image_dimensions(path: Path) -> Optional[Tuple[int, int]]:
    """Get image dimensions (width, height)."""
    img = load_image_cv2(path)
    if img is not None:
        h, w = img.shape[:2]
        return w, h
    return None


def resize_image(
    img: np.ndarray,
    target_size: int = 256,
    center_crop: bool = True,
) -> np.ndarray:
    """
    Resize image to target size with optional center crop.

    Args:
        img: Input image (HxWxC)
        target_size: Target short side length
        center_crop: If True, center crop to square

    Returns:
        Resized/cropped image
    """
    h, w = img.shape[:2]
    scale = target_size / min(h, w)

    # Resize maintaining aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if center_crop:
        # Center crop to target_size x target_size
        h, w = img.shape[:2]
        y1 = (h - target_size) // 2
        x1 = (w - target_size) // 2
        img = img[y1 : y1 + target_size, x1 : x1 + target_size]

    return img


def normalize_image(img: np.ndarray, mean: Tuple[float, ...], std: Tuple[float, ...]) -> np.ndarray:
    """
    Normalize image using mean and std (for model inference).

    Args:
        img: Input image (HxWxC, float32 in [0, 1] or [0, 255])
        mean: Normalization mean per channel
        std: Normalization std per channel

    Returns:
        Normalized image
    """
    img = img.astype(np.float32)

    # Convert from [0, 255] to [0, 1]
    if img.max() > 1:
        img = img / 255.0

    # Apply normalization
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    img = (img - mean_arr) / std_arr

    return img


def preprocess_image_for_embedding(
    path: Path,
    target_size: int = 256,
    mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
    std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
) -> Optional[np.ndarray]:
    """
    Preprocess image for embedding model (SigLIP, OpenCLIP).

    Args:
        path: Path to image
        target_size: Target resolution
        mean: Normalization mean (RGB order)
        std: Normalization std (RGB order)

    Returns:
        Preprocessed image (1xCxHxW, float32)
    """
    img = load_image_rgb(path)
    if img is None:
        return None

    img = resize_image(img, target_size=target_size, center_crop=True)
    img = normalize_image(img, mean=mean, std=std)

    # Convert HxWxC to CxHxW and add batch dimension
    img = np.transpose(img, (2, 0, 1))  # CxHxW
    img = np.expand_dims(img, axis=0)  # 1xCxHxW

    return img.astype(np.float32)


def preprocess_image_for_face_detection(path: Path) -> Optional[np.ndarray]:
    """
    Preprocess image for face detection (no normalization, resize to 640).

    Args:
        path: Path to image

    Returns:
        Image array or None
    """
    img = load_image_rgb(path)
    if img is None:
        return None

    # For SCRFD, typically use 640x640
    img = resize_image(img, target_size=640, center_crop=False)
    return img.astype(np.float32)

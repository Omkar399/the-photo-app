"""Embedding model wrappers (SigLIP, OpenCLIP)."""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import open_clip
    from PIL import Image
    import torch
except ImportError:
    logger.warning("open_clip not installed. Install with: pip install open-clip-torch")


class EmbeddingModel:
    """Base embedding model wrapper."""

    def __init__(self, model_name: str = "siglip_base_256", device: str = "cuda"):
        """
        Initialize embedding model.

        Args:
            model_name: Model name (e.g., "siglip_base_256", "openclip_vitb32")
            device: "cuda" or "cpu"
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.image_encoder = None
        self.text_encoder = None
        self.processor = None
        self.tokenizer = None
        self.embedding_dim = 256

        self._load_model()

    def _load_model(self) -> None:
        """Load model from open_clip."""
        try:
            if "siglip" in self.model_name.lower():
                # SigLIP model
                model_name = "ViT-B-16-SigLIP-256"
                pretrained = "webli"
                self.embedding_dim = 256
            elif "vitb32" in self.model_name.lower():
                # OpenCLIP ViT-B/32
                model_name = "ViT-B-32"
                pretrained = "openai"
                self.embedding_dim = 512
            else:
                raise ValueError(f"Unknown model: {self.model_name}")

            result = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.model, self.processor = result[0], result[1]
            self.model.eval()
            
            # Load model-specific tokenizer
            self.tokenizer = open_clip.get_tokenizer(model_name)

            # Get actual embedding dimension from model output
            # The model name suggests 256 but actual output might be different
            import torch
            with torch.no_grad():
                test_text = self.tokenizer(["test"]).to(self.device)
                test_emb = self.model.encode_text(test_text)
                self.embedding_dim = test_emb.shape[-1]
            
            logger.info(f"Loaded embedding model: {model_name} ({pretrained}), actual dim={self.embedding_dim}")
        except AssertionError as e:
            if "CUDA" in str(e):
                logger.warning(f"CUDA not available, falling back to CPU")
                self.device = "cpu"
                result = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device="cpu"
                )
                self.model, self.processor = result[0], result[1]
                self.model.eval()
                
                # Load model-specific tokenizer
                self.tokenizer = open_clip.get_tokenizer(model_name)
                
                logger.info(f"Loaded embedding model on CPU: {model_name} ({pretrained})")
            else:
                raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode_image(
        self, image: Union[Path, np.ndarray], normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a single image to embedding.

        Args:
            image: Path to image or numpy array (RGB)
            normalize: If True, L2-normalize the embedding

        Returns:
            Embedding vector (1-D, float32)
        """
        if isinstance(image, Path):
            img = Image.open(image).convert("RGB")
        else:
            img = Image.fromarray(image.astype("uint8"))

        img_tensor = self.processor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(img_tensor)

        embedding = embedding.detach().cpu().numpy().astype(np.float32)

        if normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.squeeze(0)

    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode text to embedding.

        Args:
            text: Text string
            normalize: If True, L2-normalize the embedding

        Returns:
            Embedding vector (1-D, float32)
        """
        try:
            # Use model-specific tokenizer
            text_tokens = self.tokenizer([text]).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_text(text_tokens)

            embedding = embedding.detach().cpu().numpy().astype(np.float32)

            if normalize:
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            return embedding.squeeze(0)
        except Exception as e:
            logger.error(f"Failed to encode text '{text}': {e}", exc_info=True)
            # Return a zero vector as last resort
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def encode_images_batch(
        self, images: list[Union[Path, np.ndarray]], batch_size: int = 32, normalize: bool = True
    ) -> np.ndarray:
        """
        Encode multiple images to embeddings.

        Args:
            images: List of image paths or arrays
            batch_size: Batch size for processing
            normalize: If True, L2-normalize embeddings

        Returns:
            Embeddings (N x D, float32)
        """
        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_imgs = []

            for img in batch:
                if isinstance(img, Path):
                    pil_img = Image.open(img).convert("RGB")
                else:
                    pil_img = Image.fromarray(img.astype("uint8"))
                batch_imgs.append(self.processor(pil_img))

            img_tensor = torch.stack(batch_imgs).to(self.device)

            with torch.no_grad():
                batch_embeddings = self.model.encode_image(img_tensor)

            batch_embeddings = batch_embeddings.detach().cpu().numpy().astype(np.float32)

            if normalize:
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / (norms + 1e-8)

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

"""CLIP ViT-B/32 encoder for object crops with optional disk caching.

Usage:
    encoder = CLIPEncoder(device="cuda", cache=DiskCache("./cache/clip"))
    embs = encoder.encode_crops(image_pil, [[x1,y1,x2,y2], ...])  # (N, 512)
"""

import hashlib
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

# CLIP is installed from https://github.com/openai/CLIP
try:
    import clip
except ImportError:
    raise ImportError(
        "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
    )

from src.utils.cache import DiskCache


CLIP_DIM = 512  # ViT-B/32 output dimension


def make_clip_cache_key(image_id: Union[int, str], boxes: List[List[float]]) -> str:
    """Canonical cache key for a (image_id, boxes) pair.

    This is the single source of truth imported by:
      - scripts/warmup_cache.py
      - src/datasets/vg_dataset.py
      - CLIPEncoder.encode_from_cache_or_compute

    Key format: ``clip_{image_id}_{sha256(box_sig)[:16]}``
    where box_sig serialises every box as ``x1.f,y1.f,x2.f,y2.f``
    joined by underscores.  Floating-point format ``.1f`` is used so
    integer and float box coordinates hash identically.
    """
    box_sig = "_".join(
        f"{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}" for b in boxes
    )
    return f"clip_{image_id}_{hashlib.sha256(box_sig.encode()).hexdigest()[:16]}"


class CLIPEncoder:
    """Encodes object crops using frozen CLIP ViT-B/32.

    Args:
        model_name: CLIP model variant (default "ViT-B/32").
        device:     "cuda" or "cpu".
        cache:      DiskCache instance. Pass None to disable caching.
        batch_size: Number of crops per CLIP forward pass.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        cache: Optional[DiskCache] = None,
        batch_size: int = 32,
    ):
        device = device if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cache = cache
        self.batch_size = batch_size

    def _crop_and_preprocess(
        self, image: Image.Image, boxes: List[List[float]]
    ) -> torch.Tensor:
        """Crop image regions and stack into a batch tensor."""
        crops = []
        W, H = image.size
        for box in boxes:
            x1, y1, x2, y2 = box
            # Clamp to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(W, int(x2))
            y2 = min(H, int(y2))
            if x2 <= x1 or y2 <= y1:
                # Degenerate box — use full image as fallback
                crop = image
            else:
                crop = image.crop((x1, y1, x2, y2))
            crops.append(self.preprocess(crop))
        return torch.stack(crops)  # (N, 3, 224, 224)

    @torch.no_grad()
    def encode_crops(
        self, image: Image.Image, boxes: List[List[float]]
    ) -> np.ndarray:
        """Encode a list of boxes from an image.

        Returns:
            np.ndarray of shape (N, 512) dtype float32 (L2-normalised).
        """
        if not boxes:
            return np.zeros((0, CLIP_DIM), dtype=np.float32)

        all_embs = []
        for start in range(0, len(boxes), self.batch_size):
            batch_boxes = boxes[start : start + self.batch_size]
            tensor = self._crop_and_preprocess(image, batch_boxes).to(self.device)
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2-normalise
            all_embs.append(emb.cpu().float().numpy())

        return np.concatenate(all_embs, axis=0)  # (N, 512)

    def encode_from_cache_or_compute(
        self,
        image_id: Union[int, str],
        image: Image.Image,
        boxes: List[List[float]],
    ) -> np.ndarray:
        """Return cached embeddings if available, otherwise compute and cache.

        Cache key encodes image_id + box coordinates so any box change
        triggers a recompute.
        """
        key = make_clip_cache_key(image_id, boxes)

        if self.cache is not None and self.cache.exists(key):
            return self.cache.load(key)

        embs = self.encode_crops(image, boxes)

        if self.cache is not None:
            self.cache.save(key, embs)

        return embs

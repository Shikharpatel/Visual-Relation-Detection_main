"""
CLIP Feature Extractor for Zero-Shot VRD
==========================================

Uses a frozen CLIP ViT-B/32 model to extract visual features for objects
and text embeddings for class labels. Replaces CNN-based feature extraction
for the zero-shot pipeline.

Key properties:
- All CLIP parameters are frozen (no gradients)
- Image features: [N, 512] per object crop
- Text features: [C, 512] per class name
- ~150MB VRAM for ViT-B/32 (fits easily on RTX 3050)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np


class CLIPFeatureExtractor(nn.Module):
    """
    Frozen CLIP feature extractor for zero-shot VRD.

    Extracts visual features by cropping bounding box regions from images
    and encoding them with CLIP's image encoder. Also provides text
    embeddings for object class names.

    Args:
        clip_model_name: CLIP model architecture (e.g., 'ViT-B-32')
        pretrained: Pretrained weights source (e.g., 'openai')
        device: Device to load model on
    """

    def __init__(
        self,
        clip_model_name: str = 'ViT-B-32',
        pretrained: str = 'openai',
        device: str = 'cpu'
    ):
        super().__init__()

        self.clip_model_name = clip_model_name
        self.pretrained = pretrained
        self._device = device

        # Lazy-load CLIP model
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.feature_dim = 512  # ViT-B/32 output dim

    def _load_model(self):
        """Lazy-load the CLIP model on first use."""
        if self._model is not None:
            return

        try:
            import open_clip

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name,
                pretrained=self.pretrained,
                device=self._device
            )
            self._tokenizer = open_clip.get_tokenizer(self.clip_model_name)

            # Freeze all CLIP parameters
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False

            print(f"Loaded CLIP {self.clip_model_name} ({self.pretrained}) — "
                  f"all parameters frozen")

        except ImportError:
            raise ImportError(
                "open_clip_torch is required for CLIP features. "
                "Install with: pip install open_clip_torch"
            )

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        boxes: List[torch.Tensor],
        image_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> torch.Tensor:
        """
        Extract CLIP visual features for bounding box crops.

        Args:
            images: [B, C, H, W] batch of images (normalized 0-1)
            boxes: List of [N_i, 4] box tensors per image (x1, y1, x2, y2)
            image_sizes: Optional list of (H, W) tuples

        Returns:
            [sum(N_i), 512] CLIP image features
        """
        self._load_model()

        total_boxes = sum(len(b) for b in boxes)
        if total_boxes == 0:
            return torch.zeros(0, self.feature_dim, device=images.device)

        # Crop each box and preprocess for CLIP
        crops = []
        for b_idx, box_tensor in enumerate(boxes):
            if len(box_tensor) == 0:
                continue

            img = images[b_idx]  # [C, H, W]
            H, W = img.shape[1], img.shape[2]

            for box in box_tensor:
                x1 = max(0, int(box[0].item()))
                y1 = max(0, int(box[1].item()))
                x2 = min(W, int(box[2].item()))
                y2 = min(H, int(box[3].item()))

                # Ensure valid crop
                if x2 <= x1 or y2 <= y1:
                    x1, y1, x2, y2 = 0, 0, min(32, W), min(32, H)

                # Crop and convert to PIL for CLIP preprocessing
                crop = img[:, y1:y2, x1:x2]  # [C, h, w]
                crop_pil = Image.fromarray(
                    (crop.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )

                # Apply CLIP preprocessing (resize to 224x224 + normalize)
                crop_processed = self._preprocess(crop_pil)
                crops.append(crop_processed)

        if not crops:
            return torch.zeros(0, self.feature_dim, device=images.device)

        # Stack and encode
        crops_tensor = torch.stack(crops).to(images.device)
        features = self._model.encode_image(crops_tensor)

        # L2 normalize (standard for CLIP)
        features = features / features.norm(dim=-1, keepdim=True)

        return features.float()

    @torch.no_grad()
    def encode_text(
        self,
        class_names: List[str],
        prompt_template: str = "a photo of a {}"
    ) -> torch.Tensor:
        """
        Encode class names into CLIP text embeddings.

        Args:
            class_names: List of object class names (e.g., ['person', 'horse', ...])
            prompt_template: Template for text prompts

        Returns:
            [num_classes, 512] text embeddings
        """
        self._load_model()

        prompts = [prompt_template.format(name) for name in class_names]
        tokens = self._tokenizer(prompts)

        device = next(self._model.parameters()).device
        tokens = tokens.to(device)

        text_features = self._model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.float()

    @torch.no_grad()
    def extract_dummy_features(
        self,
        num_objects: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate zero features when no images are available.
        Used for non-visual training modes (PredCls with label embeddings only).

        Args:
            num_objects: Number of objects
            device: Torch device

        Returns:
            [num_objects, 512] zero features
        """
        return torch.zeros(num_objects, self.feature_dim, device=device)

    def get_feature_dim(self) -> int:
        """Get output feature dimension (512 for ViT-B/32)."""
        return self.feature_dim

    def forward(self, *args, **kwargs):
        """Not used directly — use extract_features() or encode_text()."""
        raise NotImplementedError(
            "Use extract_features() or encode_text() instead."
        )


def create_clip_extractor(config: dict) -> CLIPFeatureExtractor:
    """
    Factory function to create CLIP extractor from config.

    Args:
        config: Model config dict with clip_model, clip_pretrained keys

    Returns:
        CLIPFeatureExtractor instance
    """
    model_cfg = config.get('model', {})
    return CLIPFeatureExtractor(
        clip_model_name=model_cfg.get('clip_model', 'ViT-B-32'),
        pretrained=model_cfg.get('clip_pretrained', 'openai'),
        device=config.get('device', 'cpu')
    )

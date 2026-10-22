"""
Relation Head Module for VRD
=============================

Predicts predicates for subject-object pairs using visual and spatial features.

The relation head combines:
1. Visual features: f_s (subject), f_o (object), f_u (union)
2. Spatial features: g(b_s, b_o) encoding relative geometry
3. (Optional) Frequency bias: P(predicate | subject_class, object_class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math

from ..utils.box_ops import compute_spatial_features, box_union


class SpatialEncoder(nn.Module):
    """
    Encodes spatial relationships between subject and object boxes.

    Features computed:
    - Relative position: (dx, dy) / subject_size
    - Relative scale: log(obj_size / subj_size)
    - IoU between boxes
    - Normalized box coordinates

    Args:
        output_dim: Output feature dimension
        use_coords: Include normalized box coordinates
    """

    def __init__(
        self,
        output_dim: int = 128,
        use_coords: bool = True
    ):
        super().__init__()

        # Input dimension: 5 base features + 8 coord features (if used)
        input_dim = 5 + (8 if use_coords else 0)
        self.use_coords = use_coords

        # MLP to encode spatial features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim)
        )

    def forward(
        self,
        subj_boxes: torch.Tensor,
        obj_boxes: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Compute spatial features for box pairs.

        Args:
            subj_boxes: [N, 4] subject boxes
            obj_boxes: [N, 4] object boxes
            image_size: (height, width) for normalization

        Returns:
            [N, output_dim] spatial features
        """
        # Only use coords if image_size is provided AND use_coords is True
        use_img_size = image_size if (self.use_coords and image_size is not None) else None

        # Compute raw spatial features
        spatial_feats = compute_spatial_features(
            subj_boxes, obj_boxes,
            use_img_size
        )

        # If we expected coords but didn't get them, pad with zeros
        if self.use_coords and image_size is None:
            # Pad from 5 features to 13 features
            padding = torch.zeros(
                spatial_feats.size(0), 8,
                device=spatial_feats.device,
                dtype=spatial_feats.dtype
            )
            spatial_feats = torch.cat([spatial_feats, padding], dim=-1)

        # Encode through MLP
        return self.encoder(spatial_feats)


class RelationHead(nn.Module):
    """
    Relation prediction head for VRD.

    For each object pair (i, j), predicts predicate p_ij using:
    - Visual features of subject and object
    - Union box feature
    - Spatial encoding of box geometry

    Architecture:
        x = concat(f_s, f_o, f_u, g_spatial)
        logits = MLP(x) = W2 * ReLU(W1 * x + b1) + b2

    Args:
        visual_dim: Dimension of visual features
        spatial_dim: Dimension of spatial features
        hidden_dim: Hidden layer dimension
        num_predicates: Number of predicate classes
        dropout: Dropout probability
        use_union_features: Include union box features
        use_spatial: Include spatial features
    """

    def __init__(
        self,
        visual_dim: int = 256,
        spatial_dim: int = 128,
        hidden_dim: int = 512,
        num_predicates: int = 51,
        dropout: float = 0.3,
        use_union_features: bool = True,
        use_spatial: bool = True
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.num_predicates = num_predicates
        self.use_union_features = use_union_features
        self.use_spatial = use_spatial

        # Spatial encoder
        if use_spatial:
            self.spatial_encoder = SpatialEncoder(output_dim=spatial_dim)
        else:
            self.spatial_encoder = None

        # Compute input dimension
        input_dim = visual_dim * 2  # subject + object
        if use_union_features:
            input_dim += visual_dim
        if use_spatial:
            input_dim += spatial_dim

        # Predicate classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicates)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        subj_features: torch.Tensor,
        obj_features: torch.Tensor,
        subj_boxes: torch.Tensor,
        obj_boxes: torch.Tensor,
        union_features: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Predict predicate logits for object pairs.

        Args:
            subj_features: [N, visual_dim] subject visual features
            obj_features: [N, visual_dim] object visual features
            subj_boxes: [N, 4] subject boxes
            obj_boxes: [N, 4] object boxes
            union_features: [N, visual_dim] optional union box features
            image_size: (height, width) for spatial feature normalization

        Returns:
            [N, num_predicates] predicate logits
        """
        # Collect features to concatenate
        features = [subj_features, obj_features]

        # Add union features
        if self.use_union_features and union_features is not None:
            features.append(union_features)

        # Add spatial features
        if self.use_spatial:
            spatial_feats = self.spatial_encoder(subj_boxes, obj_boxes, image_size)
            features.append(spatial_feats)

        # Concatenate all features
        x = torch.cat(features, dim=-1)

        # Classify
        logits = self.classifier(x)

        return logits

    def get_input_dim(self) -> int:
        """Get expected input dimension."""
        dim = self.visual_dim * 2
        if self.use_union_features:
            dim += self.visual_dim
        if self.use_spatial:
            dim += self.spatial_dim
        return dim


class RelationHeadWithContext(nn.Module):
    """
    Relation head with context modeling using self-attention.

    Adds global context by attending over all objects in the scene
    before predicting predicates.

    Args:
        visual_dim: Visual feature dimension
        spatial_dim: Spatial feature dimension
        hidden_dim: Hidden layer dimension
        num_predicates: Number of predicate classes
        num_heads: Number of attention heads
        num_layers: Number of attention layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        visual_dim: int = 256,
        spatial_dim: int = 128,
        hidden_dim: int = 512,
        num_predicates: int = 51,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.num_predicates = num_predicates

        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(output_dim=spatial_dim)

        # Context encoder (self-attention over objects)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=visual_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Input projection
        self.input_proj = nn.Linear(visual_dim, visual_dim)

        # Predicate classifier
        input_dim = visual_dim * 3 + spatial_dim  # context_subj, context_obj, union, spatial
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicates)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        obj_features: torch.Tensor,
        pair_indices: torch.Tensor,
        boxes: torch.Tensor,
        union_features: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Predict predicates with context.

        Args:
            obj_features: [B, N, visual_dim] object features
            pair_indices: [B, R, 2] relation pair indices (subj, obj)
            boxes: [B, N, 4] object boxes
            union_features: [B, R, visual_dim] union features
            object_mask: [B, N] mask for valid objects
            image_size: Image dimensions

        Returns:
            [B, R, num_predicates] predicate logits
        """
        B, N, D = obj_features.shape
        _, R, _ = pair_indices.shape

        # Project features
        obj_features = self.input_proj(obj_features)

        # Apply self-attention for context
        if object_mask is not None:
            # Create attention mask (True = ignore)
            attn_mask = ~object_mask
        else:
            attn_mask = None

        context_features = self.context_encoder(
            obj_features,
            src_key_padding_mask=attn_mask
        )

        # Gather features for each relation
        batch_idx = torch.arange(B, device=obj_features.device).unsqueeze(1).expand(-1, R)

        subj_idx = pair_indices[:, :, 0]  # [B, R]
        obj_idx = pair_indices[:, :, 1]   # [B, R]

        subj_features = context_features[batch_idx, subj_idx]  # [B, R, D]
        obj_features_rel = context_features[batch_idx, obj_idx]  # [B, R, D]

        subj_boxes = boxes[batch_idx, subj_idx]  # [B, R, 4]
        obj_boxes = boxes[batch_idx, obj_idx]    # [B, R, 4]

        # Compute spatial features
        spatial_feats = []
        for b in range(B):
            sf = self.spatial_encoder(subj_boxes[b], obj_boxes[b], image_size)
            spatial_feats.append(sf)
        spatial_feats = torch.stack(spatial_feats)  # [B, R, spatial_dim]

        # Concatenate all features
        x = torch.cat([
            subj_features,
            obj_features_rel,
            union_features,
            spatial_feats
        ], dim=-1)

        # Classify
        logits = self.classifier(x)

        return logits


class PairFeatureExtractor(nn.Module):
    """
    Extracts features for subject-object pairs from individual object features.

    Handles efficient batched extraction and union box computation.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        spatial_dim: int = 128,
        use_union: bool = True,
        use_spatial: bool = True
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        self.use_union = use_union
        self.use_spatial = use_spatial

        if use_spatial:
            self.spatial_encoder = SpatialEncoder(output_dim=spatial_dim)

    def forward(
        self,
        obj_features: torch.Tensor,
        boxes: torch.Tensor,
        pair_indices: torch.Tensor,
        union_features: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract pair features.

        Args:
            obj_features: [N, feature_dim] object features
            boxes: [N, 4] object boxes
            pair_indices: [R, 2] pairs (subj_idx, obj_idx)
            union_features: [R, feature_dim] optional precomputed union features
            image_size: Image size for spatial features

        Returns:
            Dictionary with:
                - subj_features: [R, feature_dim]
                - obj_features: [R, feature_dim]
                - union_features: [R, feature_dim] (if use_union)
                - spatial_features: [R, spatial_dim] (if use_spatial)
                - subj_boxes: [R, 4]
                - obj_boxes: [R, 4]
        """
        subj_idx = pair_indices[:, 0]
        obj_idx = pair_indices[:, 1]

        # Gather features
        subj_features = obj_features[subj_idx]
        obj_features_out = obj_features[obj_idx]

        subj_boxes = boxes[subj_idx]
        obj_boxes = boxes[obj_idx]

        result = {
            'subj_features': subj_features,
            'obj_features': obj_features_out,
            'subj_boxes': subj_boxes,
            'obj_boxes': obj_boxes
        }

        if self.use_union and union_features is not None:
            result['union_features'] = union_features

        if self.use_spatial:
            spatial = self.spatial_encoder(subj_boxes, obj_boxes, image_size)
            result['spatial_features'] = spatial

        return result

    def get_output_dim(self) -> int:
        """Get total output dimension."""
        dim = self.feature_dim * 2
        if self.use_union:
            dim += self.feature_dim
        if self.use_spatial:
            dim += self.spatial_dim
        return dim

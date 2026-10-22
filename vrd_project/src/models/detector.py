"""
Object Detector Module for VRD
===============================

Provides object detection capabilities for SGDet mode using pretrained models.
For PredCls/SGCls modes, ground truth boxes are used instead.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align, boxes as box_ops
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class ObjectDetector(nn.Module):
    """
    Object detector wrapper for VRD.

    Can use:
    - Pretrained Faster R-CNN for SGDet mode
    - Ground truth boxes for PredCls/SGCls modes

    Args:
        backbone: Backbone type ('fasterrcnn_resnet50_fpn' or 'none')
        num_classes: Number of object classes (including background)
        pretrained: Use pretrained weights
        feature_dim: Output feature dimension for ROI features
        trainable_backbone_layers: Number of trainable backbone layers (0-5)
    """

    def __init__(
        self,
        backbone: str = 'fasterrcnn_resnet50_fpn',
        num_classes: int = 151,
        pretrained: bool = True,
        feature_dim: int = 256,
        trainable_backbone_layers: int = 3
    ):
        super().__init__()

        self.backbone_type = backbone
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        if backbone == 'fasterrcnn_resnet50_fpn':
            # Load pretrained Faster R-CNN
            if pretrained:
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                self.detector = fasterrcnn_resnet50_fpn(
                    weights=weights,
                    trainable_backbone_layers=trainable_backbone_layers
                )
            else:
                self.detector = fasterrcnn_resnet50_fpn(
                    weights=None,
                    trainable_backbone_layers=trainable_backbone_layers
                )

            # Replace the classifier head for our number of classes
            in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
            self.detector.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )

            # Store ROI feature dimension
            self.roi_feature_dim = in_features

            # Feature projection if needed
            if feature_dim != in_features:
                self.feature_proj = nn.Linear(in_features, feature_dim)
            else:
                self.feature_proj = nn.Identity()

        elif backbone == 'none':
            # No detector - for PredCls mode with precomputed features
            self.detector = None
            self.roi_feature_dim = feature_dim
            self.feature_proj = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Feature extractor for getting ROI features from arbitrary boxes
        self.roi_pool_size = 7

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for detection.

        Args:
            images: Batch of images [B, C, H, W]
            targets: Optional targets for training (list of dicts with 'boxes', 'labels')

        Returns:
            Dictionary with:
                - boxes: Detected/given boxes [B, N, 4]
                - labels: Detected/given labels [B, N]
                - scores: Detection scores [B, N]
                - features: ROI features [B, N, D]
                - losses: Training losses (if targets provided)
        """
        if self.detector is None:
            raise RuntimeError("No detector available. Use extract_features with GT boxes instead.")

        # Run detector
        if self.training and targets is not None:
            # Training mode: returns losses
            loss_dict = self.detector(images, targets)
            return {'losses': loss_dict}
        else:
            # Inference mode: returns detections
            detections = self.detector(images)
            return {
                'detections': detections
            }

    def extract_features(
        self,
        images: torch.Tensor,
        boxes: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract ROI features for given boxes using the detector's
        multi-scale ROI pooler and box_head (TwoMLPHead).

        Args:
            images: Batch of images [B, C, H, W]
            boxes: List of box tensors [N_i, 4] per image

        Returns:
            ROI features [sum(N_i), roi_feature_dim (1024)]
        """
        if self.detector is None:
            raise RuntimeError("No backbone available for feature extraction.")

        total_boxes = sum(len(b) for b in boxes)
        if total_boxes == 0:
            return torch.zeros(0, self.roi_feature_dim, device=images.device)

        with torch.no_grad():
            # Extract multi-scale FPN feature maps
            features = self.detector.backbone(images)

            # Get image shapes for ROI pooler
            image_shapes = [(images.shape[-2], images.shape[-1])] * images.shape[0]

            # Use the detector's built-in multi-scale ROI pooler
            # This handles FPN level assignment based on box area
            pooled_features = self.detector.roi_heads.box_roi_pool(
                features, boxes, image_shapes
            )  # [sum(N_i), 256, 7, 7]

            # Pass through the TwoMLPHead (flatten + 2x Linear+ReLU)
            # This produces 1024-dim features
            roi_features = self.detector.roi_heads.box_head(pooled_features)
            # [sum(N_i), 1024]

        return roi_features

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.feature_dim


class VisualFeatureExtractor(nn.Module):
    """
    Extracts visual features from images for given bounding boxes.

    Uses a pretrained CNN backbone with ROI pooling.

    Args:
        backbone: Backbone architecture ('resnet50', 'resnet101')
        feature_dim: Output feature dimension
        pretrained: Use pretrained weights
        freeze_backbone: Freeze backbone weights
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        feature_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # Load backbone
        if backbone == 'resnet50':
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = torchvision.models.resnet50(weights=weights)
        elif backbone == 'resnet101':
            weights = torchvision.models.ResNet101_Weights.DEFAULT if pretrained else None
            resnet = torchvision.models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Extract feature layers (remove FC layer)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # Get backbone output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.backbone(dummy)
            backbone_dim = out.shape[1]

        # Freeze if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ROI pooling parameters
        self.roi_pool_size = 7
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

        # Feature projection
        self.fc = nn.Linear(backbone_dim, feature_dim)

    def forward(
        self,
        images: torch.Tensor,
        boxes: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract features for given boxes.

        Args:
            images: [B, C, H, W] images
            boxes: List of [N_i, 4] boxes per image

        Returns:
            [sum(N_i), feature_dim] features
        """
        # Get feature maps
        feature_maps = self.backbone(images)

        # Prepare boxes
        batch_boxes = []
        for i, box in enumerate(boxes):
            if len(box) > 0:
                batch_idx = torch.full((len(box), 1), i, device=box.device, dtype=box.dtype)
                batch_boxes.append(torch.cat([batch_idx, box], dim=1))

        if not batch_boxes:
            return torch.zeros(0, self.feature_dim, device=images.device)

        batch_boxes = torch.cat(batch_boxes, dim=0)

        # Compute spatial scale
        img_h, img_w = images.shape[-2:]
        feat_h, feat_w = feature_maps.shape[-2:]
        spatial_scale = feat_h / img_h

        # ROI Align
        roi_features = roi_align(
            feature_maps,
            batch_boxes,
            output_size=self.roi_pool_size,
            spatial_scale=spatial_scale,
            aligned=True
        )

        # Pool and flatten
        roi_features = self.adaptive_pool(roi_features)
        roi_features = roi_features.flatten(start_dim=1)

        # Project
        roi_features = self.fc(roi_features)

        return roi_features


def create_detector(
    config: Dict[str, Any]
) -> ObjectDetector:
    """
    Factory function to create object detector from config.

    Args:
        config: Configuration dictionary with keys:
            - backbone: Backbone type
            - num_obj_classes: Number of object classes
            - pretrained: Use pretrained weights
            - feature_dim: Feature dimension
            - trainable_backbone_layers: Number of trainable layers

    Returns:
        ObjectDetector instance
    """
    return ObjectDetector(
        backbone=config.get('backbone', 'fasterrcnn_resnet50_fpn'),
        num_classes=config.get('num_obj_classes', 151),
        pretrained=config.get('pretrained', True),
        feature_dim=config.get('feature_dim', 256),
        trainable_backbone_layers=config.get('trainable_backbone_layers', 3)
    )

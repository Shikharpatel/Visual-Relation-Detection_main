"""
Scene Graph Model for VRD
==========================

Main model that combines object detection, feature extraction, and relation
prediction into a unified scene graph generation pipeline.

Supports three evaluation modes:
- PredCls: Ground truth boxes + labels given, predict predicates only
- SGCls: Ground truth boxes given, predict labels + predicates
- SGDet: Predict boxes + labels + predicates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .detector import ObjectDetector, VisualFeatureExtractor
from .relation_head import RelationHead, SpatialEncoder
from .frequency_bias import FrequencyBias
from .motifnet import MotifNetRelationHead
from .vtranse import VTransE
from .graph_context import GraphContextEncoder
from ..utils.box_ops import box_union
from ..utils.losses import focal_cross_entropy, compute_class_weights


class SceneGraphModel(nn.Module):
    """
    Complete scene graph generation model.

    Components:
    1. Object detector/feature extractor
    2. Object classifier (for SGCls/SGDet)
    3. Relation head for predicate prediction
    4. Frequency bias module

    Args:
        num_obj_classes: Number of object classes (including background)
        num_pred_classes: Number of predicate classes
        mode: Evaluation mode ('predcls', 'sgcls', 'sgdet')
        feature_dim: Visual feature dimension
        hidden_dim: Hidden layer dimension
        spatial_dim: Spatial feature dimension
        use_bias: Use frequency bias
        bias_alpha: Weight for frequency bias
        dropout: Dropout probability
        use_union_features: Use union box features
    """

    def __init__(
        self,
        num_obj_classes: int = 151,
        num_pred_classes: int = 51,
        mode: str = 'predcls',
        feature_dim: int = 256,
        hidden_dim: int = 512,
        spatial_dim: int = 128,
        use_bias: bool = True,
        bias_alpha: float = 1.0,
        dropout: float = 0.3,
        use_union_features: bool = True
    ):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_pred_classes = num_pred_classes
        self.mode = mode.lower()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.bias_alpha = bias_alpha
        self.use_union_features = use_union_features

        # Object label embedding (for combining with visual features)
        self.obj_embed = nn.Embedding(num_obj_classes, feature_dim)

        # Feature projection (from ROI features to model feature_dim)
        # Adjust based on typical ROI feature sizes
        self.feature_proj = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

        # For SGCls/SGDet: object classifier
        if self.mode in ['sgcls', 'sgdet']:
            self.obj_classifier = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_obj_classes)
            )
        else:
            self.obj_classifier = None

        # Relation head
        self.relation_head = RelationHead(
            visual_dim=feature_dim,
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            num_predicates=num_pred_classes,
            dropout=dropout,
            use_union_features=use_union_features,
            use_spatial=True
        )

        # Frequency bias
        if use_bias:
            self.freq_bias = FrequencyBias(
                num_obj_classes=num_obj_classes,
                num_pred_classes=num_pred_classes,
                alpha=bias_alpha
            )
        else:
            self.freq_bias = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.obj_embed.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        obj_features: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        relation_pairs: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        relation_mask: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for scene graph generation.

        Args:
            obj_features: [B, N, D] object visual features (or [N, D] for single image)
            boxes: [B, N, 4] object boxes
            labels: [B, N] object labels (ground truth for predcls, predicted for others)
            relation_pairs: [B, R, 2] relation pair indices
            object_mask: [B, N] valid object mask
            relation_mask: [B, R] valid relation mask
            image_sizes: List of (H, W) tuples

        Returns:
            Dictionary with:
                - pred_logits: [B, R, num_pred] predicate logits
                - obj_logits: [B, N, num_obj] object logits (if mode != 'predcls')
                - pred_scores: [B, R, num_pred] predicate probabilities
        """
        # Handle single image input (no batch dimension)
        single_input = obj_features.dim() == 2
        if single_input:
            obj_features = obj_features.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            labels = labels.unsqueeze(0)
            relation_pairs = relation_pairs.unsqueeze(0)
            if object_mask is not None:
                object_mask = object_mask.unsqueeze(0)
            if relation_mask is not None:
                relation_mask = relation_mask.unsqueeze(0)

        B, N, D = obj_features.shape
        _, R, _ = relation_pairs.shape

        # Project features if needed
        if D != self.feature_dim:
            obj_features = self.feature_proj(obj_features)

        # Get object predictions (for sgcls/sgdet)
        obj_logits = None
        if self.obj_classifier is not None and self.mode != 'predcls':
            obj_logits = self.obj_classifier(obj_features)  # [B, N, num_obj]
            # Use predicted labels
            if self.mode in ['sgcls', 'sgdet']:
                labels = obj_logits.argmax(dim=-1)

        # Add label embeddings to visual features
        label_embeds = self.obj_embed(labels)  # [B, N, feature_dim]
        obj_features = obj_features + label_embeds

        # Pre-allocate output tensors — avoids per-image zeros+cat padding
        device = obj_features.device
        pred_logits = torch.zeros(B, R, self.num_pred_classes, device=device)
        if self.use_bias and self.freq_bias is not None:
            pred_bias = torch.zeros(B, R, self.num_pred_classes, device=device)
        else:
            pred_bias = None

        for b in range(B):
            # Skip empty images without .item() (stays on GPU)
            if relation_mask is not None and not relation_mask[b].any():
                continue

            # Get pairs for this image
            pairs = relation_pairs[b]  # [R, 2]
            if relation_mask is not None:
                valid_mask_b = relation_mask[b]
                valid_pairs = pairs[valid_mask_b]
            else:
                valid_mask_b = None
                valid_pairs = pairs

            subj_idx = valid_pairs[:, 0].long()
            obj_idx = valid_pairs[:, 1].long()

            # Gather features
            subj_features = obj_features[b, subj_idx]      # [n_rel, D]
            obj_features_rel = obj_features[b, obj_idx]    # [n_rel, D]

            subj_boxes = boxes[b, subj_idx]  # [n_rel, 4]
            obj_boxes = boxes[b, obj_idx]    # [n_rel, 4]

            # Compute union features if needed
            union_features = None
            if self.use_union_features:
                union_features = (subj_features + obj_features_rel) / 2

            # Get image size
            img_size = image_sizes[b] if image_sizes else None

            # Relation head forward
            rel_logits = self.relation_head(
                subj_features=subj_features,
                obj_features=obj_features_rel,
                subj_boxes=subj_boxes,
                obj_boxes=obj_boxes,
                union_features=union_features,
                image_size=img_size
            )  # [n_rel, num_pred]

            # Scatter into pre-allocated tensor — cast to float32 for AMP compatibility
            # (relation_head may return float16 under autocast; pre-allocated tensor is float32)
            if valid_mask_b is not None:
                pred_logits[b, valid_mask_b] = rel_logits.float()
            else:
                pred_logits[b] = rel_logits.float()

            # Frequency bias
            if pred_bias is not None:
                bias_logits = self.freq_bias(labels[b, subj_idx], labels[b, obj_idx])
                if valid_mask_b is not None:
                    pred_bias[b, valid_mask_b] = bias_logits.float()
                else:
                    pred_bias[b] = bias_logits.float()

        # Add frequency bias
        if pred_bias is not None:
            pred_logits = pred_logits + pred_bias

        # Compute probabilities
        pred_scores = F.softmax(pred_logits, dim=-1)

        # Build output
        outputs = {
            'pred_logits': pred_logits,
            'pred_scores': pred_scores
        }

        if obj_logits is not None:
            outputs['obj_logits'] = obj_logits
            outputs['obj_scores'] = F.softmax(obj_logits, dim=-1)

        # Remove batch dimension for single input
        if single_input:
            outputs = {k: v.squeeze(0) for k, v in outputs.items()}

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        relation_mask: Optional[torch.Tensor] = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses with optional focal loss and class weights.

        Args:
            outputs: Model outputs from forward()
            targets: Ground truth dict with:
                - relation_labels: [B, R] predicate labels
                - labels: [B, N] object labels (for sgcls/sgdet)
            relation_mask: [B, R] valid relation mask
            use_focal_loss: Whether to use focal loss instead of CE
            focal_gamma: Focal loss gamma parameter
            focal_alpha: Focal loss alpha parameter
            class_weights: Optional class weights for imbalanced predicates
            label_smoothing: Label smoothing factor

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Predicate classification loss
        pred_logits = outputs['pred_logits']  # [B, R, num_pred]
        gt_predicates = targets['relation_labels']  # [B, R]

        if relation_mask is not None:
            # Only compute loss on valid relations
            pred_logits_flat = pred_logits[relation_mask]
            gt_predicates_flat = gt_predicates[relation_mask]
        else:
            pred_logits_flat = pred_logits.view(-1, self.num_pred_classes)
            gt_predicates_flat = gt_predicates.view(-1)

        if len(gt_predicates_flat) > 0:
            if use_focal_loss:
                # Focal loss for hard examples
                pred_loss = focal_cross_entropy(
                    pred_logits_flat, gt_predicates_flat,
                    alpha=focal_alpha, gamma=focal_gamma,
                    weight=class_weights
                )
            else:
                # Standard cross entropy with optional class weights and label smoothing
                pred_loss = F.cross_entropy(
                    pred_logits_flat, gt_predicates_flat,
                    weight=class_weights,
                    label_smoothing=label_smoothing
                )
        else:
            pred_loss = torch.tensor(0.0, device=pred_logits.device)

        losses['pred_loss'] = pred_loss

        # Object classification loss (for sgcls/sgdet)
        if 'obj_logits' in outputs and 'labels' in targets:
            obj_logits = outputs['obj_logits']  # [B, N, num_obj]
            gt_labels = targets['labels']  # [B, N]

            if 'object_mask' in targets:
                obj_mask = targets['object_mask']
                obj_logits_flat = obj_logits[obj_mask]
                gt_labels_flat = gt_labels[obj_mask]
            else:
                obj_logits_flat = obj_logits.view(-1, self.num_obj_classes)
                gt_labels_flat = gt_labels.view(-1)

            if len(gt_labels_flat) > 0:
                if use_focal_loss:
                    obj_loss = focal_cross_entropy(
                        obj_logits_flat, gt_labels_flat,
                        alpha=focal_alpha, gamma=focal_gamma
                    )
                else:
                    obj_loss = F.cross_entropy(
                        obj_logits_flat, gt_labels_flat,
                        label_smoothing=label_smoothing
                    )
            else:
                obj_loss = torch.tensor(0.0, device=obj_logits.device)

            losses['obj_loss'] = obj_loss

        # VTransE auxiliary loss if available
        if 'vtranse_loss' in outputs:
            losses['vtranse_loss'] = outputs['vtranse_loss']

        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses

    def predict(
        self,
        obj_features: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        relation_pairs: torch.Tensor,
        top_k: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make predictions and return top-K triplets.

        Args:
            obj_features: Object visual features
            boxes: Object boxes
            labels: Object labels (GT or predicted depending on mode)
            relation_pairs: Relation pair indices
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                obj_features, boxes, labels, relation_pairs, **kwargs
            )

        pred_scores = outputs['pred_scores']  # [R, num_pred] or [B, R, num_pred]

        # Handle batched or single
        if pred_scores.dim() == 3:
            # Return predictions per image in batch
            results = []
            for b in range(pred_scores.size(0)):
                result = self._process_predictions(
                    pred_scores[b], relation_pairs[b],
                    labels[b], boxes[b], top_k
                )
                results.append(result)
            return {'batch_predictions': results}
        else:
            return self._process_predictions(
                pred_scores, relation_pairs, labels, boxes, top_k
            )

    def _process_predictions(
        self,
        pred_scores: torch.Tensor,
        relation_pairs: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        top_k: int
    ) -> Dict[str, Any]:
        """Process predictions for a single image."""
        R, num_pred = pred_scores.shape

        # Get best predicate per pair
        best_pred_scores, best_pred_idx = pred_scores.max(dim=-1)  # [R]

        # Get top-K relations
        if R > top_k:
            top_scores, top_indices = best_pred_scores.topk(top_k)
        else:
            top_scores = best_pred_scores
            top_indices = torch.arange(R, device=pred_scores.device)

        # Build triplets — batched extraction instead of per-element .item() calls
        top_idx     = top_indices.long()
        subj_idxs   = relation_pairs[top_idx, 0]
        obj_idxs    = relation_pairs[top_idx, 1]
        subj_boxes_np = boxes[subj_idxs].cpu().numpy()   # [K, 4]
        obj_boxes_np  = boxes[obj_idxs].cpu().numpy()    # [K, 4]
        subj_lbls   = labels[subj_idxs].cpu().tolist()
        obj_lbls    = labels[obj_idxs].cpu().tolist()
        pred_idxs   = best_pred_idx[top_idx].cpu().tolist()
        scores_l    = top_scores.cpu().tolist()

        triplets = [
            {
                'subject_idx': int(si), 'object_idx': int(oi),
                'subject_label': sl, 'object_label': ol,
                'predicate': pi, 'score': sc,
                'subject_box': sb.tolist(), 'object_box': ob.tolist()
            }
            for si, oi, sl, ol, pi, sc, sb, ob in zip(
                subj_idxs.tolist(), obj_idxs.tolist(),
                subj_lbls, obj_lbls, pred_idxs, scores_l,
                subj_boxes_np, obj_boxes_np
            )
        ]

        return {
            'triplets': triplets,
            'num_relations': len(triplets)
        }

    def set_bias_alpha(self, alpha: float) -> None:
        """Update frequency bias weight."""
        self.bias_alpha = alpha
        if self.freq_bias is not None:
            self.freq_bias.alpha = alpha


def create_model(config: Dict[str, Any]) -> SceneGraphModel:
    """
    Factory function to create model from config.

    Args:
        config: Configuration dictionary

    Returns:
        SceneGraphModel instance
    """
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})

    return SceneGraphModel(
        num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
        num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
        mode=config.get('mode', 'predcls'),
        feature_dim=model_cfg.get('feature_dim', 256),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        spatial_dim=model_cfg.get('spatial_dim', 128),
        use_bias=model_cfg.get('use_bias', True),
        bias_alpha=model_cfg.get('bias_alpha', 1.0),
        dropout=model_cfg.get('dropout', 0.3),
        use_union_features=model_cfg.get('use_union_features', True)
    )


class SimpleVRDModel(nn.Module):
    """
    Simplified VRD model for quick experimentation.

    Uses only object embeddings and spatial features (no visual features).
    Good for testing the pipeline without full feature extraction.
    """

    def __init__(
        self,
        num_obj_classes: int = 151,
        num_pred_classes: int = 51,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        use_spatial: bool = True
    ):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_pred_classes = num_pred_classes
        self.embedding_dim = embedding_dim
        self.use_spatial = use_spatial

        # Object embeddings
        self.obj_embed = nn.Embedding(num_obj_classes, embedding_dim)

        # Spatial encoder
        if use_spatial:
            self.spatial_encoder = SpatialEncoder(output_dim=128)
            input_dim = embedding_dim * 2 + 128
        else:
            self.spatial_encoder = None
            input_dim = embedding_dim * 2

        # Predicate classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_pred_classes)
        )

    def forward(
        self,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        relation_pairs: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            labels: [N] object labels
            boxes: [N, 4] object boxes
            relation_pairs: [R, 2] relation pairs
            image_size: Optional image size

        Returns:
            [R, num_pred] predicate logits
        """
        # Get embeddings
        obj_embeds = self.obj_embed(labels)  # [N, embed_dim]

        # Gather for pairs
        subj_idx = relation_pairs[:, 0]
        obj_idx = relation_pairs[:, 1]

        subj_embeds = obj_embeds[subj_idx]
        obj_embeds_rel = obj_embeds[obj_idx]

        features = [subj_embeds, obj_embeds_rel]

        # Spatial features
        if self.use_spatial:
            subj_boxes = boxes[subj_idx]
            obj_boxes = boxes[obj_idx]
            spatial_feats = self.spatial_encoder(subj_boxes, obj_boxes, image_size)
            features.append(spatial_feats)

        # Concatenate and classify
        x = torch.cat(features, dim=-1)
        logits = self.classifier(x)

        return logits


class MotifNetSceneGraphModel(SceneGraphModel):
    """
    MotifNet-based Scene Graph Model with BiLSTM context encoding.

    Extends SceneGraphModel with Neural Motifs (Zellers et al., CVPR 2018)
    architecture that uses bidirectional LSTMs to capture global context.

    Key differences from baseline:
    1. Objects are sorted spatially before processing
    2. BiLSTM encodes object-level context
    3. BiLSTM encodes edge-level context
    4. Context-enriched features improve predicate prediction

    Args:
        num_obj_classes: Number of object classes
        num_pred_classes: Number of predicate classes
        mode: Evaluation mode ('predcls', 'sgcls', 'sgdet')
        feature_dim: Visual feature dimension
        hidden_dim: Hidden layer dimension
        spatial_dim: Spatial feature dimension
        lstm_hidden_dim: LSTM hidden dimension
        lstm_num_layers: Number of LSTM layers
        use_object_context: Whether to use object-level BiLSTM
        use_edge_context: Whether to use edge-level BiLSTM
        use_bias: Whether to use frequency bias
        bias_alpha: Weight for frequency bias
        dropout: Dropout probability
        use_union_features: Whether to use union box features
    """

    def __init__(
        self,
        num_obj_classes: int = 151,
        num_pred_classes: int = 51,
        mode: str = 'predcls',
        feature_dim: int = 256,
        hidden_dim: int = 512,
        spatial_dim: int = 128,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        use_object_context: bool = True,
        use_edge_context: bool = True,
        use_bias: bool = True,
        bias_alpha: float = 1.0,
        dropout: float = 0.3,
        use_union_features: bool = True,
        use_vtranse: bool = False,
        vtranse_embed_dim: int = 256,
        vtranse_weight: float = 0.3
    ):
        # Initialize parent class but we'll replace relation_head
        super().__init__(
            num_obj_classes=num_obj_classes,
            num_pred_classes=num_pred_classes,
            mode=mode,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            use_bias=use_bias,
            bias_alpha=bias_alpha,
            dropout=dropout,
            use_union_features=use_union_features
        )

        # Store MotifNet-specific parameters
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.use_object_context = use_object_context
        self.use_edge_context = use_edge_context

        # VTransE auxiliary loss module
        self.use_vtranse = use_vtranse
        self.vtranse_weight = vtranse_weight
        if use_vtranse:
            self.vtranse = VTransE(
                visual_dim=feature_dim,
                embed_dim=vtranse_embed_dim,
                num_predicates=num_pred_classes,
                dropout=dropout
            )

        # Replace relation head with MotifNet version
        self.relation_head = MotifNetRelationHead(
            visual_dim=feature_dim,
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            num_predicates=num_pred_classes,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout,
            use_object_context=use_object_context,
            use_edge_context=use_edge_context,
            use_union_features=use_union_features,
            use_spatial=True
        )

    def forward(
        self,
        obj_features: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        relation_pairs: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        relation_mask: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
        relation_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MotifNet scene graph generation.

        The key difference from baseline is that MotifNetRelationHead handles
        the context encoding internally, including object sorting and BiLSTM.

        Args:
            obj_features: [B, N, D] object visual features
            boxes: [B, N, 4] object boxes
            labels: [B, N] object labels
            relation_pairs: [B, R, 2] relation pair indices
            object_mask: [B, N] valid object mask
            relation_mask: [B, R] valid relation mask
            image_sizes: List of (H, W) tuples
            relation_labels: [B, R] GT predicate labels (for VTransE aux loss during training)

        Returns:
            Dictionary with pred_logits, pred_scores, and optionally obj_logits
        """
        # Handle single image input
        single_input = obj_features.dim() == 2
        if single_input:
            obj_features = obj_features.unsqueeze(0)
            boxes = boxes.unsqueeze(0)
            labels = labels.unsqueeze(0)
            relation_pairs = relation_pairs.unsqueeze(0)
            if object_mask is not None:
                object_mask = object_mask.unsqueeze(0)
            if relation_mask is not None:
                relation_mask = relation_mask.unsqueeze(0)
            if relation_labels is not None:
                relation_labels = relation_labels.unsqueeze(0)

        B, N, D = obj_features.shape
        _, R, _ = relation_pairs.shape

        # Project features if needed
        if D != self.feature_dim:
            obj_features = self.feature_proj(obj_features)

        # Get object predictions (for sgcls/sgdet)
        obj_logits = None
        if self.obj_classifier is not None and self.mode != 'predcls':
            obj_logits = self.obj_classifier(obj_features)
            if self.mode in ['sgcls', 'sgdet']:
                labels = obj_logits.argmax(dim=-1)

        # Add label embeddings to visual features
        label_embeds = self.obj_embed(labels)
        obj_features_with_labels = obj_features + label_embeds

        # Compute union features for each relation pair
        # Simple approach: average subject and object features
        union_features = self._compute_union_features(
            obj_features_with_labels, relation_pairs
        )

        # Get image size for spatial features
        image_size = image_sizes[0] if image_sizes else None

        # MotifNet relation head handles context encoding internally
        pred_logits = self.relation_head(
            obj_features=obj_features_with_labels,
            boxes=boxes,
            pair_indices=relation_pairs,
            union_features=union_features,
            object_mask=object_mask,
            relation_mask=relation_mask,
            image_size=image_size
        )

        # Add frequency bias
        if self.use_bias and self.freq_bias is not None:
            pred_bias = self._compute_frequency_bias(
                labels, relation_pairs, relation_mask
            )
            pred_logits = pred_logits + pred_bias

        # Compute probabilities
        pred_scores = F.softmax(pred_logits, dim=-1)

        # Build output
        outputs = {
            'pred_logits': pred_logits,
            'pred_scores': pred_scores
        }

        if obj_logits is not None:
            outputs['obj_logits'] = obj_logits
            outputs['obj_scores'] = F.softmax(obj_logits, dim=-1)

        # VTransE auxiliary loss (training only)
        if self.use_vtranse and self.training and relation_labels is not None:
            vtranse_loss = self._compute_vtranse_loss(
                obj_features_with_labels, relation_pairs,
                relation_labels, relation_mask
            )
            outputs['vtranse_loss'] = self.vtranse_weight * vtranse_loss

        # Remove batch dimension for single input
        if single_input:
            outputs = {k: v.squeeze(0) for k, v in outputs.items()}

        return outputs

    def _compute_vtranse_loss(
        self,
        obj_features: torch.Tensor,
        relation_pairs: torch.Tensor,
        relation_labels: torch.Tensor,
        relation_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute VTransE auxiliary loss over valid relations in batch (vectorized)."""
        B, N, D = obj_features.shape
        _, R, _ = relation_pairs.shape
        device = obj_features.device

        # Batch advanced indexing — no per-image loop
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, R)
        subj_idx  = relation_pairs[:, :, 0].long()
        obj_idx   = relation_pairs[:, :, 1].long()

        all_subj = obj_features[batch_idx, subj_idx]   # [B, R, D]
        all_obj  = obj_features[batch_idx, obj_idx]    # [B, R, D]

        if relation_mask is not None:
            flat_mask   = relation_mask.reshape(-1)                   # [B*R]
            subj_feats  = all_subj.reshape(B * R, D)[flat_mask]
            obj_feats   = all_obj.reshape(B * R, D)[flat_mask]
            pred_labels = relation_labels.reshape(-1)[flat_mask]
        else:
            subj_feats  = all_subj.reshape(B * R, D)
            obj_feats   = all_obj.reshape(B * R, D)
            pred_labels = relation_labels.reshape(-1)

        if subj_feats.shape[0] == 0:
            return torch.tensor(0.0, device=device)

        return self.vtranse.compute_loss(subj_feats, obj_feats, pred_labels)

    def _compute_union_features(
        self,
        obj_features: torch.Tensor,
        relation_pairs: torch.Tensor
    ) -> torch.Tensor:
        """Compute union features for relation pairs."""
        B, N, D = obj_features.shape
        _, R, _ = relation_pairs.shape
        device = obj_features.device

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, R)
        subj_idx = relation_pairs[:, :, 0]
        obj_idx = relation_pairs[:, :, 1]

        subj_features = obj_features[batch_idx, subj_idx]
        obj_features_rel = obj_features[batch_idx, obj_idx]

        # Simple union: average of subject and object
        union_features = (subj_features + obj_features_rel) / 2

        return union_features

    def _compute_frequency_bias(
        self,
        labels: torch.Tensor,
        relation_pairs: torch.Tensor,
        relation_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute frequency bias for all relations in batch (vectorized)."""
        B, N = labels.shape
        _, R, _ = relation_pairs.shape
        device = labels.device

        # Batched advanced indexing — single freq_bias call on [B*R] instead of B calls on [R]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, R)
        subj_idx  = relation_pairs[:, :, 0].long()
        obj_idx   = relation_pairs[:, :, 1].long()
        subj_flat = labels[batch_idx, subj_idx].reshape(B * R)
        obj_flat  = labels[batch_idx, obj_idx].reshape(B * R)
        bias_flat = self.freq_bias(subj_flat, obj_flat)          # [B*R, num_pred]
        return bias_flat.view(B, R, -1)


def create_motifnet_model(config: Dict[str, Any]) -> MotifNetSceneGraphModel:
    """
    Factory function to create MotifNet model from config.

    Args:
        config: Configuration dictionary

    Returns:
        MotifNetSceneGraphModel instance
    """
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})

    return MotifNetSceneGraphModel(
        num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
        num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
        mode=config.get('mode', 'predcls'),
        feature_dim=model_cfg.get('feature_dim', 256),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        spatial_dim=model_cfg.get('spatial_dim', 128),
        lstm_hidden_dim=model_cfg.get('lstm_hidden_dim', 256),
        lstm_num_layers=model_cfg.get('lstm_num_layers', 2),
        use_object_context=model_cfg.get('use_object_context', True),
        use_edge_context=model_cfg.get('use_edge_context', True),
        use_bias=model_cfg.get('use_bias', True),
        bias_alpha=model_cfg.get('bias_alpha', 1.0),
        dropout=model_cfg.get('dropout', 0.3),
        use_union_features=model_cfg.get('use_union_features', True),
        use_vtranse=model_cfg.get('use_vtranse', False),
        vtranse_embed_dim=model_cfg.get('vtranse_embed_dim', 256),
        vtranse_weight=model_cfg.get('vtranse_weight', 0.3)
    )


class ZeroShotSceneGraphModel(SceneGraphModel):
    """
    Zero-Shot Scene Graph Model using frozen CLIP features + LLM Semantic Bias.

    Extends SceneGraphModel with:
    1. Feature projection from 512 (CLIP dim) instead of 1024 (CNN dim)
    2. LLM semantic bias instead of frequency bias
    3. CLIP text embeddings for object label initialization

    Only the relation head, feature projection, and bias temperature are trainable.
    CLIP backbone and LLM priors are frozen/static.

    Args:
        num_obj_classes: Number of object classes
        num_pred_classes: Number of predicate classes
        mode: Evaluation mode ('predcls', 'sgcls', 'sgdet')
        clip_feature_dim: CLIP output dimension (512 for ViT-B/32)
        feature_dim: Internal feature dimension
        hidden_dim: MLP hidden dimension
        spatial_dim: Spatial feature dimension
        use_bias: Use LLM semantic bias
        bias_alpha: Weight for bias
        learnable_temperature: Whether bias temperature is learnable
        initial_temperature: Initial temperature value
        dropout: Dropout probability
        use_union_features: Use union box features
    """

    def __init__(
        self,
        num_obj_classes: int = 151,
        num_pred_classes: int = 51,
        mode: str = 'predcls',
        clip_feature_dim: int = 512,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        spatial_dim: int = 128,
        use_bias: bool = True,
        bias_alpha: float = 1.0,
        learnable_temperature: bool = True,
        initial_temperature: float = 1.0,
        dropout: float = 0.3,
        use_union_features: bool = True
    ):
        # Initialize parent with use_bias=False first
        # (we'll set up our own bias below)
        super().__init__(
            num_obj_classes=num_obj_classes,
            num_pred_classes=num_pred_classes,
            mode=mode,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            spatial_dim=spatial_dim,
            use_bias=False,  # Disable parent's FrequencyBias
            bias_alpha=bias_alpha,
            dropout=dropout,
            use_union_features=use_union_features
        )

        self.clip_feature_dim = clip_feature_dim

        # Override feature projection: CLIP 512 → feature_dim
        # (parent creates 1024 → feature_dim)
        # LayerNorm stabilizes CLIP's L2-normalized features after projection
        self.feature_proj = nn.Sequential(
            nn.Linear(clip_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Setup LLM semantic bias
        self.use_bias = use_bias
        if use_bias:
            from .llm_semantic_bias import LLMSemanticBias
            self.freq_bias = LLMSemanticBias(
                num_obj_classes=num_obj_classes,
                num_pred_classes=num_pred_classes,
                alpha=bias_alpha,
                learnable_temperature=learnable_temperature,
                initial_temperature=initial_temperature
            )
        else:
            self.freq_bias = None

        # Re-initialize weights for the new feature_proj
        for m in self.feature_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"ZeroShotSceneGraphModel: {trainable:,} trainable / {total:,} total params")

    def initialize_clip_embeddings(
        self,
        clip_text_embeddings: torch.Tensor
    ) -> None:
        """
        Initialize object label embeddings from CLIP text features.

        Args:
            clip_text_embeddings: [num_obj_classes, 512] CLIP text embeddings
        """
        with torch.no_grad():
            # Project CLIP 512-dim text embeddings to feature_dim
            # Use a simple linear projection
            proj = nn.Linear(clip_text_embeddings.size(1), self.feature_dim, bias=False)
            nn.init.orthogonal_(proj.weight)
            projected = proj(clip_text_embeddings)

            # Copy into obj_embed
            self.obj_embed.weight.copy_(projected)
            print(f"Initialized obj_embed from CLIP text embeddings "
                  f"({clip_text_embeddings.shape} → {projected.shape})")


def create_zeroshot_model(config: Dict[str, Any]) -> ZeroShotSceneGraphModel:
    """
    Factory function to create Zero-Shot model from config.

    Args:
        config: Configuration dictionary

    Returns:
        ZeroShotSceneGraphModel instance
    """
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})

    return ZeroShotSceneGraphModel(
        num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
        num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
        mode=config.get('mode', 'predcls'),
        clip_feature_dim=model_cfg.get('clip_feature_dim', 512),
        feature_dim=model_cfg.get('feature_dim', 256),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        spatial_dim=model_cfg.get('spatial_dim', 128),
        use_bias=model_cfg.get('use_bias', True),
        bias_alpha=model_cfg.get('bias_alpha', 1.0),
        learnable_temperature=model_cfg.get('learnable_temperature', True),
        initial_temperature=model_cfg.get('initial_temperature', 1.0),
        dropout=model_cfg.get('dropout', 0.3),
        use_union_features=model_cfg.get('use_union_features', True)
    )

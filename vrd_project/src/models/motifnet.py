"""
MotifNet (Neural Motifs) Implementation for VRD
=================================================

Implements the Neural Motifs architecture (Zellers et al., CVPR 2018) for
Visual Relationship Detection with bidirectional LSTMs for global context.

Key components:
1. ObjectSorter: Sorts objects by spatial position for sequential processing
2. ObjectContextEncoder: BiLSTM for object-level global context
3. EdgeContextEncoder: BiLSTM for relation-level global context
4. MotifNetRelationHead: Complete relation head with context modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, List, Tuple, Optional, Any
import math

from .relation_head import SpatialEncoder


def sort_objects_by_position(
    boxes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sort objects by their spatial position (top-to-bottom, left-to-right).

    This ordering is important for MotifNet as it provides a consistent
    sequential structure for the BiLSTM to process.

    Args:
        boxes: [N, 4] boxes in [x1, y1, x2, y2] format

    Returns:
        sorted_indices: [N] indices to sort objects
        inverse_indices: [N] indices to unsort back to original order
    """
    if boxes.size(0) == 0:
        return (
            torch.tensor([], dtype=torch.long, device=boxes.device),
            torch.tensor([], dtype=torch.long, device=boxes.device)
        )

    # Compute center coordinates
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2

    # Sort by y first (top to bottom), then x (left to right)
    # Multiply y by a large number to prioritize vertical ordering
    # Assumes image width < 10000 pixels
    sort_key = cy * 10000 + cx
    sorted_indices = torch.argsort(sort_key)

    # Compute inverse permutation to unsort
    inverse_indices = torch.zeros_like(sorted_indices)
    inverse_indices[sorted_indices] = torch.arange(
        len(sorted_indices), device=boxes.device
    )

    return sorted_indices, inverse_indices


class ObjectContextEncoder(nn.Module):
    """
    Bidirectional LSTM for encoding global object-level context.

    Processes object features in spatial order to capture scene-level
    dependencies. Each object's output representation includes information
    about all other objects in the scene.

    Args:
        input_dim: Dimension of input object features
        hidden_dim: LSTM hidden dimension (will be split for bidirectional)
        num_layers: Number of LSTM layers
        dropout: Dropout probability between LSTM layers
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM - hidden_dim//2 because bidirectional doubles it
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output projection (optional, for matching dimensions)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights with orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

    def forward(
        self,
        obj_features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode object features with bidirectional LSTM.

        Args:
            obj_features: [B, N, input_dim] object features (should be sorted)
            lengths: [B] number of valid objects per image (for packing)

        Returns:
            [B, N, hidden_dim] contextualized object features
        """
        B, N, D = obj_features.shape

        if lengths is not None:
            # Pack sequences for efficient processing of variable lengths
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = pack_padded_sequence(
                obj_features, lengths_cpu,
                batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(output, batch_first=True, total_length=N)
        else:
            output, _ = self.lstm(obj_features)

        # Apply layer norm and projection
        output = self.layer_norm(output)
        output = self.output_proj(output)

        return output


class EdgeContextEncoder(nn.Module):
    """
    Bidirectional LSTM for encoding global edge-level (relation) context.

    Processes relation features to capture dependencies between different
    relationships in the scene.

    Args:
        input_dim: Dimension of input edge features
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection to normalize feature dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

    def forward(
        self,
        edge_features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode edge features with bidirectional LSTM.

        Args:
            edge_features: [B, R, input_dim] edge features
            lengths: [B] number of valid relations per image

        Returns:
            [B, R, hidden_dim] contextualized edge features
        """
        B, R, D = edge_features.shape

        # Project input features
        x = self.input_proj(edge_features)

        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = pack_padded_sequence(
                x, lengths_cpu,
                batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(output, batch_first=True, total_length=R)
        else:
            output, _ = self.lstm(x)

        # Apply layer norm
        output = self.layer_norm(output)

        return output


class MotifNetRelationHead(nn.Module):
    """
    MotifNet-style relation prediction head with BiLSTM context encoding.

    Pipeline:
    1. Sort objects spatially
    2. Encode object-level context with BiLSTM
    3. Gather pair features from contextualized objects
    4. Add spatial and union features
    5. Encode edge-level context with BiLSTM
    6. Classify predicates

    Args:
        visual_dim: Dimension of visual features
        spatial_dim: Dimension of spatial features
        hidden_dim: Hidden layer dimension for classifier
        num_predicates: Number of predicate classes
        lstm_hidden_dim: LSTM hidden dimension
        lstm_num_layers: Number of LSTM layers
        dropout: Dropout probability
        use_object_context: Whether to use object-level BiLSTM
        use_edge_context: Whether to use edge-level BiLSTM
        use_union_features: Whether to include union box features
        use_spatial: Whether to include spatial features
    """

    def __init__(
        self,
        visual_dim: int = 256,
        spatial_dim: int = 128,
        hidden_dim: int = 512,
        num_predicates: int = 51,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        use_object_context: bool = True,
        use_edge_context: bool = True,
        use_union_features: bool = True,
        use_spatial: bool = True
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.num_predicates = num_predicates
        self.lstm_hidden_dim = lstm_hidden_dim
        self.use_object_context = use_object_context
        self.use_edge_context = use_edge_context
        self.use_union_features = use_union_features
        self.use_spatial = use_spatial

        # Spatial encoder (reuse from relation_head)
        if use_spatial:
            self.spatial_encoder = SpatialEncoder(output_dim=spatial_dim)
        else:
            self.spatial_encoder = None

        # Object context encoder (BiLSTM)
        if use_object_context:
            self.obj_context_encoder = ObjectContextEncoder(
                input_dim=visual_dim,
                hidden_dim=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                dropout=dropout
            )
            obj_feature_dim = lstm_hidden_dim
        else:
            self.obj_context_encoder = None
            obj_feature_dim = visual_dim

        # Compute pair feature dimension
        pair_feature_dim = obj_feature_dim * 2  # subject + object
        if use_union_features:
            pair_feature_dim += visual_dim
        if use_spatial:
            pair_feature_dim += spatial_dim

        self.pair_feature_dim = pair_feature_dim

        # Edge context encoder (BiLSTM)
        if use_edge_context:
            self.edge_context_encoder = EdgeContextEncoder(
                input_dim=pair_feature_dim,
                hidden_dim=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                dropout=dropout
            )
            classifier_input_dim = lstm_hidden_dim
        else:
            self.edge_context_encoder = None
            classifier_input_dim = pair_feature_dim

        # Predicate classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_predicates)
        )

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        obj_features: torch.Tensor,
        boxes: torch.Tensor,
        pair_indices: torch.Tensor,
        union_features: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        relation_mask: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Forward pass for MotifNet relation prediction.

        Args:
            obj_features: [B, N, visual_dim] object visual features
            boxes: [B, N, 4] object bounding boxes
            pair_indices: [B, R, 2] relation pair indices (subject, object)
            union_features: [B, R, visual_dim] optional union box features
            object_mask: [B, N] mask for valid objects
            relation_mask: [B, R] mask for valid relations
            image_size: (height, width) for spatial feature normalization

        Returns:
            [B, R, num_predicates] predicate logits
        """
        B, N, D = obj_features.shape
        _, R, _ = pair_indices.shape
        device = obj_features.device

        # Step 1: Process each image - sort objects and apply context encoding
        if self.use_object_context:
            context_features = self._encode_object_context(
                obj_features, boxes, object_mask
            )
        else:
            context_features = obj_features

        # Step 2: Gather pair features
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, R)
        subj_idx = pair_indices[:, :, 0]
        obj_idx = pair_indices[:, :, 1]

        subj_features = context_features[batch_idx, subj_idx]  # [B, R, D]
        obj_features_rel = context_features[batch_idx, obj_idx]  # [B, R, D]

        # Collect pair features
        pair_features = [subj_features, obj_features_rel]

        # Add union features
        if self.use_union_features and union_features is not None:
            pair_features.append(union_features)

        # Add spatial features
        if self.use_spatial:
            subj_boxes = boxes[batch_idx, subj_idx]  # [B, R, 4]
            obj_boxes = boxes[batch_idx, obj_idx]    # [B, R, 4]
            spatial_feats = self._compute_batch_spatial(
                subj_boxes, obj_boxes, image_size
            )
            pair_features.append(spatial_feats)

        # Concatenate all pair features
        edge_features = torch.cat(pair_features, dim=-1)  # [B, R, pair_dim]

        # Step 3: Apply edge context encoding
        if self.use_edge_context:
            # Get relation lengths for packing
            if relation_mask is not None:
                rel_lengths = relation_mask.sum(dim=1)
            else:
                rel_lengths = None

            edge_features = self.edge_context_encoder(edge_features, rel_lengths)

        # Step 4: Classify predicates
        logits = self.classifier(edge_features)

        return logits

    def _encode_object_context(
        self,
        obj_features: torch.Tensor,
        boxes: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode object-level context using BiLSTM with spatial sorting.

        Args:
            obj_features: [B, N, D] object features
            boxes: [B, N, 4] bounding boxes
            object_mask: [B, N] valid object mask

        Returns:
            [B, N, lstm_hidden_dim] contextualized features
        """
        B, N, D = obj_features.shape
        device = obj_features.device

        # Process each image in the batch
        sorted_features_list = []
        inverse_indices_list = []
        lengths_list = []

        for b in range(B):
            if object_mask is not None:
                valid_n = max(1, int(object_mask[b].sum()))  # int() avoids CUDA sync barrier
            else:
                valid_n = N

            # Sort objects by spatial position
            sort_idx, inv_idx = sort_objects_by_position(boxes[b, :valid_n])

            # Reorder features
            sorted_feats = obj_features[b, sort_idx]

            # Pad to N
            if valid_n < N:
                padding = torch.zeros(N - valid_n, D, device=device)
                sorted_feats = torch.cat([sorted_feats, padding], dim=0)

                # Extend inverse indices
                inv_idx_padded = torch.zeros(N, dtype=torch.long, device=device)
                inv_idx_padded[:valid_n] = inv_idx
                inv_idx = inv_idx_padded

            sorted_features_list.append(sorted_feats)
            inverse_indices_list.append(inv_idx)
            lengths_list.append(valid_n)

        # Stack batch
        sorted_features = torch.stack(sorted_features_list)  # [B, N, D]
        lengths = torch.tensor(lengths_list, device=device)

        # Apply BiLSTM
        context_features = self.obj_context_encoder(sorted_features, lengths)

        # Unsort to original order
        output = torch.zeros_like(context_features)
        for b in range(B):
            inv_idx = inverse_indices_list[b]
            valid_n = lengths_list[b]
            output[b, :valid_n] = context_features[b, inv_idx[:valid_n]]

        return output

    def _compute_batch_spatial(
        self,
        subj_boxes: torch.Tensor,
        obj_boxes: torch.Tensor,
        image_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Compute spatial features for batched box pairs.

        Args:
            subj_boxes: [B, R, 4] subject boxes
            obj_boxes: [B, R, 4] object boxes
            image_size: Optional image size for normalization

        Returns:
            [B, R, spatial_dim] spatial features
        """
        B, R, _ = subj_boxes.shape
        device = subj_boxes.device

        # Flatten batch+relation dims → single encoder call → reshape back
        sf = self.spatial_encoder(
            subj_boxes.view(B * R, 4),
            obj_boxes.view(B * R, 4),
            image_size
        )  # [B*R, spatial_dim]
        return sf.view(B, R, -1)


class MotifNetDecoder(nn.Module):
    """
    Alternative MotifNet decoder that uses separate decoders for
    object labels and predicates.

    This follows the original Neural Motifs paper more closely,
    with separate LSTM passes for objects and relations.
    """

    def __init__(
        self,
        num_obj_classes: int = 151,
        num_pred_classes: int = 51,
        obj_embed_dim: int = 200,
        visual_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_pred_classes = num_pred_classes

        # Object label embedding
        self.obj_embed = nn.Embedding(num_obj_classes, obj_embed_dim)

        # Object context LSTM (for refining object predictions)
        self.obj_ctx_lstm = nn.LSTM(
            input_size=visual_dim + obj_embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Object classifier
        self.obj_classifier = nn.Linear(hidden_dim, num_obj_classes)

        # Edge context LSTM
        self.edge_ctx_lstm = nn.LSTM(
            input_size=hidden_dim * 2,  # subj + obj context
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Predicate classifier
        self.pred_classifier = nn.Linear(hidden_dim, num_pred_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.obj_embed.weight)
        nn.init.xavier_uniform_(self.obj_classifier.weight)
        nn.init.zeros_(self.obj_classifier.bias)
        nn.init.xavier_uniform_(self.pred_classifier.weight)
        nn.init.zeros_(self.pred_classifier.bias)

    def forward(
        self,
        obj_features: torch.Tensor,
        obj_labels: torch.Tensor,
        pair_indices: torch.Tensor,
        obj_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obj_features: [B, N, visual_dim] object features
            obj_labels: [B, N] object label predictions (or GT in PredCls)
            pair_indices: [B, R, 2] relation pairs
            obj_mask: [B, N] valid object mask

        Returns:
            obj_logits: [B, N, num_obj] refined object logits
            pred_logits: [B, R, num_pred] predicate logits
        """
        B, N, D = obj_features.shape
        _, R, _ = pair_indices.shape

        # Get object embeddings
        obj_embeds = self.obj_embed(obj_labels)  # [B, N, embed_dim]

        # Concatenate visual and label features
        obj_input = torch.cat([obj_features, obj_embeds], dim=-1)

        # Object context LSTM
        obj_ctx, _ = self.obj_ctx_lstm(obj_input)  # [B, N, hidden]

        # Refined object predictions
        obj_logits = self.obj_classifier(obj_ctx)

        # Gather context features for pairs
        batch_idx = torch.arange(B, device=obj_features.device).unsqueeze(1).expand(-1, R)
        subj_idx = pair_indices[:, :, 0]
        obj_idx = pair_indices[:, :, 1]

        subj_ctx = obj_ctx[batch_idx, subj_idx]
        obj_ctx_rel = obj_ctx[batch_idx, obj_idx]

        # Edge input
        edge_input = torch.cat([subj_ctx, obj_ctx_rel], dim=-1)

        # Edge context LSTM
        edge_ctx, _ = self.edge_ctx_lstm(edge_input)

        # Predicate predictions
        pred_logits = self.pred_classifier(edge_ctx)

        return obj_logits, pred_logits

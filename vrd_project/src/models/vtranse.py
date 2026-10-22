"""
VTransE: Visual Translation Embedding for Visual Relationship Detection.
========================================================================

Implements the translation embedding approach where relationships are modeled as:
    subject_embedding + predicate_embedding ≈ object_embedding

Reference: Zhang et al., "Visual Translation Embedding Network for Visual
           Relation Detection", CVPR 2017

This module can be used as:
1. A standalone relation prediction head
2. An auxiliary loss to augment existing models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class VTransE(nn.Module):
    """
    Visual Translation Embedding module.

    The core idea is that in embedding space:
        f(subject) + t(predicate) ≈ f(object)

    Where f() transforms visual features and t() is the predicate embedding.

    Args:
        visual_dim: Dimension of input visual features
        embed_dim: Dimension of translation embedding space
        num_predicates: Number of predicate classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        visual_dim: int = 256,
        embed_dim: int = 256,
        num_predicates: int = 51,
        dropout: float = 0.3
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.embed_dim = embed_dim
        self.num_predicates = num_predicates

        # Transform visual features to embedding space
        self.subj_transform = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        self.obj_transform = nn.Sequential(
            nn.Linear(visual_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # Predicate translation embeddings
        self.pred_embed = nn.Embedding(num_predicates, embed_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.pred_embed.weight)

    def forward(
        self,
        subj_features: torch.Tensor,
        obj_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute predicate scores using translation embedding.

        For each (subject, object) pair, scores all predicates based on
        how well subject + predicate ≈ object holds in embedding space.

        Args:
            subj_features: Subject visual features [N, visual_dim]
            obj_features: Object visual features [N, visual_dim]

        Returns:
            Predicate logits [N, num_predicates]
        """
        # Transform to embedding space
        subj_embed = self.subj_transform(subj_features)  # [N, embed_dim]
        obj_embed = self.obj_transform(obj_features)      # [N, embed_dim]

        # Get all predicate embeddings
        pred_embeds = self.pred_embed.weight  # [num_pred, embed_dim]

        # Compute translation scores
        # For each pair, compute: -||subj + pred - obj||^2 for all predicates
        # Higher score = better match

        # Expand for broadcasting
        # subj_embed: [N, 1, embed_dim]
        # pred_embeds: [1, num_pred, embed_dim]
        # obj_embed: [N, 1, embed_dim]

        subj_expanded = subj_embed.unsqueeze(1)  # [N, 1, D]
        obj_expanded = obj_embed.unsqueeze(1)     # [N, 1, D]
        pred_expanded = pred_embeds.unsqueeze(0)  # [1, P, D]

        # Translation: subj + pred should equal obj
        translation = subj_expanded + pred_expanded  # [N, P, D]
        diff = translation - obj_expanded            # [N, P, D]

        # Negative L2 distance as score (higher = better)
        scores = -torch.sum(diff ** 2, dim=-1)  # [N, P]

        return scores

    def compute_loss(
        self,
        subj_features: torch.Tensor,
        obj_features: torch.Tensor,
        pred_labels: torch.Tensor,
        margin: float = 1.0,
        num_negatives: int = 3
    ) -> torch.Tensor:
        """
        Compute VTransE margin ranking loss.

        Uses margin-based ranking to ensure correct predicate has higher
        score than incorrect predicates by at least a margin.

        Args:
            subj_features: Subject features [N, visual_dim]
            obj_features: Object features [N, visual_dim]
            pred_labels: Ground truth predicate labels [N]
            margin: Margin for ranking loss
            num_negatives: Number of negative samples per positive

        Returns:
            Ranking loss value
        """
        N = subj_features.size(0)
        device = subj_features.device

        # Get embeddings
        subj_embed = self.subj_transform(subj_features)
        obj_embed = self.obj_transform(obj_features)

        # Positive scores (correct predicates)
        pos_pred_embed = self.pred_embed(pred_labels)  # [N, D]
        pos_translation = subj_embed + pos_pred_embed
        pos_dist = torch.sum((pos_translation - obj_embed) ** 2, dim=-1).clamp(max=100.0)  # [N]

        # Sample negative predicates
        neg_labels = torch.randint(
            0, self.num_predicates, (N, num_negatives), device=device
        )

        # Ensure negatives are different from positives
        for i in range(num_negatives):
            mask = neg_labels[:, i] == pred_labels
            while mask.any():
                neg_labels[mask, i] = torch.randint(
                    0, self.num_predicates, (mask.sum(),), device=device
                )
                mask = neg_labels[:, i] == pred_labels

        # Compute negative distances
        total_loss = 0.0
        for i in range(num_negatives):
            neg_pred_embed = self.pred_embed(neg_labels[:, i])
            neg_translation = subj_embed + neg_pred_embed
            neg_dist = torch.sum((neg_translation - obj_embed) ** 2, dim=-1).clamp(max=100.0)

            # Margin ranking loss: pos_dist + margin < neg_dist
            # max(0, pos_dist - neg_dist + margin)
            loss = F.relu(pos_dist - neg_dist + margin)
            total_loss = total_loss + loss.mean()

        return total_loss / num_negatives


class VTransEHead(nn.Module):
    """
    VTransE-enhanced relation prediction head.

    Combines VTransE translation scores with visual feature classification.

    Args:
        visual_dim: Dimension of visual features
        embed_dim: VTransE embedding dimension
        hidden_dim: Hidden layer dimension
        num_predicates: Number of predicate classes
        vtranse_weight: Weight for VTransE scores (0-1)
        dropout: Dropout probability
    """

    def __init__(
        self,
        visual_dim: int = 256,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_predicates: int = 51,
        vtranse_weight: float = 0.5,
        dropout: float = 0.3
    ):
        super().__init__()
        self.vtranse_weight = vtranse_weight

        # VTransE module
        self.vtranse = VTransE(
            visual_dim=visual_dim,
            embed_dim=embed_dim,
            num_predicates=num_predicates,
            dropout=dropout
        )

        # Visual classifier (standard MLP)
        self.visual_classifier = nn.Sequential(
            nn.Linear(visual_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicates)
        )

    def forward(
        self,
        subj_features: torch.Tensor,
        obj_features: torch.Tensor,
        union_features: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute combined predicate scores.

        Args:
            subj_features: Subject features [N, visual_dim]
            obj_features: Object features [N, visual_dim]
            union_features: Optional union box features (unused in VTransE)
            spatial_features: Optional spatial features (unused in VTransE)

        Returns:
            Tuple of (combined_logits, vtranse_logits)
        """
        # VTransE scores
        vtranse_scores = self.vtranse(subj_features, obj_features)

        # Visual classifier scores
        concat_features = torch.cat([subj_features, obj_features], dim=-1)
        visual_scores = self.visual_classifier(concat_features)

        # Combine scores
        combined = (1 - self.vtranse_weight) * visual_scores + \
                   self.vtranse_weight * vtranse_scores

        return combined, vtranse_scores

    def compute_auxiliary_loss(
        self,
        subj_features: torch.Tensor,
        obj_features: torch.Tensor,
        pred_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VTransE auxiliary loss for training.

        Can be added to the main classification loss.
        """
        return self.vtranse.compute_loss(
            subj_features, obj_features, pred_labels
        )


class VTransELoss(nn.Module):
    """
    Combined loss for VTransE training.

    Combines classification loss with VTransE margin ranking loss.

    Args:
        vtranse_weight: Weight for VTransE loss component
        margin: Margin for ranking loss
    """

    def __init__(
        self,
        vtranse_weight: float = 0.3,
        margin: float = 1.0
    ):
        super().__init__()
        self.vtranse_weight = vtranse_weight
        self.margin = margin

    def forward(
        self,
        vtranse_module: VTransE,
        pred_logits: torch.Tensor,
        subj_features: torch.Tensor,
        obj_features: torch.Tensor,
        pred_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            vtranse_module: VTransE module for computing ranking loss
            pred_logits: Predicted logits [N, num_pred]
            subj_features: Subject features [N, D]
            obj_features: Object features [N, D]
            pred_labels: Ground truth labels [N]

        Returns:
            Dictionary with 'total', 'cls', and 'vtranse' losses
        """
        # Classification loss
        cls_loss = F.cross_entropy(pred_logits, pred_labels)

        # VTransE ranking loss
        vtranse_loss = vtranse_module.compute_loss(
            subj_features, obj_features, pred_labels,
            margin=self.margin
        )

        # Combined loss
        total_loss = cls_loss + self.vtranse_weight * vtranse_loss

        return {
            'total': total_loss,
            'cls': cls_loss,
            'vtranse': vtranse_loss
        }

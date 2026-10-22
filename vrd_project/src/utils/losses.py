"""
Custom loss functions for VRD training.
============================================

Includes:
- FocalLoss: For handling class imbalance with hard example mining
- compute_class_weights: For inverse frequency weighting
- LabelSmoothingCrossEntropy: For regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses training on hard, misclassified examples by down-weighting
    well-classified examples.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    Args:
        alpha: Weighting factor for positive/negative examples (default: 0.25)
        gamma: Focusing parameter (default: 2.0, higher = more focus on hard examples)
        reduction: 'none', 'mean', or 'sum'
        class_weights: Optional per-class weights [num_classes]
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('class_weights', class_weights)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Logits of shape [N, C] where C is num_classes
            targets: Ground truth labels of shape [N]

        Returns:
            Focal loss value
        """
        # Compute standard cross entropy (no reduction)
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,
            reduction='none'
        )

        # Get probability of correct class
        p_t = torch.exp(-ce_loss)

        # Compute focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Smooths one-hot targets to prevent overconfidence:
    y_smooth = (1 - smoothing) * y_onehot + smoothing / num_classes

    Args:
        smoothing: Label smoothing factor (default: 0.1)
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label smoothing cross entropy.

        Args:
            inputs: Logits of shape [N, C]
            targets: Ground truth labels of shape [N]

        Returns:
            Loss value
        """
        num_classes = inputs.size(-1)

        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # Compute loss
        loss = (-smooth_targets * log_probs).sum(dim=-1)

        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    method: str = 'inverse_freq',
    smoothing: float = 1.0
) -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.

    Args:
        labels: Tensor of class labels
        num_classes: Total number of classes
        method: Weighting method
            - 'inverse_freq': 1 / (count + smoothing)
            - 'inverse_sqrt': 1 / sqrt(count + smoothing)
            - 'effective_samples': (1 - beta^n) / (1 - beta)
        smoothing: Smoothing factor to prevent division by zero

    Returns:
        Class weights tensor of shape [num_classes]
    """
    # Count occurrences of each class
    counts = torch.bincount(labels, minlength=num_classes).float()

    if method == 'inverse_freq':
        # Inverse frequency weighting
        weights = 1.0 / (counts + smoothing)

    elif method == 'inverse_sqrt':
        # Inverse square root frequency (less aggressive)
        weights = 1.0 / torch.sqrt(counts + smoothing)

    elif method == 'effective_samples':
        # Effective number of samples (Cui et al., CVPR 2019)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to have mean 1
    weights = weights / weights.mean()

    return weights


def compute_predicate_weights_from_bias(
    freq_matrix: torch.Tensor,
    method: str = 'inverse_freq'
) -> torch.Tensor:
    """
    Compute predicate weights from frequency bias matrix.

    Args:
        freq_matrix: Frequency matrix of shape [num_obj, num_obj, num_pred]
                    or [num_obj * num_obj, num_pred]
        method: Weighting method

    Returns:
        Predicate weights of shape [num_pred]
    """
    # Sum over all (subject, object) pairs to get predicate frequencies
    if freq_matrix.dim() == 3:
        pred_counts = freq_matrix.sum(dim=(0, 1))
    else:
        pred_counts = freq_matrix.sum(dim=0)

    return compute_class_weights(
        torch.arange(len(pred_counts)),
        len(pred_counts),
        method=method
    ).to(freq_matrix.device)


class CombinedLoss(nn.Module):
    """
    Combined loss with focal loss, class weights, and label smoothing.

    Args:
        num_classes: Number of classes
        focal_gamma: Focal loss gamma (0 disables focal loss)
        focal_alpha: Focal loss alpha
        label_smoothing: Label smoothing factor (0 disables)
        class_weights: Optional class weights tensor
    """

    def __init__(
        self,
        num_classes: int,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing
        self.register_buffer('class_weights', class_weights)

        # Use focal loss if gamma > 0
        if focal_gamma > 0:
            self.loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                class_weights=class_weights
            )
        elif label_smoothing > 0:
            self.loss_fn = LabelSmoothingCrossEntropy(
                smoothing=label_smoothing
            )
        else:
            self.loss_fn = None  # Use standard cross entropy

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss."""
        if self.loss_fn is not None:
            return self.loss_fn(inputs, targets)
        else:
            return F.cross_entropy(
                inputs, targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )


# Convenience functions for easy integration

def focal_cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Functional interface for focal loss.

    Args:
        inputs: Logits [N, C]
        targets: Labels [N]
        alpha: Focal loss alpha
        gamma: Focal loss gamma
        weight: Optional class weights
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Loss value
    """
    ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_weight = alpha * (1 - p_t) ** gamma
    loss = focal_weight * ce_loss

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

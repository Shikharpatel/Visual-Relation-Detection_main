"""
Frequency Bias Module for VRD
==============================

Computes prior predicate probabilities based on subject-object class frequencies.
P(predicate | subject_class, object_class) estimated from training data.

This bias helps capture statistical regularities like:
- "person" + "horse" -> "riding"
- "car" + "street" -> "on"
- "man" + "shirt" -> "wearing"
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any


class FrequencyBias(nn.Module):
    """
    Frequency-based prior for predicate prediction.

    Maintains a lookup table of P(predicate | subject_class, object_class)
    estimated from training data co-occurrence statistics.

    During inference:
        logits_final = logits_visual + alpha * log(P(predicate | subject, object))

    Args:
        num_obj_classes: Number of object classes
        num_pred_classes: Number of predicate classes
        alpha: Weight for frequency bias (0 = no bias, 1 = full bias)
        smoothing: Laplace smoothing factor
    """

    def __init__(
        self,
        num_obj_classes: int = 151,
        num_pred_classes: int = 51,
        alpha: float = 1.0,
        smoothing: float = 1e-3
    ):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_pred_classes = num_pred_classes
        self.alpha = alpha
        self.smoothing = smoothing

        # Frequency matrix: [num_obj, num_obj, num_pred]
        # Stores P(pred | subj_class, obj_class)
        self.register_buffer(
            'freq_matrix',
            torch.zeros(num_obj_classes, num_obj_classes, num_pred_classes)
        )

        # Flag for whether bias has been computed
        self.register_buffer('is_fitted', torch.tensor(False))

    def fit(
        self,
        train_data: List[Dict],
        obj_to_idx: Optional[Dict[str, int]] = None,
        pred_to_idx: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Compute frequency statistics from training data.

        Args:
            train_data: List of training annotations with 'objects' and 'relations'
            obj_to_idx: Optional object name to index mapping
            pred_to_idx: Optional predicate name to index mapping
        """
        # Count triplet occurrences
        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for img_data in train_data:
            objects = img_data['objects']
            relations = img_data['relations']

            for rel in relations:
                subj_idx = objects[rel['subject_idx']]['class_idx']
                obj_idx = objects[rel['object_idx']]['class_idx']
                pred_idx = rel['predicate_idx']

                counts[subj_idx][obj_idx][pred_idx] += 1

        # Convert to probability matrix with smoothing
        freq_matrix = torch.full(
            (self.num_obj_classes, self.num_obj_classes, self.num_pred_classes),
            self.smoothing
        )

        for subj_idx in counts:
            for obj_idx in counts[subj_idx]:
                for pred_idx, count in counts[subj_idx][obj_idx].items():
                    freq_matrix[subj_idx, obj_idx, pred_idx] = count + self.smoothing

        # Normalize to get P(pred | subj, obj)
        # Sum over predicates for each (subj, obj) pair
        totals = freq_matrix.sum(dim=-1, keepdim=True)
        freq_matrix = freq_matrix / totals.clamp(min=1e-8)

        self.freq_matrix.copy_(freq_matrix)
        self.is_fitted.fill_(True)

        print(f"Fitted frequency bias on {sum(len(counts[s]) for s in counts)} unique (subj, obj) pairs")

    def forward(
        self,
        subj_labels: torch.Tensor,
        obj_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Get frequency bias logits for subject-object pairs.

        Args:
            subj_labels: [N] subject class indices
            obj_labels: [N] object class indices

        Returns:
            [N, num_pred_classes] bias logits (log probabilities)
        """
        if not self.is_fitted:
            # Return zeros if not fitted
            return torch.zeros(
                len(subj_labels),
                self.num_pred_classes,
                device=subj_labels.device
            )

        # Lookup probabilities
        probs = self.freq_matrix[subj_labels, obj_labels]  # [N, num_pred]

        # Convert to log probabilities (logits)
        logits = torch.log(probs.clamp(min=1e-10))

        return logits * self.alpha

    def get_top_predicates(
        self,
        subj_label: int,
        obj_label: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Get top-K most likely predicates for a (subject, object) pair.

        Args:
            subj_label: Subject class index
            obj_label: Object class index
            top_k: Number of top predicates to return

        Returns:
            List of (predicate_idx, probability) tuples
        """
        probs = self.freq_matrix[subj_label, obj_label]  # [num_pred]
        top_probs, top_indices = probs.topk(top_k)

        return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]

    def save(self, path: str) -> None:
        """Save frequency bias to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'freq_matrix': self.freq_matrix,
            'num_obj_classes': self.num_obj_classes,
            'num_pred_classes': self.num_pred_classes,
            'alpha': self.alpha,
            'smoothing': self.smoothing,
            'is_fitted': self.is_fitted
        }, path)
        print(f"Saved frequency bias to {path}")

    def load(self, path: str) -> None:
        """Load frequency bias from file."""
        data = torch.load(path, map_location='cpu')
        self.freq_matrix.copy_(data['freq_matrix'])
        self.alpha = data['alpha']
        self.smoothing = data['smoothing']
        self.is_fitted.fill_(data['is_fitted'])
        print(f"Loaded frequency bias from {path}")


def compute_frequency_counts(
    train_data: List[Dict]
) -> Dict[Tuple[int, int], Dict[int, int]]:
    """
    Compute raw frequency counts from training data.

    Args:
        train_data: Training annotations

    Returns:
        {(subj_class, obj_class): {pred_class: count}}
    """
    counts = defaultdict(lambda: defaultdict(int))

    for img_data in train_data:
        objects = img_data['objects']
        relations = img_data['relations']

        for rel in relations:
            subj_idx = objects[rel['subject_idx']]['class_idx']
            obj_idx = objects[rel['object_idx']]['class_idx']
            pred_idx = rel['predicate_idx']

            counts[(subj_idx, obj_idx)][pred_idx] += 1

    return dict(counts)


def load_frequency_bias(
    data_dir: str,
    num_obj_classes: int = 151,
    num_pred_classes: int = 51,
    alpha: float = 1.0,
    train_file: str = 'relationships_train.json'
) -> FrequencyBias:
    """
    Load and fit frequency bias from data directory.

    Args:
        data_dir: Path to processed data directory
        num_obj_classes: Number of object classes
        num_pred_classes: Number of predicate classes
        alpha: Bias weight
        train_file: Name of training file

    Returns:
        Fitted FrequencyBias module
    """
    data_dir = Path(data_dir)

    # Check for precomputed bias
    bias_path = data_dir / 'frequency_bias.pt'
    if bias_path.exists():
        bias = FrequencyBias(num_obj_classes, num_pred_classes, alpha)
        bias.load(bias_path)
        return bias

    # Load training data and compute
    train_path = data_dir / train_file
    if not train_path.exists():
        print(f"Warning: Training file not found at {train_path}")
        print("Returning uninitialized frequency bias")
        return FrequencyBias(num_obj_classes, num_pred_classes, alpha)

    with open(train_path, 'r') as f:
        train_data = json.load(f)

    bias = FrequencyBias(num_obj_classes, num_pred_classes, alpha)
    bias.fit(train_data)

    # Save for future use
    bias.save(bias_path)

    return bias


class LearnableFrequencyBias(nn.Module):
    """
    Learnable frequency bias as an embedding lookup.

    Instead of fixed frequencies, learns a bias embedding for each
    (subject_class, object_class) pair.

    Args:
        num_obj_classes: Number of object classes
        num_pred_classes: Number of predicate classes
        embedding_dim: Internal embedding dimension
    """

    def __init__(
        self,
        num_obj_classes: int = 151,
        num_pred_classes: int = 51,
        embedding_dim: int = 64
    ):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_pred_classes = num_pred_classes
        self.embedding_dim = embedding_dim

        # Embeddings for subject and object classes
        self.subj_embed = nn.Embedding(num_obj_classes, embedding_dim)
        self.obj_embed = nn.Embedding(num_obj_classes, embedding_dim)

        # Classifier to predict bias
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, num_pred_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.subj_embed.weight)
        nn.init.xavier_uniform_(self.obj_embed.weight)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        subj_labels: torch.Tensor,
        obj_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Get learnable bias logits.

        Args:
            subj_labels: [N] subject class indices
            obj_labels: [N] object class indices

        Returns:
            [N, num_pred_classes] bias logits
        """
        subj_emb = self.subj_embed(subj_labels)  # [N, embed_dim]
        obj_emb = self.obj_embed(obj_labels)      # [N, embed_dim]

        # Concatenate and classify
        x = torch.cat([subj_emb, obj_emb], dim=-1)
        logits = self.classifier(x)

        return logits

    def initialize_from_counts(
        self,
        freq_bias: FrequencyBias,
        num_epochs: int = 100,
        lr: float = 0.01
    ) -> None:
        """
        Initialize learnable bias to match frequency counts.

        Args:
            freq_bias: Fitted FrequencyBias module
            num_epochs: Training epochs
            lr: Learning rate
        """
        if not freq_bias.is_fitted:
            print("Warning: FrequencyBias not fitted, skipping initialization")
            return

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Generate all (subj, obj) pairs
        device = next(self.parameters()).device
        subj_indices = torch.arange(self.num_obj_classes, device=device)
        obj_indices = torch.arange(self.num_obj_classes, device=device)

        subj_grid, obj_grid = torch.meshgrid(subj_indices, obj_indices, indexing='ij')
        subj_flat = subj_grid.flatten()
        obj_flat = obj_grid.flatten()

        # Target: log probabilities from frequency bias
        target = freq_bias.freq_matrix.flatten(end_dim=1)  # [N*N, num_pred]
        target = torch.log(target.clamp(min=1e-10)).to(device)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            pred = self.forward(subj_flat, obj_flat)
            loss = nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Bias init epoch {epoch+1}/{num_epochs}, loss: {loss.item():.4f}")

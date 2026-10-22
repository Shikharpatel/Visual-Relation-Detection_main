"""
Explainability Script for VRD
==============================

SHAP-based feature importance analysis for predicate predictions.
Analyzes which feature groups contribute most to predictions:
- visual_subject: Subject visual features
- visual_object: Object visual features
- union_visual: Union box visual features
- geometry: Spatial/geometric features

Usage:
    python -m src.explain --config configs/default.yaml --checkpoint outputs/best_model.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.data.vg_dataset import VGDataset, VGCollator, SyntheticVGDataset
from src.models.scene_graph_model import SceneGraphModel
from src.models.relation_head import RelationHead


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        model_config = config

    dataset_cfg = model_config.get('dataset', {})
    model_cfg = model_config.get('model', {})

    model = SceneGraphModel(
        num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
        num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
        mode=model_config.get('mode', 'predcls'),
        feature_dim=model_cfg.get('feature_dim', 256),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        spatial_dim=model_cfg.get('spatial_dim', 128),
        use_bias=model_cfg.get('use_bias', True),
        bias_alpha=model_cfg.get('bias_alpha', 1.0),
        dropout=model_cfg.get('dropout', 0.3)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, model_config


class RelationClassifierWrapper(nn.Module):
    """
    Wrapper for the relation classifier to be used with SHAP.

    Takes concatenated feature vector and outputs predicate logits.
    """

    def __init__(
        self,
        classifier: nn.Module,
        feature_dim: int = 256,
        spatial_dim: int = 128,
        use_union: bool = True
    ):
        super().__init__()
        self.classifier = classifier
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        self.use_union = use_union

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [N, D] concatenated features
               [subj_visual | obj_visual | union_visual | spatial]

        Returns:
            [N, num_predicates] logits
        """
        return self.classifier(x)

    def get_feature_groups(self) -> Dict[str, Tuple[int, int]]:
        """
        Get feature group indices.

        Returns:
            {group_name: (start_idx, end_idx)}
        """
        groups = {}
        idx = 0

        # Subject visual
        groups['visual_subject'] = (idx, idx + self.feature_dim)
        idx += self.feature_dim

        # Object visual
        groups['visual_object'] = (idx, idx + self.feature_dim)
        idx += self.feature_dim

        # Union visual (if used)
        if self.use_union:
            groups['union_visual'] = (idx, idx + self.feature_dim)
            idx += self.feature_dim

        # Spatial/geometry
        groups['geometry'] = (idx, idx + self.spatial_dim)

        return groups


def compute_feature_importance_gradient(
    model: nn.Module,
    features: torch.Tensor,
    target_class: int,
    feature_groups: Dict[str, Tuple[int, int]]
) -> Dict[str, float]:
    """
    Compute feature importance using gradient-based method.

    Args:
        model: Classifier model
        features: [N, D] input features
        target_class: Target predicate class
        feature_groups: Feature group indices

    Returns:
        {group_name: importance_score}
    """
    features = features.detach().requires_grad_(True)

    # Forward pass
    logits = model(features)  # [N, num_predicates]
    target_logits = logits[:, target_class].sum()

    # Backward pass
    target_logits.backward()
    gradients = features.grad.abs()  # [N, D]

    # Aggregate importance per group
    importance = {}
    for group_name, (start, end) in feature_groups.items():
        group_grad = gradients[:, start:end].mean().item()
        importance[group_name] = group_grad

    # Normalize
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    return importance


def compute_feature_importance_ablation(
    model: nn.Module,
    features: torch.Tensor,
    feature_groups: Dict[str, Tuple[int, int]]
) -> Dict[str, float]:
    """
    Compute feature importance using ablation.

    Zero out each feature group and measure impact on predictions.

    Args:
        model: Classifier model
        features: [N, D] input features
        feature_groups: Feature group indices

    Returns:
        {group_name: importance_score}
    """
    model.eval()

    with torch.no_grad():
        # Baseline predictions
        baseline_logits = model(features)
        baseline_probs = torch.softmax(baseline_logits, dim=-1)

        importance = {}

        for group_name, (start, end) in feature_groups.items():
            # Zero out this group
            ablated_features = features.clone()
            ablated_features[:, start:end] = 0

            # Get ablated predictions
            ablated_logits = model(ablated_features)
            ablated_probs = torch.softmax(ablated_logits, dim=-1)

            # Measure difference (KL divergence or L2)
            diff = (baseline_probs - ablated_probs).pow(2).sum(dim=-1).mean()
            importance[group_name] = diff.item()

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

    return importance


def compute_shap_values(
    model: nn.Module,
    features: torch.Tensor,
    feature_groups: Dict[str, Tuple[int, int]],
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Compute SHAP-style feature importance using permutation.

    This is a simplified version that estimates contribution by
    permuting feature groups.

    Args:
        model: Classifier model
        features: [N, D] input features
        feature_groups: Feature group indices
        num_samples: Number of permutation samples

    Returns:
        {group_name: importance_score}
    """
    try:
        import shap
        has_shap = True
    except ImportError:
        has_shap = False
        print("Warning: SHAP not installed. Using permutation-based approximation.")
        print("Install SHAP with: pip install shap")

    model.eval()

    # If SHAP is available, use it directly
    if has_shap:
        return compute_shap_values_with_shap(model, features, feature_groups)

    # Otherwise use permutation-based approach
    return compute_feature_importance_permutation(
        model, features, feature_groups, num_samples
    )


def compute_feature_importance_permutation(
    model: nn.Module,
    features: torch.Tensor,
    feature_groups: Dict[str, Tuple[int, int]],
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Compute feature importance using permutation.

    Args:
        model: Classifier model
        features: [N, D] input features
        feature_groups: Feature group indices
        num_samples: Number of permutation samples

    Returns:
        {group_name: importance_score}
    """
    model.eval()
    N = features.size(0)

    with torch.no_grad():
        # Baseline predictions
        baseline_logits = model(features)
        baseline_preds = baseline_logits.argmax(dim=-1)

        importance = {}

        for group_name, (start, end) in feature_groups.items():
            accuracy_drops = []

            for _ in range(num_samples):
                # Permute this feature group across samples
                permuted_features = features.clone()
                perm_idx = torch.randperm(N)
                permuted_features[:, start:end] = features[perm_idx, start:end]

                # Get permuted predictions
                permuted_logits = model(permuted_features)
                permuted_preds = permuted_logits.argmax(dim=-1)

                # Measure accuracy drop
                matches = (permuted_preds == baseline_preds).float().mean()
                accuracy_drops.append(1 - matches.item())

            importance[group_name] = np.mean(accuracy_drops)

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

    return importance


def compute_shap_values_with_shap(
    model: nn.Module,
    features: torch.Tensor,
    feature_groups: Dict[str, Tuple[int, int]]
) -> Dict[str, float]:
    """
    Compute SHAP values using the SHAP library.

    Args:
        model: Classifier model
        features: [N, D] input features
        feature_groups: Feature group indices

    Returns:
        {group_name: importance_score}
    """
    import shap

    model.eval()
    features_np = features.detach().cpu().numpy()

    # Create prediction function
    def predict_fn(x):
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(features.device)
            logits = model(x_tensor)
            return logits.cpu().numpy()

    # Use DeepExplainer or KernelExplainer
    try:
        # Try KernelExplainer (model-agnostic)
        background = features_np[:min(50, len(features_np))]
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(features_np[:min(20, len(features_np))])

        if isinstance(shap_values, list):
            # Multi-class: average across classes
            shap_values = np.abs(np.stack(shap_values)).mean(axis=0)
        else:
            shap_values = np.abs(shap_values)

        # Aggregate by feature group
        importance = {}
        for group_name, (start, end) in feature_groups.items():
            importance[group_name] = shap_values[:, start:end].mean()

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    except Exception as e:
        print(f"SHAP computation failed: {e}")
        print("Falling back to permutation-based method")
        return compute_feature_importance_permutation(
            model, features, feature_groups, num_samples=50
        )


def explain_predictions(
    model: SceneGraphModel,
    dataloader,
    device: torch.device,
    num_samples: int = 100,
    method: str = 'ablation'
) -> Dict[str, Any]:
    """
    Run explainability analysis on a batch of predictions.

    Args:
        model: VRD model
        dataloader: Data loader
        device: Torch device
        num_samples: Number of samples to analyze
        method: Explanation method ('gradient', 'ablation', 'shap')

    Returns:
        Explanation results
    """
    model.eval()

    # Get relation head and its classifier
    relation_head = model.relation_head
    classifier = relation_head.classifier

    # Feature dimensions
    feature_dim = model.feature_dim
    spatial_dim = relation_head.spatial_encoder.encoder[-1].out_features if relation_head.spatial_encoder else 128
    use_union = model.use_union_features

    # Create wrapper
    wrapper = RelationClassifierWrapper(
        classifier=classifier,
        feature_dim=feature_dim,
        spatial_dim=spatial_dim,
        use_union=use_union
    )
    feature_groups = wrapper.get_feature_groups()

    print(f"Feature groups: {feature_groups}")

    # Collect features
    all_features = []
    all_predictions = []

    samples_collected = 0

    for batch in dataloader:
        if samples_collected >= num_samples:
            break

        boxes = batch['boxes'].to(device)
        labels = batch['labels'].to(device)
        relation_pairs = batch['relation_pairs'].to(device)
        relation_mask = batch['relation_mask'].to(device)

        batch_size = boxes.size(0)

        # Dummy visual features
        obj_features = torch.zeros(
            batch_size, boxes.size(1), 1024,
            device=device
        )

        # Forward pass with feature extraction
        with torch.no_grad():
            # Get object embeddings
            label_embeds = model.obj_embed(labels)

            for b in range(batch_size):
                if samples_collected >= num_samples:
                    break

                valid_mask = relation_mask[b]
                n_rel = valid_mask.sum().item()

                if n_rel == 0:
                    continue

                valid_pairs = relation_pairs[b, valid_mask]

                subj_idx = valid_pairs[:, 0]
                obj_idx = valid_pairs[:, 1]

                # Get features
                subj_embeds = label_embeds[b, subj_idx]  # [n_rel, D]
                obj_embeds = label_embeds[b, obj_idx]    # [n_rel, D]

                # Union features (simplified)
                union_embeds = (subj_embeds + obj_embeds) / 2

                # Spatial features
                subj_boxes = boxes[b, subj_idx]
                obj_boxes = boxes[b, obj_idx]
                spatial_feats = relation_head.spatial_encoder(
                    subj_boxes, obj_boxes
                )

                # Concatenate
                if use_union:
                    features = torch.cat([
                        subj_embeds, obj_embeds, union_embeds, spatial_feats
                    ], dim=-1)
                else:
                    features = torch.cat([
                        subj_embeds, obj_embeds, spatial_feats
                    ], dim=-1)

                all_features.append(features)
                samples_collected += n_rel

    if not all_features:
        print("No features collected!")
        return {}

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)[:num_samples]
    print(f"Collected {len(all_features)} feature vectors")

    # Compute importance
    print(f"\nComputing feature importance using {method} method...")

    if method == 'gradient':
        # Get predictions first
        with torch.no_grad():
            logits = wrapper(all_features)
            predictions = logits.argmax(dim=-1)

        # Compute per-class importance
        unique_preds = predictions.unique()
        importance_by_class = {}

        for pred_class in unique_preds[:10]:  # Top 10 classes
            mask = predictions == pred_class
            class_features = all_features[mask]

            if len(class_features) > 0:
                importance = compute_feature_importance_gradient(
                    wrapper, class_features, pred_class.item(), feature_groups
                )
                importance_by_class[pred_class.item()] = importance

        # Average importance
        importance = {}
        for group in feature_groups.keys():
            values = [imp[group] for imp in importance_by_class.values() if group in imp]
            importance[group] = np.mean(values) if values else 0.0

    elif method == 'ablation':
        importance = compute_feature_importance_ablation(
            wrapper, all_features, feature_groups
        )

    elif method == 'shap':
        importance = compute_shap_values(
            wrapper, all_features, feature_groups
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        'importance': importance,
        'feature_groups': {k: list(v) for k, v in feature_groups.items()},
        'num_samples': len(all_features),
        'method': method
    }


def print_importance(results: Dict[str, Any]) -> None:
    """Print feature importance results."""
    print("\n" + "="*60)
    print("Feature Group Importance")
    print("="*60)

    importance = results.get('importance', {})

    # Sort by importance
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for group, score in sorted_imp:
        bar = '#' * int(score * 50)
        print(f"  {group:20s}: {score:.4f} |{bar}")

    print("="*60)
    print(f"Method: {results.get('method', 'unknown')}")
    print(f"Samples analyzed: {results.get('num_samples', 0)}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Explain VRD predictions')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--method', type=str, default='ablation',
                        choices=['gradient', 'ablation', 'shap'],
                        help='Explanation method')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of samples to analyze')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--toy', action='store_true',
                        help='Use toy dataset')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    dataset_cfg = config.get('dataset', {})

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, model_config = load_model(args.checkpoint, config, device)

    # Setup data
    if args.synthetic:
        print("Using synthetic dataset")
        dataset = SyntheticVGDataset(
            num_samples=500,
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51)
        )
    else:
        data_dir = dataset_cfg.get('processed_dir', 'data/processed')
        if args.toy:
            data_dir = Path(data_dir) / 'toy'

        dataset = VGDataset(
            data_dir=str(data_dir),
            split='val',
            mode='predcls'
        )

    collator = VGCollator(mode='predcls')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collator
    )

    # Run explanation
    results = explain_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=args.num_samples,
        method=args.method
    )

    # Print results
    print_importance(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

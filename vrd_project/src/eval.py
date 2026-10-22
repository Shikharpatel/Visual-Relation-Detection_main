"""
Evaluation Script for VRD
==========================

Evaluate trained VRD models on validation or test sets.

Usage:
    python -m src.eval --config configs/default.yaml --checkpoint outputs/best_model.pt
    python -m src.eval --config configs/default.yaml --checkpoint outputs/best_model.pt --split test
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logging import setup_logger
from src.utils.metrics import VRDMetrics, format_metrics
from src.data.vg_dataset import VGDataset, VGCollator, SyntheticVGDataset
from src.models.scene_graph_model import SceneGraphModel, MotifNetSceneGraphModel
from src.models.frequency_bias import load_frequency_bias


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
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Torch device

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint if available
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        model_config = config

    dataset_cfg = model_config.get('dataset', {})
    model_cfg = model_config.get('model', {})

    model_type = model_cfg.get('type', 'baseline')
    if model_type == 'motifnet':
        model = MotifNetSceneGraphModel(
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
            mode=model_config.get('mode', 'predcls'),
            feature_dim=model_cfg.get('feature_dim', 256),
            hidden_dim=model_cfg.get('hidden_dim', 512),
            spatial_dim=model_cfg.get('spatial_dim', 128),
            lstm_hidden_dim=model_cfg.get('lstm_hidden_dim', 512),
            lstm_num_layers=model_cfg.get('lstm_num_layers', 2),
            use_object_context=model_cfg.get('use_object_context', True),
            use_edge_context=model_cfg.get('use_edge_context', True),
            use_bias=model_cfg.get('use_bias', True),
            bias_alpha=model_cfg.get('bias_alpha', 1.0),
            dropout=model_cfg.get('dropout', 0.3)
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()
        return model
    else:
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

    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    mode: str = 'predcls',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on dataset.

    Args:
        model: VRD model
        dataloader: Evaluation dataloader
        device: Torch device
        config: Configuration
        mode: Evaluation mode
        verbose: Print progress

    Returns:
        Dictionary of metrics
    """
    model.eval()

    dataset_cfg = config.get('dataset', {})
    num_predicates = dataset_cfg.get('num_pred_classes', 51)

    metrics = VRDMetrics(
        k_values=[20, 50, 100],
        num_predicates=num_predicates,
        mode=mode
    )

    iterator = tqdm(dataloader, desc='Evaluating') if verbose else dataloader

    for batch in iterator:
        boxes = batch['boxes'].to(device)
        labels = batch['labels'].to(device)
        relation_pairs = batch['relation_pairs'].to(device)
        relation_labels = batch['relation_labels'].to(device)
        object_mask = batch['object_mask'].to(device)
        relation_mask = batch['relation_mask'].to(device)

        batch_size = boxes.size(0)

        # Dummy features
        obj_features = torch.zeros(
            batch_size, boxes.size(1), 1024,
            device=device
        )

        # Forward pass
        outputs = model(
            obj_features=obj_features,
            boxes=boxes,
            labels=labels,
            relation_pairs=relation_pairs,
            object_mask=object_mask,
            relation_mask=relation_mask,
            image_sizes=batch['image_sizes']
        )

        pred_scores = outputs['pred_scores']

        # Process each image
        for b in range(batch_size):
            valid_mask = relation_mask[b]
            n_valid = valid_mask.sum().item()

            if n_valid == 0:
                continue

            valid_pairs = relation_pairs[b, valid_mask]
            valid_scores = pred_scores[b, valid_mask]
            valid_gt_labels = relation_labels[b, valid_mask]

            best_scores, best_preds = valid_scores.max(dim=-1)

            # Build predictions
            pred_triplets = []
            pred_scores_list = []
            pred_subj_boxes = []
            pred_obj_boxes = []

            for i in range(n_valid):
                subj_idx = valid_pairs[i, 0].item()
                obj_idx = valid_pairs[i, 1].item()

                pred_triplets.append((
                    labels[b, subj_idx].item(),
                    best_preds[i].item(),
                    labels[b, obj_idx].item()
                ))
                pred_scores_list.append(best_scores[i].item())
                pred_subj_boxes.append(boxes[b, subj_idx].cpu().numpy())
                pred_obj_boxes.append(boxes[b, obj_idx].cpu().numpy())

            # Build GT
            gt_triplets = []
            gt_subj_boxes = []
            gt_obj_boxes = []

            for i in range(n_valid):
                subj_idx = valid_pairs[i, 0].item()
                obj_idx = valid_pairs[i, 1].item()

                gt_triplets.append((
                    labels[b, subj_idx].item(),
                    valid_gt_labels[i].item(),
                    labels[b, obj_idx].item()
                ))
                gt_subj_boxes.append(boxes[b, subj_idx].cpu().numpy())
                gt_obj_boxes.append(boxes[b, obj_idx].cpu().numpy())

            metrics.update(
                pred_triplets=pred_triplets,
                pred_scores=pred_scores_list,
                pred_subj_boxes=pred_subj_boxes,
                pred_obj_boxes=pred_obj_boxes,
                gt_triplets=gt_triplets,
                gt_subj_boxes=gt_subj_boxes,
                gt_obj_boxes=gt_obj_boxes
            )

    return metrics.compute()


def evaluate_with_ablations(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    mode: str = 'predcls'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model with different ablation settings.

    Tests:
    1. Full model (visual + bias)
    2. Visual only (no bias)
    3. Bias only (no visual)

    Args:
        model: VRD model
        dataloader: Evaluation dataloader
        device: Torch device
        config: Configuration
        mode: Evaluation mode

    Returns:
        Dictionary of ablation name -> metrics
    """
    results = {}

    # Full model
    print("\n=== Full Model (Visual + Bias) ===")
    model.use_bias = True
    results['full'] = evaluate(model, dataloader, device, config, mode)
    print(format_metrics(results['full'], mode))

    # Visual only
    print("\n=== Visual Only (No Bias) ===")
    model.use_bias = False
    results['visual_only'] = evaluate(model, dataloader, device, config, mode)
    print(format_metrics(results['visual_only'], mode))

    # Restore
    model.use_bias = True

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate VRD model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Data split to evaluate')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['predcls', 'sgcls', 'sgdet'],
                        help='Evaluation mode (overrides checkpoint)')
    parser.add_argument('--toy', action='store_true',
                        help='Use toy dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--ablations', action='store_true',
                        help='Run ablation studies')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--bias_alpha', type=float, default=None,
                        help='Override bias_alpha (applied to loaded model at eval time)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)

    # Get mode
    mode = args.mode or config.get('mode', 'predcls')
    print(f"Evaluation mode: {mode}")

    # Setup data
    dataset_cfg = config.get('dataset', {})
    training_cfg = config.get('training', {})

    if args.synthetic:
        print("Using synthetic dataset")
        eval_dataset = SyntheticVGDataset(
            num_samples=1000,
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
            seed=42
        )
    else:
        data_dir = dataset_cfg.get('processed_dir', 'data/processed')
        if args.toy:
            data_dir = Path(data_dir) / 'toy'

        eval_dataset = VGDataset(
            data_dir=str(data_dir),
            split=args.split,
            mode=mode
        )

    batch_size = args.batch_size or training_cfg.get('batch_size', 16)
    collator = VGCollator(mode=mode)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True
    )

    print(f"Evaluating on {len(eval_dataset)} samples")

    # Load frequency bias
    if not args.synthetic:
        try:
            data_dir = dataset_cfg.get('processed_dir', 'data/processed')
            if args.toy:
                data_dir = Path(data_dir) / 'toy'

            freq_bias = load_frequency_bias(
                data_dir=str(data_dir),
                num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
                num_pred_classes=dataset_cfg.get('num_pred_classes', 51)
            )
            model.freq_bias = freq_bias.to(device)
            print("Loaded frequency bias")
        except Exception as e:
            print(f"Warning: Could not load frequency bias: {e}")

    # Apply bias_alpha override after freq_bias is loaded (loading resets alpha to saved value)
    if args.bias_alpha is not None:
        config['model']['bias_alpha'] = args.bias_alpha
        model.set_bias_alpha(args.bias_alpha)
        print(f"Overriding bias_alpha to {args.bias_alpha}")

    # Evaluate
    if args.ablations:
        results = evaluate_with_ablations(model, eval_loader, device, config, mode)
    else:
        metrics = evaluate(model, eval_loader, device, config, mode)
        print(format_metrics(metrics, mode))
        results = {'main': metrics}

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    return results


if __name__ == '__main__':
    main()

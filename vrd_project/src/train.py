"""
Training Script for VRD
========================

Main training loop for Visual Relationship Detection.

Usage:
    python -m src.train --config configs/default.yaml --mode predcls
    python -m src.train --config configs/default.yaml --mode predcls --toy
"""

import gc
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logging import setup_logger, CSVLogger, MetricTracker
from src.utils.metrics import VRDMetrics, format_metrics
from src.data.vg_dataset import VGDataset, VGCollator, SyntheticVGDataset
from src.models.scene_graph_model import SceneGraphModel, SimpleVRDModel, MotifNetSceneGraphModel, ZeroShotSceneGraphModel
from src.models.frequency_bias import load_frequency_bias
from src.models.llm_semantic_bias import load_llm_semantic_bias
from src.models.detector import ObjectDetector
from src.utils.losses import compute_class_weights


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    state: Dict[str, Any],
    save_dir: str,
    filename: str = 'checkpoint.pt',
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / filename
    torch.save(state, path)

    if is_best:
        best_path = save_dir / 'best_model.pt'
        torch.save(state, best_path)


def generate_all_pairs(num_objects: int, device: torch.device) -> torch.Tensor:
    """
    Generate all possible subject-object pairs for a set of objects.

    Args:
        num_objects: Number of objects
        device: Torch device

    Returns:
        [N*(N-1), 2] tensor of pairs
    """
    if num_objects <= 1:
        return torch.zeros(0, 2, dtype=torch.long, device=device)

    idx = torch.arange(num_objects, device=device)
    subj, obj = torch.meshgrid(idx, idx, indexing='ij')
    mask = subj != obj
    return torch.stack([subj[mask], obj[mask]], dim=1)


def extract_batch_features(
    batch: Dict[str, Any],
    feature_extractor: nn.Module,
    device: torch.device,
    batch_size: int,
    max_objects: int
) -> torch.Tensor:
    """Extract Faster R-CNN features for a batch of images.

    Handles variable-size images by padding to the max size in the batch.

    Args:
        batch: Collated batch dict (must contain 'images')
        feature_extractor: ObjectDetector instance
        device: Torch device
        batch_size: Number of images in batch
        max_objects: Max padded object dimension

    Returns:
        obj_features: [batch_size, max_objects, 1024]
    """
    images = batch['images']
    object_mask = batch['object_mask']

    # Handle list of variable-size images: pad to max size in batch
    if isinstance(images, list):
        images = [img.to(device) for img in images]
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded = torch.zeros(len(images), 3, max_h, max_w, device=device)
        for i, img in enumerate(images):
            padded[i, :, :img.shape[1], :img.shape[2]] = img
        images_tensor = padded
    else:
        images_tensor = images.to(device)

    # Build list of per-image GT boxes for ROI pooling
    boxes_batch = batch['boxes'].to(device)
    # Compute all object counts in one GPU op, then move to CPU once
    n_objects = object_mask.sum(dim=1).cpu()
    boxes_list = [boxes_batch[i, :int(n_objects[i])] for i in range(batch_size)]

    try:
        with torch.no_grad():
            roi_features = feature_extractor.extract_features(images_tensor, boxes_list)

        # Scatter back to padded tensor
        feat_dim = roi_features.size(-1) if roi_features.numel() > 0 else getattr(feature_extractor, 'feature_dim', 1024)
        obj_features = torch.zeros(batch_size, max_objects, feat_dim, device=device)
        offset = 0
        for i in range(batch_size):
            n = int(n_objects[i])
            if n > 0 and offset + n <= roi_features.size(0):
                obj_features[i, :n] = roi_features[offset:offset + n]
                offset += n

        return obj_features
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return torch.zeros(batch_size, max_objects, 1024, device=device)
        raise


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger,
    config: Dict[str, Any],
    feature_extractor: Optional[nn.Module] = None,
    class_weights: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: VRD model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Torch device
        epoch: Current epoch number
        logger: Logger instance
        config: Configuration
        feature_extractor: Optional ObjectDetector for visual features

    Returns:
        Dictionary of average losses
    """
    model.train()
    tracker = MetricTracker()

    training_cfg = config.get('training', {})
    use_amp = training_cfg.get('use_amp', True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        boxes = batch['boxes'].to(device)
        labels = batch['labels'].to(device)
        relation_pairs = batch['relation_pairs'].to(device)
        relation_labels = batch['relation_labels'].to(device)
        object_mask = batch['object_mask'].to(device)
        relation_mask = batch['relation_mask'].to(device)

        batch_size = boxes.size(0)

        # Extract visual features from images or use zeros as fallback
        if feature_extractor is not None and 'images' in batch:
            obj_features = extract_batch_features(
                batch, feature_extractor, device, batch_size, boxes.size(1)
            )
        else:
            feat_dim = model.clip_feature_dim if hasattr(model, 'clip_feature_dim') else 1024
            obj_features = torch.zeros(
                batch_size, boxes.size(1), feat_dim, device=device
            )

        # Forward pass
        optimizer.zero_grad()

        # Pass relation_labels for VTransE auxiliary loss during training
        forward_kwargs = dict(
            obj_features=obj_features,
            boxes=boxes,
            labels=labels,
            relation_pairs=relation_pairs,
            object_mask=object_mask,
            relation_mask=relation_mask,
            image_sizes=batch['image_sizes']
        )
        # MotifNetSceneGraphModel accepts relation_labels for VTransE
        if hasattr(model, 'use_vtranse') and model.use_vtranse:
            forward_kwargs['relation_labels'] = relation_labels

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(**forward_kwargs)

            # Compute loss
            targets = {
                'relation_labels': relation_labels,
                'labels': labels,
                'object_mask': object_mask
            }

            losses = model.compute_loss(
                outputs, targets, relation_mask,
                use_focal_loss=training_cfg.get('use_focal_loss', False),
                focal_gamma=training_cfg.get('focal_gamma', 2.0),
                focal_alpha=training_cfg.get('focal_alpha', 0.25),
                class_weights=class_weights,
                label_smoothing=training_cfg.get('label_smoothing', 0.0)
            )

        # Backward pass
        loss = losses['total_loss']
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient clipping
        if training_cfg.get('clip_grad', 0) > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                training_cfg['clip_grad']
            )

        scaler.step(optimizer)
        scaler.update()

        # Update tracker
        tracker.update('loss', loss.item(), batch_size)
        tracker.update('pred_loss', losses['pred_loss'].item(), batch_size)
        if 'obj_loss' in losses:
            tracker.update('obj_loss', losses['obj_loss'].item(), batch_size)
        if 'vtranse_loss' in losses:
            tracker.update('vtranse_loss', losses['vtranse_loss'].item(), batch_size)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{tracker.get_average('loss'):.4f}",
            'pred_loss': f"{tracker.get_average('pred_loss'):.4f}"
        })

    return tracker.get_all_averages()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    mode: str = 'predcls',
    feature_extractor: Optional[nn.Module] = None
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.

    Args:
        model: VRD model
        dataloader: Evaluation dataloader
        device: Torch device
        config: Configuration
        mode: Evaluation mode
        feature_extractor: Optional ObjectDetector for visual features

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

    dataset = dataloader.dataset

    for batch in tqdm(dataloader, desc='Evaluating', leave=False):
        boxes = batch['boxes'].to(device)
        labels = batch['labels'].to(device)
        relation_pairs = batch['relation_pairs'].to(device)
        relation_labels = batch['relation_labels'].to(device)
        object_mask = batch['object_mask'].to(device)
        relation_mask = batch['relation_mask'].to(device)

        batch_size = boxes.size(0)

        # Extract visual features or use zeros
        if feature_extractor is not None and 'images' in batch:
            obj_features = extract_batch_features(
                batch, feature_extractor, device, batch_size, boxes.size(1)
            )
        else:
            feat_dim = model.clip_feature_dim if hasattr(model, 'clip_feature_dim') else 1024
            obj_features = torch.zeros(
                batch_size, boxes.size(1), feat_dim, device=device
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

        pred_scores = outputs['pred_scores']  # [B, R, num_pred]

        # Process each image in batch
        for b in range(batch_size):
            # Get valid relations for this image
            valid_mask = relation_mask[b]
            n_valid = valid_mask.sum().item()

            if n_valid == 0:
                continue

            valid_pairs = relation_pairs[b, valid_mask]
            valid_scores = pred_scores[b, valid_mask]
            valid_gt_labels = relation_labels[b, valid_mask]

            # Get best predicate per pair
            best_scores, best_preds = valid_scores.max(dim=-1)

            # Batched extraction — one .cpu() call instead of n_valid × .item()/.cpu()
            subj_idxs = valid_pairs[:, 0].long()
            obj_idxs  = valid_pairs[:, 1].long()

            subj_boxes_np = boxes[b, subj_idxs].cpu().numpy()  # [n_valid, 4]
            obj_boxes_np  = boxes[b, obj_idxs].cpu().numpy()   # [n_valid, 4]
            subj_labels   = labels[b, subj_idxs].cpu().tolist()
            obj_labels    = labels[b, obj_idxs].cpu().tolist()

            # Build prediction list
            pred_triplets    = list(zip(subj_labels, best_preds.cpu().tolist(), obj_labels))
            pred_scores_list = best_scores.cpu().tolist()
            pred_subj_boxes  = list(subj_boxes_np)
            pred_obj_boxes   = list(obj_boxes_np)

            # Build GT list (reuse same indices — no second loop needed)
            gt_triplets  = list(zip(subj_labels, valid_gt_labels.cpu().tolist(), obj_labels))
            gt_subj_boxes = list(subj_boxes_np)
            gt_obj_boxes  = list(obj_boxes_np)

            # Update metrics
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


def main():
    parser = argparse.ArgumentParser(description='Train VRD model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='predcls',
                        choices=['predcls', 'sgcls', 'sgdet'],
                        help='Evaluation mode')
    parser.add_argument('--toy', action='store_true',
                        help='Use toy dataset for quick testing')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (no VG download needed)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['baseline', 'motifnet', 'zeroshot'],
                        help='Model type (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override num_epochs from config')
    # Ablation flags
    parser.add_argument('--disable_llm_bias', action='store_true',
                        help='Ablation: disable LLM semantic bias (CLIP-only baseline)')
    parser.add_argument('--freeze_tau', action='store_true',
                        help='Ablation: freeze temperature tau at initial value (no learning)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line args
    if args.seed is not None:
        config['seed'] = args.seed
    config['mode'] = args.mode
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs

    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = Path(args.output_dir) / f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logger('train', log_dir=str(output_dir))
    csv_logger = CSVLogger(str(output_dir), 'training_log.csv')

    logger.info(f"Config: {config}")
    logger.info(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Setup data
    dataset_cfg = config.get('dataset', {})
    training_cfg = config.get('training', {})

    image_dir = None  # Will be set below if using real VG data with images

    if args.synthetic:
        logger.info("Using synthetic dataset")
        train_dataset = SyntheticVGDataset(
            num_samples=training_cfg.get('synthetic_train_size', 5000),
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
            seed=seed
        )
        val_dataset = SyntheticVGDataset(
            num_samples=training_cfg.get('synthetic_val_size', 1000),
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
            seed=seed + 1
        )
    else:
        data_dir = dataset_cfg.get('processed_dir', 'data/processed')
        if args.toy:
            data_dir = Path(data_dir) / 'toy'
            logger.info(f"Using toy dataset from {data_dir}")

        image_dir = dataset_cfg.get('image_dir', None)
        if image_dir and not Path(image_dir).exists():
            logger.warning(f"image_dir '{image_dir}' not found, using dummy features")
            image_dir = None

        try:
            train_dataset = VGDataset(
                data_dir=str(data_dir),
                split='train',
                mode=args.mode,
                image_dir=image_dir
            )
            val_dataset = VGDataset(
                data_dir=str(data_dir),
                split='val',
                mode=args.mode,
                image_dir=image_dir
            )
        except FileNotFoundError as e:
            logger.error(f"Dataset not found: {e}")
            logger.error("Run preprocessing first or use --synthetic flag")
            sys.exit(1)

    # Create dataloaders
    collator = VGCollator(mode=args.mode)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get('batch_size', 16),
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get('batch_size', 16),
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        collate_fn=collator,
        pin_memory=True
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create model
    model_cfg = config.get('model', {})

    # Determine model type from args or config
    model_type = args.model_type or model_cfg.get('type', 'baseline')
    logger.info(f"Creating model: {model_type}")

    if model_type == 'zeroshot':
        model = ZeroShotSceneGraphModel(
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
            mode=args.mode,
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
    elif model_type == 'motifnet':
        model = MotifNetSceneGraphModel(
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
            mode=args.mode,
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
    else:
        model = SceneGraphModel(
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
            mode=args.mode,
            feature_dim=model_cfg.get('feature_dim', 256),
            hidden_dim=model_cfg.get('hidden_dim', 512),
            spatial_dim=model_cfg.get('spatial_dim', 128),
            use_bias=model_cfg.get('use_bias', True),
            bias_alpha=model_cfg.get('bias_alpha', 1.0),
            dropout=model_cfg.get('dropout', 0.3)
        )

    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize Faster R-CNN feature extractor if images are available
    feature_extractor = None
    use_images = (not args.synthetic) and image_dir is not None
    if use_images:
        if model_type == 'zeroshot':
            logger.info("Initializing CLIP feature extractor (Zero-Shot)")
            from src.models.clip_feature_extractor import create_clip_extractor
            config['device'] = 'cuda' if device.type == 'cuda' else 'cpu'
            feature_extractor = create_clip_extractor(config).to(device)
            # CLIP parameters are frozen inside _load_model() automatically
            logger.info("CLIP feature extractor ready")
        else:
            logger.info("Initializing Faster R-CNN feature extractor (frozen backbone)")
            feature_extractor = ObjectDetector(
                backbone='fasterrcnn_resnet50_fpn',
                num_classes=91,  # Keep COCO classes (pretrained)
                pretrained=True,
                feature_dim=1024,
                trainable_backbone_layers=0
            ).to(device).eval()
            # Freeze all parameters
            for param in feature_extractor.parameters():
                param.requires_grad = False
            logger.info("Faster R-CNN feature extractor ready")

    # Load frequency bias or LLM semantic bias
    if model_cfg.get('use_bias', True) and not args.synthetic:
        bias_type = model_cfg.get('bias_type', 'frequency')
        try:
            if model_type == 'zeroshot' or bias_type == 'llm_semantic':
                llm_bias = load_llm_semantic_bias(
                    data_dir=str(data_dir) if not args.toy else str(Path(data_dir)),
                    num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
                    num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
                    alpha=model_cfg.get('bias_alpha', 1.0),
                    learnable_temperature=model_cfg.get('learnable_temperature', True),
                    priors_file=model_cfg.get('llm_priors_file', 'llm_priors.json')
                )
                model.freq_bias = llm_bias.to(device)
                logger.info("Loaded LLM semantic bias")

                # Ablation: disable LLM bias entirely (CLIP-only baseline)
                if args.disable_llm_bias:
                    model.use_bias = False
                    model.freq_bias = None
                    logger.info("ABLATION: LLM semantic bias DISABLED")

                # Ablation: freeze temperature tau (no learning)
                if args.freeze_tau and model.freq_bias is not None:
                    for param in model.freq_bias.parameters():
                        param.requires_grad = False
                    logger.info("ABLATION: Temperature tau FROZEN (not learnable)")
            else:
                freq_bias = load_frequency_bias(
                    data_dir=str(data_dir) if not args.toy else str(Path(data_dir)),
                    num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
                    num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
                    alpha=model_cfg.get('bias_alpha', 1.0)
                )
                model.freq_bias = freq_bias.to(device)
                logger.info("Loaded frequency bias")
        except Exception as e:
            logger.warning(f"Could not load bias: {e}")

    # Compute class weights for imbalanced predicates
    class_weights = None
    if training_cfg.get('use_class_weights', False) and not args.synthetic:
        logger.info("Computing class weights for imbalanced predicates...")
        num_pred_classes = dataset_cfg.get('num_pred_classes', 51)
        try:
            # Load predicate labels directly from JSON (much faster than iterating dataset)
            rel_train_path = Path(data_dir) / 'relationships_train.json'
            all_pred_labels = []
            if rel_train_path.exists():
                with open(rel_train_path, 'r') as f:
                    train_json = json.load(f)
                for img_data in train_json:
                    for rel in img_data.get('relations', []):
                        all_pred_labels.append(rel['predicate_idx'])
                del train_json  # Free memory
                logger.info(f"Collected {len(all_pred_labels)} predicate labels from JSON")
            else:
                logger.warning(f"relationships_train.json not found at {rel_train_path}")

            if all_pred_labels:
                all_pred_labels_t = torch.tensor(all_pred_labels, dtype=torch.long)
                class_weights = compute_class_weights(
                    all_pred_labels_t,
                    num_pred_classes,
                    method=training_cfg.get('class_weight_method', 'inverse_freq')
                ).to(device)
                # Clamp extreme weights to prevent instability
                class_weights = class_weights.clamp(min=0.1, max=10.0)
                logger.info(f"Class weights computed (clamped to [0.1, 10.0]): "
                            f"min={class_weights.min():.4f}, "
                            f"max={class_weights.max():.4f}, mean={class_weights.mean():.4f}")
            else:
                logger.warning("No predicate labels found, skipping class weights")
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}")

    # Setup optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_cfg.get('lr', 1e-3),
        weight_decay=training_cfg.get('weight_decay', 1e-4)
    )

    # Setup LR scheduler (cosine with warmup or step)
    lr_scheduler_type = training_cfg.get('lr_scheduler', 'step')
    warmup_epochs = training_cfg.get('warmup_epochs', 0)
    num_epochs = training_cfg.get('num_epochs', 30)

    if lr_scheduler_type == 'cosine':
        # Cosine annealing after warmup
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(num_epochs - warmup_epochs, 1),
            eta_min=training_cfg.get('lr', 1e-3) * 0.01  # Min LR = 1% of initial
        )
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = main_scheduler
        logger.info(f"Using cosine LR scheduler (warmup={warmup_epochs} epochs)")
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_cfg.get('lr_decay_step', 10),
            gamma=training_cfg.get('lr_decay_gamma', 0.5)
        )
        logger.info("Using step LR scheduler")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_r50 = 0.0

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_r50 = checkpoint.get('best_r50', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    num_epochs = training_cfg.get('num_epochs', 30)
    eval_every = training_cfg.get('eval_every', 1)

    logger.info(f"Starting training for {num_epochs} epochs")

    for epoch in range(start_epoch, num_epochs):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device, epoch + 1, logger, config,
            feature_extractor=feature_extractor,
            class_weights=class_weights
        )

        # Log training metrics
        log_dict = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            **{f'train_{k}': v for k, v in train_losses.items()}
        }

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {train_losses['loss']:.4f}, "
            f"Pred Loss: {train_losses['pred_loss']:.4f}"
        )

        # Evaluate
        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(model, val_loader, device, config, args.mode,
                                  feature_extractor=feature_extractor)

            log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})

            logger.info(format_metrics(val_metrics, args.mode))

            # Check for best model
            current_r50 = val_metrics.get('R@50', 0.0)
            is_best = current_r50 > best_r50

            if is_best:
                best_r50 = current_r50
                logger.info(f"New best R@50: {best_r50:.4f}")

            # Save checkpoint
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_r50': best_r50,
                    'config': config
                },
                save_dir=str(output_dir),
                filename=f'checkpoint_epoch{epoch + 1}.pt',
                is_best=is_best
            )

        # Log to CSV
        csv_logger.log(log_dict)

        # Step scheduler
        scheduler.step()

    # Final evaluation
    logger.info("Final evaluation on validation set")
    final_metrics = evaluate(model, val_loader, device, config, args.mode,
                             feature_extractor=feature_extractor)
    logger.info(format_metrics(final_metrics, args.mode))

    # Save final model
    save_checkpoint(
        {
            'epoch': num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'best_r50': best_r50,
            'config': config,
            'final_metrics': final_metrics
        },
        save_dir=str(output_dir),
        filename='final_model.pt'
    )

    # Save final metrics
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)

    csv_logger.close()
    logger.info(f"Training complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()

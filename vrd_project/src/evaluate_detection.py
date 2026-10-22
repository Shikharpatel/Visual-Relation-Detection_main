"""
Detection-Based Evaluation for VRD
====================================

Evaluates Phrase Detection and Relationship Detection tasks by:
1. Running Faster R-CNN to detect objects from raw images
2. Predicting relationships between detected objects
3. Computing R@K and mR@K with IoU-based matching

Usage:
    python -m src.evaluate_detection \
        --config configs/motifnet_advanced.yaml \
        --checkpoint outputs/best_model.pt \
        --mode phrase_det \
        --split val

Modes:
    - phrase_det: Match using union box IoU >= 0.5 + label match
    - rel_det:    Match using individual subject/object box IoU >= 0.5 + label match
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.metrics import VRDMetrics, format_metrics
from src.data.vg_dataset import VGDataset, VGCollator
from src.data.class_mapping import load_coco_to_vg_multi_mapping, COCO_CLASSES
from src.models.scene_graph_model import SceneGraphModel, MotifNetSceneGraphModel
from src.models.detector import ObjectDetector
from src.models.frequency_bias import load_frequency_bias
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_all_pairs(num_objects: int, device: torch.device) -> torch.Tensor:
    """Generate all subject-object pairs for N objects."""
    pairs = []
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                pairs.append([i, j])
    if not pairs:
        return torch.zeros(0, 2, dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device)


def detect_and_predict(
    image_path: str,
    detector: nn.Module,
    feature_extractor: ObjectDetector,
    model: nn.Module,
    device: torch.device,
    detection_threshold: float = 0.5,
    max_objects: int = 80,
    coco_to_vg_multi: Optional[Dict[int, List[int]]] = None,
    use_vg_detector: bool = False,
) -> Dict[str, Any]:
    """
    Run object detection + relationship prediction on a single image.

    Supports two modes:
    1. COCO detector + multi-mapping (legacy): Maps COCO labels to VG labels
    2. VG detector (preferred): Detector directly outputs VG class labels

    Args:
        image_path: Path to image file
        detector: Faster R-CNN for detection (COCO or VG trained)
        feature_extractor: ObjectDetector for ROI feature extraction
        model: Trained SceneGraphModel
        device: Torch device
        detection_threshold: Confidence threshold for detections
        max_objects: Maximum detected objects to keep
        coco_to_vg_multi: COCO class index -> list of VG class indices (for COCO mode)
        use_vg_detector: If True, detector outputs VG labels directly

    Returns:
        Dictionary with predicted triplets, scores, and boxes
    """
    empty_result = {'triplets': [], 'scores': [], 'subj_boxes': [], 'obj_boxes': []}

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception:
        return empty_result

    img_w, img_h = image.size
    img_tensor = TF.to_tensor(image).to(device)  # [3, H, W]

    # Run detector
    with torch.no_grad():
        detections = detector([img_tensor])[0]

    det_boxes = detections['boxes']      # [D, 4]
    det_labels = detections['labels']    # [D]
    det_scores = detections['scores']    # [D]

    # Filter by confidence threshold
    keep = det_scores >= detection_threshold
    det_boxes = det_boxes[keep][:max_objects]
    det_labels = det_labels[keep][:max_objects]
    det_scores = det_scores[keep][:max_objects]

    num_detected = len(det_boxes)
    if num_detected < 2:
        return empty_result

    if use_vg_detector:
        # VG detector: labels are VG class indices directly
        vg_labels = det_labels
        vg_label_candidates = [[lbl.item()] for lbl in det_labels]
    else:
        # COCO detector: map to VG labels
        vg_label_candidates = []
        for lbl in det_labels:
            candidates = coco_to_vg_multi.get(lbl.item(), [0]) if coco_to_vg_multi else [0]
            candidates = [c for c in candidates if c != 0]
            if not candidates:
                candidates = [0]
            vg_label_candidates.append(candidates)
        vg_labels = torch.tensor(
            [cands[0] for cands in vg_label_candidates],
            dtype=torch.long, device=device
        )

    # Filter out detections that map to background only
    valid_mask = torch.tensor(
        [cands[0] != 0 for cands in vg_label_candidates],
        dtype=torch.bool, device=device
    )
    if valid_mask.sum() < 2:
        return empty_result

    det_boxes = det_boxes[valid_mask]
    det_scores = det_scores[valid_mask]
    vg_labels = vg_labels[valid_mask] if use_vg_detector else torch.tensor(
        [cands[0] for cands, v in zip(vg_label_candidates, valid_mask) if v],
        dtype=torch.long, device=device
    )
    vg_label_candidates = [c for c, v in zip(vg_label_candidates, valid_mask) if v]
    num_detected = len(det_boxes)

    # Extract ROI features for detected boxes
    with torch.no_grad():
        roi_features = feature_extractor.extract_features(
            img_tensor.unsqueeze(0),
            [det_boxes]
        )  # [num_detected, 1024]

    # Generate all pairs
    relation_pairs = generate_all_pairs(num_detected, device)

    if len(relation_pairs) == 0:
        return empty_result

    # Run scene graph model
    model.eval()
    with torch.no_grad():
        outputs = model(
            obj_features=roi_features,
            boxes=det_boxes,
            labels=vg_labels,
            relation_pairs=relation_pairs,
            image_sizes=[(img_h, img_w)]
        )

    pred_scores_tensor = outputs['pred_scores']  # [R, num_pred] or [1, R, num_pred]
    if pred_scores_tensor.dim() == 3:
        pred_scores_tensor = pred_scores_tensor[0]

    # Get best predicate per pair
    best_pred_scores, best_preds = pred_scores_tensor.max(dim=-1)  # [R]

    # Build prediction list
    triplets = []
    scores = []
    subj_boxes = []
    obj_boxes = []

    for r_idx in range(len(relation_pairs)):
        s_idx = relation_pairs[r_idx, 0].item()
        o_idx = relation_pairs[r_idx, 1].item()

        det_score = (det_scores[s_idx] * det_scores[o_idx]).sqrt().item()
        pred_score = best_pred_scores[r_idx].item()
        pred_label = best_preds[r_idx].item()
        s_box = det_boxes[s_idx].cpu().numpy()
        o_box = det_boxes[o_idx].cpu().numpy()

        # Generate predictions for all VG label combinations
        for s_vg in vg_label_candidates[s_idx]:
            for o_vg in vg_label_candidates[o_idx]:
                if s_vg == 0 or o_vg == 0:
                    continue
                s_penalty = 1.0 if s_vg == vg_label_candidates[s_idx][0] else 0.8
                o_penalty = 1.0 if o_vg == vg_label_candidates[o_idx][0] else 0.8
                combined_score = det_score * pred_score * s_penalty * o_penalty

                triplets.append((s_vg, pred_label, o_vg))
                scores.append(combined_score)
                subj_boxes.append(s_box)
                obj_boxes.append(o_box)

    return {
        'triplets': triplets,
        'scores': scores,
        'subj_boxes': subj_boxes,
        'obj_boxes': obj_boxes
    }


def get_image_path(image_id: int, image_dir: str) -> Optional[str]:
    """Find the image file for a given image ID."""
    image_dir = Path(image_dir)
    # Try different subdirectories and extensions
    for subdir in ['VG_100K', 'VG_100K_2', '']:
        for ext in ['.jpg', '.jpeg', '.png']:
            path = image_dir / subdir / f"{image_id}{ext}"
            if path.exists():
                return str(path)
    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phrase/Relationship Detection')
    parser.add_argument('--config', type=str, default='configs/motifnet_advanced.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['phrase_det', 'rel_det'],
                        help='Evaluation mode: phrase_det or rel_det')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                        help='Faster R-CNN detection confidence threshold')
    parser.add_argument('--max_objects', type=int, default=80,
                        help='Maximum detected objects per image')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of images to evaluate (for testing)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--detector_checkpoint', type=str, default=None,
                        help='Path to VG-trained detector checkpoint (uses VG labels directly)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    dataset_cfg = config.get('dataset', {})
    model_cfg = config.get('model', {})

    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    data_dir = dataset_cfg.get('processed_dir', 'data/processed')
    use_vg_detector = args.detector_checkpoint is not None
    coco_to_vg_multi = None

    if use_vg_detector:
        # Load VG-trained detector — outputs VG class labels directly
        print(f"Loading VG-trained detector from {args.detector_checkpoint}...")
        from src.train_detector import create_vg_detector
        det_ckpt = torch.load(args.detector_checkpoint, map_location=device)
        num_det_classes = det_ckpt.get('num_classes', 105)
        detector = create_vg_detector(num_classes=num_det_classes, pretrained_coco=False)
        detector.load_state_dict(det_ckpt['model_state_dict'])
        detector = detector.to(device).eval()
        for param in detector.parameters():
            param.requires_grad = False
        print(f"  VG detector ready ({num_det_classes} classes, frozen)")

        # Feature extractor uses the same VG detector backbone
        feature_extractor = ObjectDetector(
            backbone='fasterrcnn_resnet50_fpn',
            num_classes=num_det_classes,
            pretrained=False,
            feature_dim=1024,
            trainable_backbone_layers=0
        ).to(device)
        # Load detector weights into feature extractor
        fe_state = feature_extractor.state_dict()
        det_state = det_ckpt['model_state_dict']
        # Map detector keys to ObjectDetector keys (detector -> detector.*)
        mapped_state = {}
        for k, v in det_state.items():
            fe_key = f'detector.{k}'
            if fe_key in fe_state and fe_state[fe_key].shape == v.shape:
                mapped_state[fe_key] = v
        fe_state.update(mapped_state)
        feature_extractor.load_state_dict(fe_state)
        feature_extractor = feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False
        print(f"  Feature extractor loaded from VG detector ({len(mapped_state)} params mapped)")
    else:
        # COCO detector mode with multi-mapping
        print("Loading COCO-to-VG class mapping (multi-label)...")
        coco_to_vg_multi = load_coco_to_vg_multi_mapping(data_dir)
        mapped_count = sum(1 for v in coco_to_vg_multi.values() if v != [0])
        total_candidates = sum(len(v) for v in coco_to_vg_multi.values() if v != [0])
        print(f"  Mapped {mapped_count}/{len(COCO_CLASSES)} COCO classes -> {total_candidates} VG candidates")

        print("Loading pretrained Faster R-CNN for detection...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        detector = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
        for param in detector.parameters():
            param.requires_grad = False
        print("  COCO detector ready (frozen, original pretrained weights)")

        print("Loading Faster R-CNN feature extractor...")
        feature_extractor = ObjectDetector(
            backbone='fasterrcnn_resnet50_fpn',
            num_classes=91,
            pretrained=True,
            feature_dim=1024,
            trainable_backbone_layers=0
        ).to(device).eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False
        print("  Feature extractor ready (frozen)")

    # Load trained scene graph model from checkpoint
    print(f"Loading scene graph model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    ckpt_config = checkpoint.get('config', config)
    ckpt_dataset_cfg = ckpt_config.get('dataset', {})
    ckpt_model_cfg = ckpt_config.get('model', {})

    model_type = ckpt_model_cfg.get('type', 'baseline')

    if model_type == 'motifnet':
        model = MotifNetSceneGraphModel(
            num_obj_classes=ckpt_dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=ckpt_dataset_cfg.get('num_pred_classes', 51),
            mode='predcls',  # We handle detection externally
            feature_dim=ckpt_model_cfg.get('feature_dim', 256),
            hidden_dim=ckpt_model_cfg.get('hidden_dim', 512),
            spatial_dim=ckpt_model_cfg.get('spatial_dim', 128),
            lstm_hidden_dim=ckpt_model_cfg.get('lstm_hidden_dim', 256),
            lstm_num_layers=ckpt_model_cfg.get('lstm_num_layers', 2),
            use_object_context=ckpt_model_cfg.get('use_object_context', True),
            use_edge_context=ckpt_model_cfg.get('use_edge_context', True),
            use_bias=ckpt_model_cfg.get('use_bias', True),
            bias_alpha=ckpt_model_cfg.get('bias_alpha', 1.0),
            dropout=ckpt_model_cfg.get('dropout', 0.3),
            use_union_features=ckpt_model_cfg.get('use_union_features', True),
            use_vtranse=ckpt_model_cfg.get('use_vtranse', False),
            vtranse_embed_dim=ckpt_model_cfg.get('vtranse_embed_dim', 256),
            vtranse_weight=ckpt_model_cfg.get('vtranse_weight', 0.3)
        )
    else:
        model = SceneGraphModel(
            num_obj_classes=ckpt_dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=ckpt_dataset_cfg.get('num_pred_classes', 51),
            mode='predcls',
            feature_dim=ckpt_model_cfg.get('feature_dim', 256),
            hidden_dim=ckpt_model_cfg.get('hidden_dim', 512),
            spatial_dim=ckpt_model_cfg.get('spatial_dim', 128),
            use_bias=ckpt_model_cfg.get('use_bias', True),
            bias_alpha=ckpt_model_cfg.get('bias_alpha', 1.0),
            dropout=ckpt_model_cfg.get('dropout', 0.3)
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    print(f"  Model loaded ({model_type}, {sum(p.numel() for p in model.parameters()):,} params)")

    # Load frequency bias if available
    try:
        freq_bias = load_frequency_bias(
            data_dir=data_dir,
            num_obj_classes=ckpt_dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=ckpt_dataset_cfg.get('num_pred_classes', 51),
            alpha=ckpt_model_cfg.get('bias_alpha', 1.0)
        )
        model.freq_bias = freq_bias.to(device)
        print("  Loaded frequency bias")
    except Exception as e:
        print(f"  Running without frequency bias: {e}")

    # Load dataset to get GT annotations and image IDs
    print(f"\nLoading {args.split} dataset annotations...")
    image_dir = dataset_cfg.get('image_dir', 'data/images')

    dataset = VGDataset(
        data_dir=data_dir,
        split=args.split,
        mode='predcls',  # We need GT labels for evaluation
        image_dir=None   # We load images manually
    )
    print(f"  {len(dataset)} images in {args.split} split")

    # Initialize metrics
    num_predicates = ckpt_dataset_cfg.get('num_pred_classes', 51)
    metrics = VRDMetrics(
        k_values=[20, 50, 100],
        num_predicates=num_predicates,
        mode=args.mode,
        iou_threshold=0.5
    )

    # Evaluate image by image
    num_images = min(len(dataset), args.limit) if args.limit else len(dataset)
    skipped = 0
    processed = 0

    print(f"\nEvaluating {args.mode} on {num_images} images...")
    print(f"  Detection threshold: {args.detection_threshold}")
    print(f"  Max objects per image: {args.max_objects}")
    print()

    for idx in tqdm(range(num_images), desc=f'Evaluating {args.mode}'):
        sample = dataset[idx]

        # Get image path
        image_id = sample.get('image_id', idx)
        img_path = get_image_path(image_id, image_dir)

        if img_path is None:
            skipped += 1
            continue

        # Run detection + relationship prediction
        predictions = detect_and_predict(
            image_path=img_path,
            detector=detector,
            feature_extractor=feature_extractor,
            model=model,
            device=device,
            detection_threshold=args.detection_threshold,
            max_objects=args.max_objects,
            coco_to_vg_multi=coco_to_vg_multi,
            use_vg_detector=use_vg_detector
        )

        # Build GT from dataset sample
        gt_triplets = []
        gt_subj_boxes = []
        gt_obj_boxes = []

        boxes = sample['boxes']       # [N, 4]
        labels = sample['labels']     # [N]
        rel_pairs = sample['relation_pairs']     # [R, 2]
        rel_labels = sample['relation_labels']   # [R]

        for r_idx in range(len(rel_pairs)):
            s_idx = rel_pairs[r_idx, 0].item()
            o_idx = rel_pairs[r_idx, 1].item()

            gt_triplets.append((
                labels[s_idx].item(),
                rel_labels[r_idx].item(),
                labels[o_idx].item()
            ))
            gt_subj_boxes.append(boxes[s_idx].numpy())
            gt_obj_boxes.append(boxes[o_idx].numpy())

        if len(gt_triplets) == 0:
            continue

        # Update metrics
        metrics.update(
            pred_triplets=predictions['triplets'],
            pred_scores=predictions['scores'],
            pred_subj_boxes=predictions['subj_boxes'],
            pred_obj_boxes=predictions['obj_boxes'],
            gt_triplets=gt_triplets,
            gt_subj_boxes=gt_subj_boxes,
            gt_obj_boxes=gt_obj_boxes
        )
        processed += 1

    # Compute and display results
    results = metrics.compute()

    print(f"\nProcessed: {processed} images")
    print(f"Skipped (no image): {skipped}")
    print(format_metrics(results, args.mode))

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'mode': args.mode,
                'split': args.split,
                'detection_threshold': args.detection_threshold,
                'max_objects': args.max_objects,
                'processed_images': processed,
                'skipped_images': skipped,
                'metrics': {k: float(v) for k, v in results.items()}
            }, f, indent=2)
        print(f"Results saved to {args.output}")

    return results


if __name__ == '__main__':
    main()

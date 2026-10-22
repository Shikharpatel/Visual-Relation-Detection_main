"""
Inference Script for VRD
=========================

Run inference on images and generate scene graphs.

Usage:
    python -m src.infer --config configs/default.yaml --checkpoint outputs/best_model.pt --image path/to/image.jpg
    python -m src.infer --checkpoint outputs/best_model.pt --image demo.jpg --output scene_graph.json
    python -m src.infer --checkpoint outputs/best_model.pt --image photo.jpg --detect  # Auto-detect objects!
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
from PIL import Image
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.viz import (
    visualize_predictions, draw_scene_graph,
    save_scene_graph_json, print_relation_summary
)
from src.utils.box_ops import compute_spatial_features
from src.models.scene_graph_model import SceneGraphModel, ZeroShotSceneGraphModel
from src.models.clip_feature_extractor import CLIPFeatureExtractor
from src.models.frequency_bias import load_frequency_bias


# COCO class names (used by pretrained Faster R-CNN)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_vocabularies(data_dir: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Load object and predicate vocabularies.

    Args:
        data_dir: Path to processed data directory

    Returns:
        (idx_to_obj, idx_to_pred) mappings
    """
    data_dir = Path(data_dir)

    with open(data_dir / 'idx_to_obj.json', 'r') as f:
        idx_to_obj = json.load(f)
    idx_to_obj = {int(k): v for k, v in idx_to_obj.items()}

    with open(data_dir / 'idx_to_pred.json', 'r') as f:
        idx_to_pred = json.load(f)
    idx_to_pred = {int(k): v for k, v in idx_to_pred.items()}

    return idx_to_obj, idx_to_pred


def load_model(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """Load trained model from checkpoint (supports baseline + zero-shot)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        model_config = config

    dataset_cfg = model_config.get('dataset', {})
    model_cfg = model_config.get('model', {})
    model_type = model_cfg.get('type', 'baseline')

    if model_type == 'zeroshot':
        print(f"Loading ZeroShotSceneGraphModel (CLIP 512-dim)")
        model = ZeroShotSceneGraphModel(
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51),
            mode=model_config.get('mode', 'predcls'),
            clip_feature_dim=model_cfg.get('clip_feature_dim', 512),
            feature_dim=model_cfg.get('feature_dim', 256),
            hidden_dim=model_cfg.get('hidden_dim', 512),
            spatial_dim=model_cfg.get('spatial_dim', 128),
            use_bias=model_cfg.get('use_bias', True),
            bias_alpha=model_cfg.get('bias_alpha', 1.0),
            dropout=model_cfg.get('dropout', 0.3),
            learnable_temperature=model_cfg.get('learnable_temperature', True),
            initial_temperature=model_cfg.get('initial_temperature', 1.0),
        )
    else:
        print(f"Loading baseline SceneGraphModel (1024-dim)")
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

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    return model, model_config


def generate_demo_data(
    num_objects: int = 5,
    image_size: Tuple[int, int] = (600, 800),
    num_obj_classes: int = 151,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate demo data for testing.

    Args:
        num_objects: Number of objects
        image_size: (height, width)
        num_obj_classes: Number of object classes
        seed: Random seed

    Returns:
        (boxes, labels, relation_pairs)
    """
    np.random.seed(seed)
    h, w = image_size

    # Generate random boxes
    boxes = []
    for _ in range(num_objects):
        x1 = np.random.randint(0, w - 100)
        y1 = np.random.randint(0, h - 100)
        x2 = x1 + np.random.randint(50, 150)
        y2 = y1 + np.random.randint(50, 150)
        boxes.append([x1, y1, min(x2, w), min(y2, h)])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.randint(1, num_obj_classes, (num_objects,))

    # Generate all pairs
    pairs = []
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                pairs.append([i, j])

    relation_pairs = torch.tensor(pairs, dtype=torch.long)

    return boxes, labels, relation_pairs


def detect_objects_auto(
    image: np.ndarray,
    device: torch.device,
    obj_to_idx: Optional[Dict[str, int]] = None,
    score_threshold: float = 0.5,
    max_objects: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Automatically detect objects in an image using pretrained Faster R-CNN.

    Args:
        image: RGB image as numpy array [H, W, 3]
        device: Torch device
        obj_to_idx: Optional mapping from object names to VG class indices
        score_threshold: Minimum detection confidence
        max_objects: Maximum number of objects to detect

    Returns:
        (boxes, labels, relation_pairs, label_names)
    """
    print("Loading pretrained Faster R-CNN detector...")

    # Load pretrained model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn(weights=weights)
    detector = detector.to(device)
    detector.eval()

    # Preprocess image
    transform = weights.transforms()
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    img_tensor = transform(img_tensor).unsqueeze(0).to(device)

    # Run detection
    print("Detecting objects...")
    with torch.no_grad():
        detections = detector(img_tensor)[0]

    # Filter by score
    keep = detections['scores'] >= score_threshold
    boxes = detections['boxes'][keep][:max_objects]
    coco_labels = detections['labels'][keep][:max_objects]
    scores = detections['scores'][keep][:max_objects]

    num_detected = len(boxes)
    print(f"Detected {num_detected} objects above threshold {score_threshold}")

    if num_detected == 0:
        print("Warning: No objects detected! Try lowering --detection_threshold")
        return (
            torch.zeros(0, 4),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.long),
            []
        )

    # Get COCO class names
    label_names = [COCO_CLASSES[lbl.item()] for lbl in coco_labels]

    # Map to VG class indices if mapping provided
    if obj_to_idx is not None:
        vg_labels = []
        for name in label_names:
            # Try exact match first
            if name in obj_to_idx:
                vg_labels.append(obj_to_idx[name])
            # Try lowercase
            elif name.lower() in obj_to_idx:
                vg_labels.append(obj_to_idx[name.lower()])
            else:
                # Use a generic object class (usually 0 or 1)
                vg_labels.append(1)
        labels = torch.tensor(vg_labels, dtype=torch.long)
    else:
        # Use COCO labels directly
        labels = coco_labels.cpu()

    # Print detected objects
    print("\nDetected objects:")
    for i, (name, score) in enumerate(zip(label_names, scores)):
        print(f"  {i+1}. {name} (confidence: {score:.3f})")

    # Generate all pairs for relationship prediction
    pairs = []
    for i in range(num_detected):
        for j in range(num_detected):
            if i != j:
                pairs.append([i, j])

    relation_pairs = torch.tensor(pairs, dtype=torch.long) if pairs else torch.zeros(0, 2, dtype=torch.long)

    return boxes.cpu(), labels, relation_pairs, label_names


def load_gt_annotations(
    annotation_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load ground truth annotations from JSON file.

    Expected format:
    {
        "objects": [{"box": [x1,y1,x2,y2], "label": int}, ...],
        "relations": [{"subject_idx": int, "object_idx": int, "predicate": int}, ...]
    }

    Args:
        annotation_path: Path to annotation JSON

    Returns:
        (boxes, labels, relation_pairs, relation_labels)
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    boxes = torch.tensor([obj['box'] for obj in data['objects']], dtype=torch.float32)
    labels = torch.tensor([obj['label'] for obj in data['objects']], dtype=torch.long)

    if data.get('relations'):
        relation_pairs = torch.tensor(
            [[r['subject_idx'], r['object_idx']] for r in data['relations']],
            dtype=torch.long
        )
        relation_labels = torch.tensor(
            [r['predicate'] for r in data['relations']],
            dtype=torch.long
        )
    else:
        # Generate all pairs if no relations specified
        pairs = []
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i != j:
                    pairs.append([i, j])
        relation_pairs = torch.tensor(pairs, dtype=torch.long) if pairs else torch.zeros(0, 2, dtype=torch.long)
        relation_labels = torch.zeros(len(relation_pairs), dtype=torch.long)

    return boxes, labels, relation_pairs, relation_labels


@torch.no_grad()
def infer(
    model: nn.Module,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    relation_pairs: torch.Tensor,
    device: torch.device,
    image_size: Optional[Tuple[int, int]] = None,
    top_k: int = 20,
    image: Optional[np.ndarray] = None,
    clip_extractor: Optional[CLIPFeatureExtractor] = None
) -> Dict[str, Any]:
    """
    Run inference on a single image.

    Args:
        model: VRD model
        boxes: [N, 4] object boxes
        labels: [N] object labels
        relation_pairs: [R, 2] relation pairs
        device: Torch device
        image_size: (height, width)
        top_k: Number of top predictions to return
        image: Optional RGB image as numpy array [H, W, 3] for CLIP feature extraction
        clip_extractor: Optional CLIPFeatureExtractor instance

    Returns:
        Prediction results
    """
    model.eval()

    # Move to device
    boxes = boxes.to(device)
    labels = labels.to(device)
    relation_pairs = relation_pairs.to(device)

    # Extract real CLIP features if image + extractor are provided
    is_zeroshot = isinstance(model, ZeroShotSceneGraphModel)
    if is_zeroshot and image is not None and clip_extractor is not None:
        print("Extracting CLIP features from detected object crops...")
        # Convert image to tensor [1, C, H, W] normalized to 0-1
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, C, H, W]

        # Extract CLIP features for each box
        obj_features = clip_extractor.extract_features(
            images=img_tensor,
            boxes=[boxes],  # List of [N, 4] tensors
        )  # [N, 512]
        print(f"Extracted {obj_features.shape[0]} CLIP features (dim={obj_features.shape[1]})")
    else:
        # Fallback to dummy features
        feat_dim = 512 if is_zeroshot else 1024
        obj_features = torch.zeros(len(boxes), feat_dim, device=device)

    # Forward pass
    outputs = model(
        obj_features=obj_features,
        boxes=boxes,
        labels=labels,
        relation_pairs=relation_pairs,
        image_sizes=[image_size] if image_size else None
    )

    pred_scores = outputs['pred_scores']  # [R, num_pred]

    # Get top predictions
    best_scores, best_preds = pred_scores.max(dim=-1)  # [R]

    # Sort by score
    sorted_indices = best_scores.argsort(descending=True)[:top_k]

    predictions = []
    for idx in sorted_indices:
        idx = idx.item()
        subj_idx = relation_pairs[idx, 0].item()
        obj_idx = relation_pairs[idx, 1].item()

        predictions.append({
            'subject_idx': subj_idx,
            'object_idx': obj_idx,
            'subject_label': labels[subj_idx].item(),
            'object_label': labels[obj_idx].item(),
            'predicate': best_preds[idx].item(),
            'score': best_scores[idx].item()
        })

    return {
        'predictions': predictions,
        'boxes': boxes.cpu().numpy(),
        'labels': labels.cpu().numpy()
    }


def build_scene_graph(
    predictions: List[Dict],
    boxes: np.ndarray,
    labels: np.ndarray,
    idx_to_obj: Dict[int, str],
    idx_to_pred: Dict[int, str],
    score_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Build scene graph JSON from predictions.

    Args:
        predictions: List of prediction dicts
        boxes: Object boxes
        labels: Object labels
        idx_to_obj: Object index to name mapping
        idx_to_pred: Predicate index to name mapping
        score_threshold: Minimum score threshold

    Returns:
        Scene graph dictionary
    """
    # Build objects list
    objects = []
    for i, (box, label) in enumerate(zip(boxes, labels)):
        objects.append({
            'id': i,
            'label': idx_to_obj.get(int(label), f'object_{label}'),
            'box': box.tolist()
        })

    # Build relations list
    relations = []
    for pred in predictions:
        if pred['score'] >= score_threshold:
            relations.append({
                'subj_id': pred['subject_idx'],
                'obj_id': pred['object_idx'],
                'predicate': idx_to_pred.get(pred['predicate'], f'pred_{pred["predicate"]}'),
                'score': round(pred['score'], 4)
            })

    return {
        'objects': objects,
        'relations': relations
    }


def main():
    parser = argparse.ArgumentParser(description='Run VRD inference')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to annotation JSON (boxes, labels)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for scene graph JSON')
    parser.add_argument('--viz_output', type=str, default=None,
                        help='Output path for visualization')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top predictions')
    parser.add_argument('--score_threshold', type=float, default=0.1,
                        help='Score threshold for output')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with synthetic data')
    parser.add_argument('--detect', action='store_true',
                        help='Auto-detect objects using pretrained Faster R-CNN')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                        help='Detection confidence threshold (for --detect mode)')
    parser.add_argument('--max_objects', type=int, default=20,
                        help='Maximum objects to detect (for --detect mode)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--no_viz', action='store_true',
                        help='Skip visualization')

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

    # Load vocabularies
    data_dir = dataset_cfg.get('processed_dir', 'data/processed')
    try:
        idx_to_obj, idx_to_pred = load_vocabularies(data_dir)
        print(f"Loaded vocabularies: {len(idx_to_obj)} objects, {len(idx_to_pred)} predicates")
    except FileNotFoundError:
        print("Warning: Could not load vocabularies, using indices")
        idx_to_obj = {i: f'object_{i}' for i in range(200)}
        idx_to_pred = {i: f'predicate_{i}' for i in range(100)}

    # Load frequency bias
    try:
        freq_bias = load_frequency_bias(
            data_dir=data_dir,
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151),
            num_pred_classes=dataset_cfg.get('num_pred_classes', 51)
        )
        model.freq_bias = freq_bias.to(device)
        print("Loaded frequency bias")
    except Exception as e:
        print(f"Note: Running without frequency bias: {e}")

    # Get input data
    image = None
    image_size = None
    detected_label_names = None  # For auto-detection mode

    if args.demo:
        print("Running demo with synthetic data")
        boxes, labels, relation_pairs = generate_demo_data(
            num_objects=6,
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151)
        )
        image_size = (600, 800)

    elif args.detect and args.image:
        # Automatic object detection mode
        print(f"Loading image from {args.image}")
        image = np.array(Image.open(args.image).convert('RGB'))
        image_size = (image.shape[0], image.shape[1])

        # Build obj_to_idx mapping if we have the vocabulary
        obj_to_idx = None
        try:
            obj_classes_path = Path(data_dir) / 'obj_classes.json'
            if obj_classes_path.exists():
                with open(obj_classes_path, 'r') as f:
                    obj_classes = json.load(f)
                obj_to_idx = {name: idx for idx, name in enumerate(obj_classes)}
                print(f"Loaded {len(obj_to_idx)} object classes for mapping")
        except Exception as e:
            print(f"Note: Could not load object class mapping: {e}")

        # Run automatic detection
        boxes, labels, relation_pairs, detected_label_names = detect_objects_auto(
            image=image,
            device=device,
            obj_to_idx=obj_to_idx,
            score_threshold=args.detection_threshold,
            max_objects=args.max_objects
        )

        if len(boxes) == 0:
            print("No objects detected. Cannot predict relationships.")
            return {}

    elif args.annotations:
        print(f"Loading annotations from {args.annotations}")
        boxes, labels, relation_pairs, _ = load_gt_annotations(args.annotations)

        if args.image:
            image = np.array(Image.open(args.image).convert('RGB'))
            image_size = (image.shape[0], image.shape[1])

    elif args.image:
        print(f"Loading image from {args.image}")
        image = np.array(Image.open(args.image).convert('RGB'))
        image_size = (image.shape[0], image.shape[1])

        # For demo without annotations, generate random boxes
        print("No annotations provided, using demo boxes")
        print("Tip: Use --detect flag to auto-detect objects!")
        boxes, labels, relation_pairs = generate_demo_data(
            num_objects=5,
            image_size=image_size,
            num_obj_classes=dataset_cfg.get('num_obj_classes', 151)
        )
    else:
        parser.error("Provide --image, --annotations, or --demo")

    print(f"Input: {len(boxes)} objects, {len(relation_pairs)} potential relations")

    # Initialize CLIP extractor for zero-shot models
    clip_extractor = None
    model_cfg = model_config.get('model', {})
    if model_cfg.get('type', 'baseline') == 'zeroshot' and image is not None:
        print("Initializing CLIP feature extractor for inference...")
        clip_extractor = CLIPFeatureExtractor(
            clip_model_name=model_cfg.get('clip_model', 'ViT-B-32'),
            pretrained=model_cfg.get('clip_pretrained', 'openai'),
            device=str(device)
        )

    # Run inference
    results = infer(
        model=model,
        boxes=boxes,
        labels=labels,
        relation_pairs=relation_pairs,
        device=device,
        image_size=image_size,
        top_k=args.top_k,
        image=image,
        clip_extractor=clip_extractor
    )

    predictions = results['predictions']
    print(f"\nFound {len(predictions)} predictions")

    # Print top relations
    print("\n" + "="*60)
    print("Top Predicted Relations")
    print("="*60)

    for i, pred in enumerate(predictions[:10]):
        # Use detected names if available (auto-detect mode), otherwise use vocabulary
        if detected_label_names is not None:
            subj_name = detected_label_names[pred['subject_idx']]
            obj_name = detected_label_names[pred['object_idx']]
        else:
            subj_name = idx_to_obj.get(pred['subject_label'], f"obj_{pred['subject_label']}")
            obj_name = idx_to_obj.get(pred['object_label'], f"obj_{pred['object_label']}")
        pred_name = idx_to_pred.get(pred['predicate'], f"pred_{pred['predicate']}")

        print(f"  {i+1:2d}. {subj_name} --[{pred_name}]--> {obj_name}  (score: {pred['score']:.4f})")

    print("="*60)

    # Build scene graph
    # Use detected names if available (for auto-detect mode)
    effective_idx_to_obj = idx_to_obj
    if detected_label_names is not None:
        # Create a temporary mapping using detected names
        effective_idx_to_obj = {i: name for i, name in enumerate(detected_label_names)}
        # Also update the original labels array with sequential indices for scene graph
        results['labels'] = np.arange(len(detected_label_names))

    scene_graph = build_scene_graph(
        predictions=predictions,
        boxes=results['boxes'],
        labels=results['labels'],
        idx_to_obj=effective_idx_to_obj,
        idx_to_pred=idx_to_pred,
        score_threshold=args.score_threshold
    )

    # Save scene graph JSON
    if args.output:
        save_scene_graph_json(
            objects=scene_graph['objects'],
            relations=scene_graph['relations'],
            save_path=args.output
        )

    # Print scene graph summary
    print(f"\nScene Graph: {len(scene_graph['objects'])} objects, {len(scene_graph['relations'])} relations")

    # Visualization
    if not args.no_viz and image is not None:
        print("\nGenerating visualization...")

        # Prepare data for visualization
        obj_labels = [obj['label'] for obj in scene_graph['objects']]
        vis_relations = [
            (rel['subj_id'], rel['obj_id'], rel['predicate'], rel['score'])
            for rel in scene_graph['relations']
        ]

        viz_path = args.viz_output or 'vrd_output.png'
        visualize_predictions(
            image=image,
            boxes=results['boxes'],
            labels=obj_labels,
            relations=vis_relations,
            save_path=viz_path,
            show=False
        )

    # Also draw scene graph visualization
    if not args.no_viz:
        graph_path = args.viz_output.replace('.png', '_graph.png') if args.viz_output else 'scene_graph.png'
        draw_scene_graph(
            objects=scene_graph['objects'],
            relations=scene_graph['relations'],
            save_path=graph_path,
            show=False
        )

    print("\nInference complete!")

    return scene_graph


if __name__ == '__main__':
    main()

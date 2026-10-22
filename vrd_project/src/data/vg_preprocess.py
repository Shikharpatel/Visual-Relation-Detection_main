"""
Visual Genome Dataset Preprocessing
====================================

Preprocesses Visual Genome annotations into a VG150 style split:
- ~150 most frequent object categories
- ~50 most frequent predicates
- Standard train/val/test splits

This script builds:
1. Object vocabulary mapping (obj_classes.json)
2. Predicate vocabulary mapping (pred_classes.json)
3. Image metadata (image_data.json)
4. Relationship annotations (relationships_train/val/test.json)
5. Statistics file (statistics.json)

Usage:
    python -m src.data.vg_preprocess --config configs/default.yaml
    python -m src.data.vg_preprocess --raw_dir data/raw --out_dir data/processed

Expected raw files in data/raw/:
    - objects.json (VG object annotations)
    - relationships.json (VG relationship annotations)
    - image_data.json (VG image metadata)

Download from: https://visualgenome.org/api/v0/api_home.html
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from tqdm import tqdm


def load_json(path: str) -> dict:
    """Load JSON file."""
    print(f"Loading {path}...")
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    """Save JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")


def clean_name(name: str) -> str:
    """Clean and normalize object/predicate names."""
    # Lowercase
    name = name.lower().strip()
    # Remove special characters
    name = ''.join(c if c.isalnum() or c == ' ' else ' ' for c in name)
    # Collapse whitespace
    name = ' '.join(name.split())
    return name


def build_vocabularies(
    objects_data: List[Dict],
    relationships_data: List[Dict],
    num_obj_classes: int = 150,
    num_pred_classes: int = 50,
    min_obj_count: int = 100,
    min_pred_count: int = 100
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build object and predicate vocabularies from most frequent classes.

    Args:
        objects_data: VG objects.json data
        relationships_data: VG relationships.json data
        num_obj_classes: Target number of object classes
        num_pred_classes: Target number of predicate classes
        min_obj_count: Minimum count threshold for objects
        min_pred_count: Minimum count threshold for predicates

    Returns:
        (obj_to_idx, pred_to_idx) vocabulary mappings
    """
    print("Building vocabularies...")

    # Count object names
    obj_counter = Counter()
    for img_data in tqdm(objects_data, desc="Counting objects"):
        for obj in img_data.get('objects', []):
            names = obj.get('names', [])
            if names:
                name = clean_name(names[0])
                if name:
                    obj_counter[name] += 1

    # Count predicate names
    pred_counter = Counter()
    for img_data in tqdm(relationships_data, desc="Counting predicates"):
        for rel in img_data.get('relationships', []):
            pred = rel.get('predicate', '')
            pred = clean_name(pred)
            if pred:
                pred_counter[pred] += 1

    # Select top classes
    obj_classes = [
        name for name, count in obj_counter.most_common(num_obj_classes * 2)
        if count >= min_obj_count
    ][:num_obj_classes]

    pred_classes = [
        name for name, count in pred_counter.most_common(num_pred_classes * 2)
        if count >= min_pred_count
    ][:num_pred_classes]

    print(f"Selected {len(obj_classes)} object classes")
    print(f"Selected {len(pred_classes)} predicate classes")

    # Create mappings (0 is reserved for background/unknown)
    obj_to_idx = {'__background__': 0}
    for i, name in enumerate(obj_classes, start=1):
        obj_to_idx[name] = i

    pred_to_idx = {'__no_relation__': 0}
    for i, name in enumerate(pred_classes, start=1):
        pred_to_idx[name] = i

    return obj_to_idx, pred_to_idx


def get_object_name_mapping(
    objects_data: List[Dict],
    obj_to_idx: Dict[str, int]
) -> Dict[int, Tuple[str, int]]:
    """
    Build mapping from VG object ID to (cleaned_name, class_idx).

    Args:
        objects_data: VG objects.json data
        obj_to_idx: Object vocabulary

    Returns:
        {object_id: (name, class_idx)}
    """
    obj_id_to_info = {}

    for img_data in objects_data:
        for obj in img_data.get('objects', []):
            obj_id = obj.get('object_id')
            names = obj.get('names', [])

            if obj_id and names:
                name = clean_name(names[0])
                class_idx = obj_to_idx.get(name, 0)  # 0 for unknown
                obj_id_to_info[obj_id] = (name, class_idx)

    return obj_id_to_info


def process_relationships(
    relationships_data: List[Dict],
    objects_data: List[Dict],
    image_data: List[Dict],
    obj_to_idx: Dict[str, int],
    pred_to_idx: Dict[str, int]
) -> List[Dict]:
    """
    Process relationship annotations.

    Args:
        relationships_data: VG relationships.json
        objects_data: VG objects.json
        image_data: VG image_data.json
        obj_to_idx: Object vocabulary
        pred_to_idx: Predicate vocabulary

    Returns:
        List of processed image annotations
    """
    print("Processing relationships...")

    # Build image ID to image info mapping
    img_id_to_info = {img['image_id']: img for img in image_data}

    # Build object ID to object info mapping
    obj_id_to_obj = {}
    for img_data in objects_data:
        for obj in img_data.get('objects', []):
            obj_id = obj.get('object_id')
            if obj_id:
                obj_id_to_obj[obj_id] = obj

    # Process each image
    processed = []

    for img_rel_data in tqdm(relationships_data, desc="Processing images"):
        image_id = img_rel_data.get('image_id')
        relationships = img_rel_data.get('relationships', [])

        # Get image info
        img_info = img_id_to_info.get(image_id, {})
        width = img_info.get('width', 0)
        height = img_info.get('height', 0)
        url = img_info.get('url', '')

        if width == 0 or height == 0:
            continue

        # Process relationships
        valid_relations = []
        objects_in_image = {}  # obj_id -> object info

        for rel in relationships:
            # Get predicate
            pred_name = clean_name(rel.get('predicate', ''))
            pred_idx = pred_to_idx.get(pred_name, 0)

            if pred_idx == 0:  # Skip unknown predicates
                continue

            # Get subject
            subj = rel.get('subject', {})
            subj_id = subj.get('object_id')
            subj_names = subj.get('names', [subj.get('name', '')])
            if subj_names:
                subj_name = clean_name(subj_names[0] if isinstance(subj_names, list) else subj_names)
            else:
                subj_name = ''
            subj_idx = obj_to_idx.get(subj_name, 0)

            # Get object
            obj = rel.get('object', {})
            obj_id = obj.get('object_id')
            obj_names = obj.get('names', [obj.get('name', '')])
            if obj_names:
                obj_name = clean_name(obj_names[0] if isinstance(obj_names, list) else obj_names)
            else:
                obj_name = ''
            obj_idx = obj_to_idx.get(obj_name, 0)

            # Skip if subject or object is unknown
            if subj_idx == 0 or obj_idx == 0:
                continue

            # Get boxes
            def get_box(entity):
                x = entity.get('x', 0)
                y = entity.get('y', 0)
                w = entity.get('w', entity.get('width', 0))
                h = entity.get('h', entity.get('height', 0))
                return [x, y, x + w, y + h]

            subj_box = get_box(subj)
            obj_box = get_box(obj)

            # Validate boxes
            if (subj_box[2] <= subj_box[0] or subj_box[3] <= subj_box[1] or
                obj_box[2] <= obj_box[0] or obj_box[3] <= obj_box[1]):
                continue

            # Store objects
            if subj_id not in objects_in_image:
                objects_in_image[subj_id] = {
                    'object_id': subj_id,
                    'name': subj_name,
                    'class_idx': subj_idx,
                    'box': subj_box
                }
            if obj_id not in objects_in_image:
                objects_in_image[obj_id] = {
                    'object_id': obj_id,
                    'name': obj_name,
                    'class_idx': obj_idx,
                    'box': obj_box
                }

            # Store relation
            valid_relations.append({
                'subject_id': subj_id,
                'object_id': obj_id,
                'predicate': pred_name,
                'predicate_idx': pred_idx
            })

        # Only keep images with valid relations
        if len(valid_relations) >= 1 and len(objects_in_image) >= 2:
            # Re-index objects from 0
            obj_id_to_new_idx = {
                old_id: new_idx
                for new_idx, old_id in enumerate(objects_in_image.keys())
            }

            objects_list = [
                {
                    'idx': obj_id_to_new_idx[obj_id],
                    'name': obj_info['name'],
                    'class_idx': obj_info['class_idx'],
                    'box': obj_info['box']
                }
                for obj_id, obj_info in objects_in_image.items()
            ]

            relations_list = [
                {
                    'subject_idx': obj_id_to_new_idx[rel['subject_id']],
                    'object_idx': obj_id_to_new_idx[rel['object_id']],
                    'predicate': rel['predicate'],
                    'predicate_idx': rel['predicate_idx']
                }
                for rel in valid_relations
            ]

            processed.append({
                'image_id': image_id,
                'width': width,
                'height': height,
                'url': url,
                'objects': objects_list,
                'relations': relations_list
            })

    print(f"Processed {len(processed)} images with valid relations")
    return processed


def create_splits(
    processed_data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data into train/val/test sets.

    Args:
        processed_data: List of processed image annotations
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        (train_data, val_data, test_data)
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(processed_data))

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_data = [processed_data[i] for i in train_indices]
    val_data = [processed_data[i] for i in val_indices]
    test_data = [processed_data[i] for i in test_indices]

    print(f"Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    return train_data, val_data, test_data


def compute_statistics(
    train_data: List[Dict],
    obj_to_idx: Dict[str, int],
    pred_to_idx: Dict[str, int]
) -> Dict:
    """
    Compute dataset statistics for frequency bias.

    Args:
        train_data: Training data
        obj_to_idx: Object vocabulary
        pred_to_idx: Predicate vocabulary

    Returns:
        Statistics dictionary
    """
    print("Computing statistics...")

    num_objects = len(obj_to_idx)
    num_predicates = len(pred_to_idx)

    # Count (subject, predicate, object) triplets
    triplet_counts = defaultdict(int)
    # Count (subject, object) -> predicate
    pair_pred_counts = defaultdict(lambda: defaultdict(int))

    for img_data in train_data:
        for rel in img_data['relations']:
            subj_idx = img_data['objects'][rel['subject_idx']]['class_idx']
            obj_idx = img_data['objects'][rel['object_idx']]['class_idx']
            pred_idx = rel['predicate_idx']

            triplet_counts[(subj_idx, pred_idx, obj_idx)] += 1
            pair_pred_counts[(subj_idx, obj_idx)][pred_idx] += 1

    # Build frequency bias matrix: P(predicate | subject, object)
    # Shape: [num_objects, num_objects, num_predicates]
    freq_bias = np.zeros((num_objects, num_objects, num_predicates), dtype=np.float32)

    for (subj_idx, obj_idx), pred_counts in pair_pred_counts.items():
        total = sum(pred_counts.values())
        for pred_idx, count in pred_counts.items():
            freq_bias[subj_idx, obj_idx, pred_idx] = count / total

    # Count per-predicate relations
    pred_counts = defaultdict(int)
    for img_data in train_data:
        for rel in img_data['relations']:
            pred_counts[rel['predicate_idx']] += 1

    # Object counts
    obj_counts = defaultdict(int)
    for img_data in train_data:
        for obj in img_data['objects']:
            obj_counts[obj['class_idx']] += 1

    return {
        'num_objects': num_objects,
        'num_predicates': num_predicates,
        'num_train_images': len(train_data),
        'num_train_relations': sum(len(img['relations']) for img in train_data),
        'predicate_counts': dict(pred_counts),
        'object_counts': dict(obj_counts),
        'freq_bias_shape': list(freq_bias.shape)
    }


def generate_toy_dataset(
    processed_data: List[Dict],
    obj_to_idx: Dict[str, int],
    pred_to_idx: Dict[str, int],
    num_images: int = 1000,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate a small toy dataset for quick testing.

    Args:
        processed_data: Full processed data
        obj_to_idx: Object vocabulary
        pred_to_idx: Predicate vocabulary
        num_images: Number of images in toy dataset
        seed: Random seed

    Returns:
        (train_data, val_data, test_data) for toy dataset
    """
    np.random.seed(seed)

    # Sample images
    num_images = min(num_images, len(processed_data))
    indices = np.random.choice(len(processed_data), num_images, replace=False)
    sampled = [processed_data[i] for i in indices]

    # Split
    train_data, val_data, test_data = create_splits(sampled, seed=seed)

    return train_data, val_data, test_data


def preprocess_vg(
    raw_dir: str,
    out_dir: str,
    num_obj_classes: int = 150,
    num_pred_classes: int = 50,
    min_obj_count: int = 100,
    min_pred_count: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    create_toy: bool = True,
    toy_size: int = 1000
) -> None:
    """
    Main preprocessing function.

    Args:
        raw_dir: Directory with raw VG files
        out_dir: Output directory for processed files
        num_obj_classes: Number of object classes to keep
        num_pred_classes: Number of predicate classes to keep
        min_obj_count: Minimum object count threshold
        min_pred_count: Minimum predicate count threshold
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        create_toy: Whether to create toy dataset
        toy_size: Size of toy dataset
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check for required files
    required_files = ['objects.json', 'relationships.json', 'image_data.json']
    missing = [f for f in required_files if not (raw_dir / f).exists()]

    if missing:
        print("ERROR: Missing required files in raw directory:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease download Visual Genome files from:")
        print("  https://visualgenome.org/api/v0/api_home.html")
        print("\nExpected files:")
        for f in required_files:
            print(f"  - {raw_dir / f}")
        sys.exit(1)

    # Load raw data
    objects_data = load_json(raw_dir / 'objects.json')
    relationships_data = load_json(raw_dir / 'relationships.json')
    image_data = load_json(raw_dir / 'image_data.json')

    # Build vocabularies
    obj_to_idx, pred_to_idx = build_vocabularies(
        objects_data,
        relationships_data,
        num_obj_classes,
        num_pred_classes,
        min_obj_count,
        min_pred_count
    )

    # Save vocabularies
    save_json(obj_to_idx, out_dir / 'obj_classes.json')
    save_json(pred_to_idx, out_dir / 'pred_classes.json')

    # Create reverse mappings
    idx_to_obj = {v: k for k, v in obj_to_idx.items()}
    idx_to_pred = {v: k for k, v in pred_to_idx.items()}
    save_json(idx_to_obj, out_dir / 'idx_to_obj.json')
    save_json(idx_to_pred, out_dir / 'idx_to_pred.json')

    # Process relationships
    processed_data = process_relationships(
        relationships_data,
        objects_data,
        image_data,
        obj_to_idx,
        pred_to_idx
    )

    # Create splits
    train_data, val_data, test_data = create_splits(
        processed_data, train_ratio, val_ratio, seed
    )

    # Save splits
    save_json(train_data, out_dir / 'relationships_train.json')
    save_json(val_data, out_dir / 'relationships_val.json')
    save_json(test_data, out_dir / 'relationships_test.json')

    # Compute and save statistics
    stats = compute_statistics(train_data, obj_to_idx, pred_to_idx)
    save_json(stats, out_dir / 'statistics.json')

    # Create toy dataset
    if create_toy:
        print("\nCreating toy dataset...")
        toy_dir = out_dir / 'toy'
        toy_dir.mkdir(exist_ok=True)

        toy_train, toy_val, toy_test = generate_toy_dataset(
            processed_data, obj_to_idx, pred_to_idx, toy_size, seed
        )

        save_json(toy_train, toy_dir / 'relationships_train.json')
        save_json(toy_val, toy_dir / 'relationships_val.json')
        save_json(toy_test, toy_dir / 'relationships_test.json')

        # Copy vocabs to toy dir
        save_json(obj_to_idx, toy_dir / 'obj_classes.json')
        save_json(pred_to_idx, toy_dir / 'pred_classes.json')
        save_json(idx_to_obj, toy_dir / 'idx_to_obj.json')
        save_json(idx_to_pred, toy_dir / 'idx_to_pred.json')

        # Compute toy stats
        toy_stats = compute_statistics(toy_train, obj_to_idx, pred_to_idx)
        save_json(toy_stats, toy_dir / 'statistics.json')

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print(f"Output directory: {out_dir}")
    print(f"Object classes: {len(obj_to_idx)}")
    print(f"Predicate classes: {len(pred_to_idx)}")
    print(f"Train images: {len(train_data)}")
    print(f"Val images: {len(val_data)}")
    print(f"Test images: {len(test_data)}")
    if create_toy:
        print(f"Toy dataset: {toy_size} images")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Visual Genome for VRD'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--raw_dir', type=str, default='data/raw',
        help='Directory with raw VG files'
    )
    parser.add_argument(
        '--out_dir', type=str, default='data/processed',
        help='Output directory'
    )
    parser.add_argument(
        '--num_obj_classes', type=int, default=150,
        help='Number of object classes'
    )
    parser.add_argument(
        '--num_pred_classes', type=int, default=50,
        help='Number of predicate classes'
    )
    parser.add_argument(
        '--min_obj_count', type=int, default=100,
        help='Minimum object count'
    )
    parser.add_argument(
        '--min_pred_count', type=int, default=100,
        help='Minimum predicate count'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no_toy', action='store_true',
        help='Skip toy dataset creation'
    )
    parser.add_argument(
        '--toy_size', type=int, default=1000,
        help='Toy dataset size'
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override with config values
        dataset_cfg = config.get('dataset', {})
        args.raw_dir = dataset_cfg.get('raw_dir', args.raw_dir)
        args.out_dir = dataset_cfg.get('processed_dir', args.out_dir)
        args.num_obj_classes = dataset_cfg.get('num_obj_classes', args.num_obj_classes)
        args.num_pred_classes = dataset_cfg.get('num_pred_classes', args.num_pred_classes)
        args.seed = config.get('seed', args.seed)

    preprocess_vg(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        num_obj_classes=args.num_obj_classes,
        num_pred_classes=args.num_pred_classes,
        min_obj_count=args.min_obj_count,
        min_pred_count=args.min_pred_count,
        seed=args.seed,
        create_toy=not args.no_toy,
        toy_size=args.toy_size
    )


if __name__ == '__main__':
    main()

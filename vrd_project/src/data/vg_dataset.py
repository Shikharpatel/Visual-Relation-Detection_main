"""
Visual Genome Dataset for VRD
==============================

PyTorch Dataset and DataLoader for Visual Genome relationships.
Supports PredCls, SGCls, and SGDet evaluation modes.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from .transforms import VRDTransform, get_train_transforms, get_eval_transforms


class VGDataset(Dataset):
    """
    Visual Genome Dataset for Visual Relationship Detection.

    Supports three modes:
    - predcls: Ground truth boxes and labels given, predict predicates only
    - sgcls: Ground truth boxes given, predict labels and predicates
    - sgdet: Predict boxes, labels, and predicates

    Args:
        data_dir: Directory with processed VG data
        split: One of 'train', 'val', 'test'
        mode: Evaluation mode ('predcls', 'sgcls', 'sgdet')
        transform: Image transforms
        image_dir: Directory with VG images (optional for predcls/sgcls with features)
        use_features: If True, load precomputed features instead of images
        features_dir: Directory with precomputed features
        max_objects: Maximum objects per image (for memory efficiency)
        max_relations: Maximum relations per image
        filter_empty: Filter images with no valid relations
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        mode: str = 'predcls',
        transform: Optional[Callable] = None,
        image_dir: Optional[str] = None,
        use_features: bool = False,
        features_dir: Optional[str] = None,
        max_objects: int = 100,
        max_relations: int = 100,
        filter_empty: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.mode = mode.lower()
        self.transform = transform
        self.image_dir = Path(image_dir) if image_dir else None
        self.use_features = use_features
        self.features_dir = Path(features_dir) if features_dir else None
        self.max_objects = max_objects
        self.max_relations = max_relations

        # Validate mode
        assert self.mode in ['predcls', 'sgcls', 'sgdet'], \
            f"Invalid mode: {mode}. Must be 'predcls', 'sgcls', or 'sgdet'"

        # Load vocabularies
        self.obj_classes = self._load_json('obj_classes.json')
        self.pred_classes = self._load_json('pred_classes.json')
        self.idx_to_obj = self._load_json('idx_to_obj.json')
        self.idx_to_pred = self._load_json('idx_to_pred.json')

        # Convert string keys to int for idx_to_* mappings
        self.idx_to_obj = {int(k): v for k, v in self.idx_to_obj.items()}
        self.idx_to_pred = {int(k): v for k, v in self.idx_to_pred.items()}

        self.num_obj_classes = len(self.obj_classes)
        self.num_pred_classes = len(self.pred_classes)

        # Load annotations
        self.annotations = self._load_json(f'relationships_{split}.json')

        # Filter empty images if requested
        if filter_empty:
            self.annotations = [
                ann for ann in self.annotations
                if len(ann.get('relations', [])) > 0
            ]

        # Load statistics
        self.statistics = self._load_json('statistics.json')

        # Build image path cache once at init — avoids 9 exists() calls per __getitem__
        self._image_path_cache: Dict[int, Path] = {}
        if self.image_dir is not None:
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for p in self.image_dir.rglob(ext):
                    try:
                        self._image_path_cache[int(p.stem)] = p
                    except ValueError:
                        pass

        print(f"Loaded VG {split} split: {len(self.annotations)} images")
        print(f"  Object classes: {self.num_obj_classes}")
        print(f"  Predicate classes: {self.num_pred_classes}")
        print(f"  Mode: {self.mode}")

    def _load_json(self, filename: str) -> Any:
        """Load JSON file from data directory."""
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns dict with:
            - image_id: int
            - image: tensor [C, H, W] (if available)
            - boxes: tensor [N, 4] object boxes
            - labels: tensor [N] object class indices
            - relation_pairs: tensor [R, 2] (subject_idx, object_idx) pairs
            - relation_labels: tensor [R] predicate class indices
            - image_size: tuple (height, width)
        """
        ann = self.annotations[idx]

        # Basic info
        image_id = ann['image_id']
        width = ann['width']
        height = ann['height']

        # Load objects
        objects = ann['objects'][:self.max_objects]
        boxes = np.array([obj['box'] for obj in objects], dtype=np.float32)
        labels = np.array([obj['class_idx'] for obj in objects], dtype=np.int64)

        # Load relations
        relations = ann['relations'][:self.max_relations]

        # Filter relations to only include valid object indices
        valid_obj_indices = set(range(len(objects)))
        relations = [
            rel for rel in relations
            if rel['subject_idx'] in valid_obj_indices
            and rel['object_idx'] in valid_obj_indices
        ]

        relation_pairs = np.array(
            [[rel['subject_idx'], rel['object_idx']] for rel in relations],
            dtype=np.int64
        )
        relation_labels = np.array(
            [rel['predicate_idx'] for rel in relations],
            dtype=np.int64
        )

        # Handle empty relations case
        if len(relations) == 0:
            relation_pairs = np.zeros((0, 2), dtype=np.int64)
            relation_labels = np.zeros((0,), dtype=np.int64)

        # Load image if needed (for SGDet or visual features)
        image = None
        if self.image_dir is not None:
            image_path = self._get_image_path(image_id)
            if image_path.exists():
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)

        # Apply transforms
        if self.transform is not None and image is not None:
            transformed = self.transform(
                image=image,
                boxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['boxes']
            # Note: labels don't change with spatial transforms

        # Convert to tensors
        sample = {
            'image_id': image_id,
            'boxes': torch.from_numpy(boxes),
            'labels': torch.from_numpy(labels),
            'relation_pairs': torch.from_numpy(relation_pairs),
            'relation_labels': torch.from_numpy(relation_labels),
            'image_size': (height, width),
            'num_objects': len(objects),
            'num_relations': len(relations)
        }

        if image is not None:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            sample['image'] = image

        return sample

    def _get_image_path(self, image_id: int) -> Path:
        """Get path to image file using pre-built cache (O(1) lookup)."""
        return self._image_path_cache.get(image_id, self.image_dir / f"{image_id}.jpg")

    def get_object_name(self, idx: int) -> str:
        """Get object class name by index."""
        return self.idx_to_obj.get(idx, f'unknown_{idx}')

    def get_predicate_name(self, idx: int) -> str:
        """Get predicate name by index."""
        return self.idx_to_pred.get(idx, f'unknown_{idx}')

    def get_triplet_string(
        self,
        subj_label: int,
        pred_label: int,
        obj_label: int
    ) -> str:
        """Convert triplet indices to readable string."""
        subj = self.get_object_name(subj_label)
        pred = self.get_predicate_name(pred_label)
        obj = self.get_object_name(obj_label)
        return f"{subj} --[{pred}]--> {obj}"


class VGCollator:
    """
    Custom collator for VG dataset to handle variable-length data.

    Pads boxes, labels, and relations to maximum batch size.
    """

    def __init__(self, mode: str = 'predcls'):
        self.mode = mode

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Collated batch dictionary
        """
        batch_size = len(batch)

        # Find max sizes
        max_objects = max(s['num_objects'] for s in batch)
        max_relations = max(s['num_relations'] for s in batch)

        # At least 1 to avoid empty tensors
        max_objects = max(max_objects, 1)
        max_relations = max(max_relations, 1)

        # Prepare tensors
        boxes = torch.zeros(batch_size, max_objects, 4)
        labels = torch.zeros(batch_size, max_objects, dtype=torch.long)
        relation_pairs = torch.zeros(batch_size, max_relations, 2, dtype=torch.long)
        relation_labels = torch.zeros(batch_size, max_relations, dtype=torch.long)

        # Masks
        object_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)
        relation_mask = torch.zeros(batch_size, max_relations, dtype=torch.bool)

        # Fill tensors
        image_ids = []
        image_sizes = []
        images = []

        for i, sample in enumerate(batch):
            n_obj = sample['num_objects']
            n_rel = sample['num_relations']

            boxes[i, :n_obj] = sample['boxes']
            labels[i, :n_obj] = sample['labels']
            object_mask[i, :n_obj] = True

            if n_rel > 0:
                relation_pairs[i, :n_rel] = sample['relation_pairs']
                relation_labels[i, :n_rel] = sample['relation_labels']
                relation_mask[i, :n_rel] = True

            image_ids.append(sample['image_id'])
            image_sizes.append(sample['image_size'])

            if 'image' in sample:
                images.append(sample['image'])

        collated = {
            'image_ids': image_ids,
            'boxes': boxes,
            'labels': labels,
            'relation_pairs': relation_pairs,
            'relation_labels': relation_labels,
            'object_mask': object_mask,
            'relation_mask': relation_mask,
            'image_sizes': image_sizes,
            'batch_size': batch_size
        }

        if images:
            # Stack images if all same size, else return list
            try:
                collated['images'] = torch.stack(images)
            except RuntimeError:
                collated['images'] = images

        return collated


def create_dataloader(
    data_dir: str,
    split: str,
    mode: str = 'predcls',
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = None,
    image_dir: Optional[str] = None,
    transform: Optional[Callable] = None,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for VG dataset.

    Args:
        data_dir: Path to processed data directory
        split: 'train', 'val', or 'test'
        mode: Evaluation mode
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (default: True for train, False otherwise)
        image_dir: Directory with images
        transform: Image transforms
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == 'train')

    # Default transforms
    if transform is None:
        if split == 'train':
            transform = get_train_transforms()
        else:
            transform = get_eval_transforms()

    dataset = VGDataset(
        data_dir=data_dir,
        split=split,
        mode=mode,
        transform=transform,
        image_dir=image_dir
    )

    collator = VGCollator(mode=mode)

    # Import worker init function
    from ..utils.seed import worker_init_fn

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,   # drop incomplete last batch during training only
    )

    return loader


class SyntheticVGDataset(Dataset):
    """
    Synthetic dataset for testing without real VG data.

    Generates random data with the same structure as VG.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_obj_classes: int = 150,
        num_pred_classes: int = 50,
        image_size: Tuple[int, int] = (600, 800),
        min_objects: int = 3,
        max_objects: int = 20,
        min_relations: int = 1,
        max_relations: int = 30,
        seed: int = 42
    ):
        """
        Initialize synthetic dataset.

        Args:
            num_samples: Number of samples to generate
            num_obj_classes: Number of object classes
            num_pred_classes: Number of predicate classes
            image_size: (height, width) of synthetic images
            min_objects: Minimum objects per image
            max_objects: Maximum objects per image
            min_relations: Minimum relations per image
            max_relations: Maximum relations per image
            seed: Random seed
        """
        self.num_samples = num_samples
        self.num_obj_classes = num_obj_classes
        self.num_pred_classes = num_pred_classes
        self.image_size = image_size
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_relations = min_relations
        self.max_relations = max_relations

        np.random.seed(seed)
        self._generate_data()

        # Create dummy vocabulary mappings
        self.idx_to_obj = {i: f'object_{i}' for i in range(num_obj_classes)}
        self.idx_to_pred = {i: f'predicate_{i}' for i in range(num_pred_classes)}

    def _generate_data(self):
        """Generate random data."""
        self.data = []

        for i in range(self.num_samples):
            h, w = self.image_size
            n_objects = np.random.randint(self.min_objects, self.max_objects + 1)
            n_relations = np.random.randint(self.min_relations, self.max_relations + 1)
            n_relations = min(n_relations, n_objects * (n_objects - 1))  # Max possible pairs

            # Generate random boxes
            boxes = []
            for _ in range(n_objects):
                x1 = np.random.randint(0, w - 50)
                y1 = np.random.randint(0, h - 50)
                x2 = np.random.randint(x1 + 20, min(x1 + 200, w))
                y2 = np.random.randint(y1 + 20, min(y1 + 200, h))
                boxes.append([x1, y1, x2, y2])
            boxes = np.array(boxes, dtype=np.float32)

            # Generate random labels
            labels = np.random.randint(1, self.num_obj_classes, size=n_objects)

            # Generate random relation pairs (no self-loops)
            all_pairs = [
                (i, j) for i in range(n_objects) for j in range(n_objects) if i != j
            ]
            if len(all_pairs) > 0:
                pair_indices = np.random.choice(
                    len(all_pairs),
                    size=min(n_relations, len(all_pairs)),
                    replace=False
                )
                relation_pairs = np.array([all_pairs[idx] for idx in pair_indices])
                relation_labels = np.random.randint(1, self.num_pred_classes, size=len(relation_pairs))
            else:
                relation_pairs = np.zeros((0, 2), dtype=np.int64)
                relation_labels = np.zeros((0,), dtype=np.int64)

            self.data.append({
                'image_id': i,
                'boxes': boxes,
                'labels': labels,
                'relation_pairs': relation_pairs,
                'relation_labels': relation_labels,
                'image_size': self.image_size
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]

        return {
            'image_id': sample['image_id'],
            'boxes': torch.from_numpy(sample['boxes']),
            'labels': torch.from_numpy(sample['labels'].astype(np.int64)),
            'relation_pairs': torch.from_numpy(sample['relation_pairs'].astype(np.int64)),
            'relation_labels': torch.from_numpy(sample['relation_labels'].astype(np.int64)),
            'image_size': sample['image_size'],
            'num_objects': len(sample['boxes']),
            'num_relations': len(sample['relation_pairs'])
        }

    def get_object_name(self, idx: int) -> str:
        return self.idx_to_obj.get(idx, f'unknown_{idx}')

    def get_predicate_name(self, idx: int) -> str:
        return self.idx_to_pred.get(idx, f'unknown_{idx}')

"""
Tests for VRD Dataset
======================

Tests for data loading, preprocessing, and transforms.
"""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.vg_dataset import SyntheticVGDataset, VGCollator
from src.data.transforms import (
    VRDTransform, Resize, RandomHorizontalFlip,
    Normalize, ToTensor, get_train_transforms, get_eval_transforms
)


class TestSyntheticDataset:
    """Tests for SyntheticVGDataset."""

    def test_dataset_creation(self):
        """Test that dataset can be created."""
        dataset = SyntheticVGDataset(
            num_samples=100,
            num_obj_classes=50,
            num_pred_classes=20
        )
        assert len(dataset) == 100

    def test_dataset_item_shapes(self):
        """Test that dataset items have correct shapes."""
        dataset = SyntheticVGDataset(
            num_samples=10,
            num_obj_classes=50,
            num_pred_classes=20,
            min_objects=3,
            max_objects=10
        )

        sample = dataset[0]

        # Check keys
        assert 'boxes' in sample
        assert 'labels' in sample
        assert 'relation_pairs' in sample
        assert 'relation_labels' in sample
        assert 'image_size' in sample

        # Check types
        assert isinstance(sample['boxes'], torch.Tensor)
        assert isinstance(sample['labels'], torch.Tensor)
        assert isinstance(sample['relation_pairs'], torch.Tensor)
        assert isinstance(sample['relation_labels'], torch.Tensor)

        # Check shapes
        n_objects = sample['num_objects']
        n_relations = sample['num_relations']

        assert sample['boxes'].shape == (n_objects, 4)
        assert sample['labels'].shape == (n_objects,)
        assert sample['relation_pairs'].shape == (n_relations, 2)
        assert sample['relation_labels'].shape == (n_relations,)

    def test_dataset_box_validity(self):
        """Test that boxes are valid (x2 > x1, y2 > y1)."""
        dataset = SyntheticVGDataset(num_samples=50)

        for i in range(len(dataset)):
            sample = dataset[i]
            boxes = sample['boxes'].numpy()

            # Check x2 > x1 and y2 > y1
            assert np.all(boxes[:, 2] > boxes[:, 0]), f"Invalid boxes at sample {i}"
            assert np.all(boxes[:, 3] > boxes[:, 1]), f"Invalid boxes at sample {i}"

    def test_dataset_label_range(self):
        """Test that labels are within valid range."""
        num_obj_classes = 50
        num_pred_classes = 20

        dataset = SyntheticVGDataset(
            num_samples=50,
            num_obj_classes=num_obj_classes,
            num_pred_classes=num_pred_classes
        )

        for i in range(len(dataset)):
            sample = dataset[i]

            # Object labels should be in [1, num_obj_classes)
            assert sample['labels'].min() >= 1
            assert sample['labels'].max() < num_obj_classes

            # Predicate labels should be in [1, num_pred_classes)
            if len(sample['relation_labels']) > 0:
                assert sample['relation_labels'].min() >= 1
                assert sample['relation_labels'].max() < num_pred_classes

    def test_dataset_relation_pairs_validity(self):
        """Test that relation pairs have valid indices and no self-loops."""
        dataset = SyntheticVGDataset(num_samples=50)

        for i in range(len(dataset)):
            sample = dataset[i]
            n_objects = sample['num_objects']
            pairs = sample['relation_pairs'].numpy()

            # Check indices are valid
            if len(pairs) > 0:
                assert pairs[:, 0].max() < n_objects
                assert pairs[:, 1].max() < n_objects

                # Check no self-loops
                assert np.all(pairs[:, 0] != pairs[:, 1])


class TestVGCollator:
    """Tests for VGCollator."""

    def test_collator_batching(self):
        """Test that collator properly batches samples."""
        dataset = SyntheticVGDataset(num_samples=10)
        collator = VGCollator(mode='predcls')

        # Create a batch
        batch = [dataset[i] for i in range(4)]
        collated = collator(batch)

        # Check batch dimension
        assert collated['boxes'].shape[0] == 4
        assert collated['labels'].shape[0] == 4
        assert collated['relation_pairs'].shape[0] == 4
        assert collated['relation_labels'].shape[0] == 4

    def test_collator_padding(self):
        """Test that collator properly pads variable-length data."""
        dataset = SyntheticVGDataset(
            num_samples=10,
            min_objects=2,
            max_objects=20
        )
        collator = VGCollator(mode='predcls')

        batch = [dataset[i] for i in range(4)]
        collated = collator(batch)

        # Padded dimension should be max of batch
        max_objects = max(s['num_objects'] for s in batch)
        max_relations = max(s['num_relations'] for s in batch)

        assert collated['boxes'].shape[1] == max_objects
        assert collated['relation_pairs'].shape[1] == max_relations

    def test_collator_masks(self):
        """Test that collator produces correct masks."""
        dataset = SyntheticVGDataset(
            num_samples=10,
            min_objects=3,
            max_objects=10
        )
        collator = VGCollator(mode='predcls')

        batch = [dataset[i] for i in range(4)]
        collated = collator(batch)

        # Check object mask
        for i, sample in enumerate(batch):
            n_obj = sample['num_objects']
            mask = collated['object_mask'][i]
            assert mask[:n_obj].all()  # First n_obj should be True
            if n_obj < mask.shape[0]:
                assert not mask[n_obj:].any()  # Rest should be False


class TestTransforms:
    """Tests for data transforms."""

    def test_resize_transform(self):
        """Test resize transform."""
        transform = Resize((300, 400))

        image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32)
        labels = np.array([1, 2])

        data = {'image': image, 'boxes': boxes, 'labels': labels}
        result = transform(data)

        assert result['image'].shape == (300, 400, 3)
        # Boxes should be scaled proportionally
        assert result['boxes'][0, 2] < 200  # x2 should be smaller

    def test_horizontal_flip(self):
        """Test horizontal flip transform."""
        transform = RandomHorizontalFlip(p=1.0)  # Always flip

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :100] = 255  # Left half is white

        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([1])

        data = {'image': image, 'boxes': boxes, 'labels': labels}
        result = transform(data)

        # After flip, white should be on right
        assert result['image'][:, 100:].mean() > 200

        # Box should be flipped
        assert result['boxes'][0, 0] > 100  # x1 should be > 100 now

    def test_normalize_transform(self):
        """Test normalize transform."""
        transform = Normalize()

        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        labels = np.array([1])

        data = {'image': image, 'boxes': boxes, 'labels': labels}
        result = transform(data)

        # Image should be a tensor now
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 100, 100)

    def test_train_transforms(self):
        """Test train transforms pipeline."""
        transform = get_train_transforms(size=(224, 224))

        image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([1])

        result = transform(image=image, boxes=boxes, labels=labels)

        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape[1:] == (224, 224)

    def test_eval_transforms(self):
        """Test eval transforms pipeline."""
        transform = get_eval_transforms(size=(224, 224))

        image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([1])

        result = transform(image=image, boxes=boxes, labels=labels)

        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape[1:] == (224, 224)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

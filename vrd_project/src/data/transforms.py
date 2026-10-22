"""
Image Transforms for VRD
========================

Provides data augmentation and preprocessing transforms for training and evaluation.
Handles both image and bounding box transformations.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import torch


class VRDTransform:
    """
    Composite transform for VRD that handles images and boxes together.

    Args:
        transforms: List of transform operations
    """

    def __init__(self, transforms: List[Any]):
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Apply transforms to image and boxes.

        Args:
            image: Image as numpy array [H, W, C]
            boxes: Boxes as numpy array [N, 4] in [x1, y1, x2, y2] format
            labels: Object labels as numpy array [N]

        Returns:
            Dictionary with transformed image, boxes, and labels
        """
        result = {'image': image, 'boxes': boxes.copy(), 'labels': labels}

        for transform in self.transforms:
            result = transform(result)

        return result


class Resize:
    """Resize image and boxes to target size."""

    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size: Target (height, width)
        """
        self.size = size

    def __call__(self, data: Dict) -> Dict:
        image = data['image']
        boxes = data['boxes']

        orig_h, orig_w = image.shape[:2]
        new_h, new_w = self.size

        # Resize image
        from PIL import Image
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)
        image = np.array(pil_image)

        # Scale boxes
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        data['image'] = image
        data['boxes'] = boxes
        return data


class RandomHorizontalFlip:
    """Randomly flip image and boxes horizontally."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, data: Dict) -> Dict:
        if np.random.random() < self.p:
            image = data['image']
            boxes = data['boxes']

            # Flip image
            image = np.fliplr(image).copy()
            width = image.shape[1]

            # Flip boxes
            if len(boxes) > 0:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = width - x2
                boxes[:, 2] = width - x1

            data['image'] = image
            data['boxes'] = boxes

        return data


class RandomCrop:
    """Random crop that keeps all boxes inside."""

    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, data: Dict) -> Dict:
        image = data['image']
        boxes = data['boxes']

        if len(boxes) == 0:
            return data

        h, w = image.shape[:2]

        # Find bounds that contain all boxes
        min_x = boxes[:, 0].min()
        min_y = boxes[:, 1].min()
        max_x = boxes[:, 2].max()
        max_y = boxes[:, 3].max()

        # Random scale
        scale = np.random.uniform(self.min_scale, self.max_scale)

        # Crop dimensions
        crop_h = int(h * scale)
        crop_w = int(w * scale)

        # Ensure crop contains all boxes
        crop_x_min = max(0, int(max_x - crop_w))
        crop_x_max = min(w - crop_w, int(min_x))
        crop_y_min = max(0, int(max_y - crop_h))
        crop_y_max = min(h - crop_h, int(min_y))

        if crop_x_min < crop_x_max and crop_y_min < crop_y_max:
            crop_x = np.random.randint(crop_x_min, crop_x_max + 1)
            crop_y = np.random.randint(crop_y_min, crop_y_max + 1)

            # Crop image
            image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w].copy()

            # Adjust boxes
            boxes[:, [0, 2]] -= crop_x
            boxes[:, [1, 3]] -= crop_y

            # Clip to image bounds
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, crop_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, crop_h)

            data['image'] = image
            data['boxes'] = boxes

        return data


class ColorJitter:
    """Random color jittering."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, data: Dict) -> Dict:
        image = data['image'].astype(np.float32)

        # Brightness
        if self.brightness > 0:
            factor = 1 + np.random.uniform(-self.brightness, self.brightness)
            image = image * factor

        # Contrast
        if self.contrast > 0:
            factor = 1 + np.random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean

        # Clip values
        image = np.clip(image, 0, 255).astype(np.uint8)

        data['image'] = image
        return data


class Normalize:
    """Normalize image to [0, 1] or with ImageNet statistics."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        to_tensor: bool = True
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_tensor = to_tensor

    def __call__(self, data: Dict) -> Dict:
        image = data['image'].astype(np.float32) / 255.0

        # Normalize
        image = (image - self.mean) / self.std

        if self.to_tensor:
            # HWC -> CHW
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)

        data['image'] = image
        return data


class ToTensor:
    """Convert image and boxes to PyTorch tensors."""

    def __call__(self, data: Dict) -> Dict:
        image = data['image']
        boxes = data['boxes']

        # Image: HWC -> CHW, scale to [0, 1]
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)

        # Boxes
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes.astype(np.float32))

        data['image'] = image
        data['boxes'] = boxes
        return data


class ClipBoxes:
    """Clip boxes to image bounds and remove invalid ones."""

    def __init__(self, min_area: float = 100):
        self.min_area = min_area

    def __call__(self, data: Dict) -> Dict:
        image = data['image']
        boxes = data['boxes']
        labels = data['labels']

        if len(boxes) == 0:
            return data

        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            h, w = image.shape[:2]

        # Clip boxes
        if isinstance(boxes, torch.Tensor):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        else:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

        # Remove invalid boxes (zero area)
        if isinstance(boxes, torch.Tensor):
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid = areas >= self.min_area
            boxes = boxes[valid]
            labels = labels[valid]
        else:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid = areas >= self.min_area
            boxes = boxes[valid]
            labels = labels[valid]

        data['boxes'] = boxes
        data['labels'] = labels
        return data


def get_train_transforms(
    size: Optional[Tuple[int, int]] = None,
    flip_p: float = 0.5,
    color_jitter: bool = True,
    normalize: bool = True
) -> VRDTransform:
    """
    Get default training transforms.

    Args:
        size: Optional resize target (height, width)
        flip_p: Horizontal flip probability
        color_jitter: Whether to apply color jitter
        normalize: Whether to normalize with ImageNet stats

    Returns:
        VRDTransform instance
    """
    transforms = []

    if size is not None:
        transforms.append(Resize(size))

    transforms.append(RandomHorizontalFlip(p=flip_p))

    if color_jitter:
        transforms.append(ColorJitter())

    transforms.append(ClipBoxes())

    if normalize:
        transforms.append(Normalize())
    else:
        transforms.append(ToTensor())

    return VRDTransform(transforms)


def get_eval_transforms(
    size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> VRDTransform:
    """
    Get default evaluation transforms.

    Args:
        size: Optional resize target (height, width)
        normalize: Whether to normalize with ImageNet stats

    Returns:
        VRDTransform instance
    """
    transforms = []

    if size is not None:
        transforms.append(Resize(size))

    transforms.append(ClipBoxes())

    if normalize:
        transforms.append(Normalize())
    else:
        transforms.append(ToTensor())

    return VRDTransform(transforms)


class IdentityTransform:
    """Identity transform that just returns data as-is with tensor conversion."""

    def __call__(
        self,
        image: Optional[np.ndarray] = None,
        boxes: np.ndarray = None,
        labels: np.ndarray = None
    ) -> Dict:
        result = {}

        if image is not None:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image.astype(np.float32) / 255.0)
                if image.dim() == 3:
                    image = image.permute(2, 0, 1)
            result['image'] = image

        if boxes is not None:
            if isinstance(boxes, np.ndarray):
                boxes = torch.from_numpy(boxes.astype(np.float32))
            result['boxes'] = boxes

        if labels is not None:
            result['labels'] = labels

        return result

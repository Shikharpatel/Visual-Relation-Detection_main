"""YOLO-World open-vocabulary object detector wrapper.

YOLO-World is accessed through the `ultralytics` package (>=8.1).
Weights are auto-downloaded by ultralytics on first use.

Isolation guarantee: if the ultralytics API changes, this is the ONLY
file that needs to be updated.

Usage:
    detector = YOLOWorldDetector(
        model_path="yolov8s-world.pt",
        vocab=["person", "car", "dog"],
        conf_thresh=0.3,
        top_k=10,
    )
    detections = detector.detect(image_pil, image_id=12345)
    # -> [{"label": "person", "box": [x1,y1,x2,y2], "score": 0.87}, ...]
"""

import hashlib
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from src.utils.cache import DiskCache


class YOLOWorldDetector:
    """Open-vocabulary detector backed by YOLO-World.

    Args:
        model_path: Path or model name for ultralytics (e.g. "yolov8s-world.pt").
                    Downloaded automatically if not on disk.
        vocab:      List of object category strings (text prompts).
        conf_thresh: Minimum confidence to keep a detection.
        top_k:      Keep at most this many boxes per image (by confidence).
        nms_iou:    IoU threshold for NMS (handled internally by ultralytics).
        cache:      DiskCache for storing detection results.
        device:     "cuda" or "cpu".
    """

    def __init__(
        self,
        model_path: str = "yolov8s-world.pt",
        vocab: Optional[List[str]] = None,
        conf_thresh: float = 0.3,
        top_k: int = 10,
        nms_iou: float = 0.5,
        cache: Optional[DiskCache] = None,
        device: str = "cuda",
    ):
        try:
            from ultralytics import YOLOWorld
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics>=8.1"
            )

        self.model = YOLOWorld(model_path)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_iou = nms_iou
        self.cache = cache
        self.device = device
        self._vocab: List[str] = []
        self._vocab_sig: str = ""

        if vocab:
            self.set_vocab(vocab)

    def set_vocab(self, vocab: List[str]) -> None:
        """Update the open-vocabulary text prompts sent to YOLO-World."""
        self._vocab = vocab
        self._vocab_sig = hashlib.sha256(
            "|".join(sorted(vocab)).encode()
        ).hexdigest()[:12]
        self.model.set_classes(vocab)

    def detect(
        self,
        image: Image.Image,
        image_id: Optional[Union[int, str]] = None,
    ) -> List[Dict]:
        """Run detection on a PIL image.

        Returns:
            List of dicts sorted by descending confidence, each with:
                label (str), box ([x1, y1, x2, y2] ints), score (float).
            At most `top_k` results returned.
        """
        if not self._vocab:
            raise RuntimeError("Call set_vocab() before detect().")

        # Check cache
        if self.cache is not None and image_id is not None:
            cache_key = f"det_{image_id}_{self._vocab_sig}"
            if self.cache.exists(cache_key):
                return self.cache.load(cache_key)

        results = self._run_model(image)

        if self.cache is not None and image_id is not None:
            self.cache.save(cache_key, results)

        return results

    def _run_model(self, image: Image.Image) -> List[Dict]:
        """Internal: run ultralytics inference and parse results."""
        import numpy as np

        # ultralytics accepts PIL images directly
        preds = self.model.predict(
            source=image,
            conf=self.conf_thresh,
            iou=self.nms_iou,
            device=self.device,
            verbose=False,
        )

        detections = []
        if not preds:
            return detections

        result = preds[0]  # single image

        # result.boxes: ultralytics Boxes object
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()    # (N, 4)
        scores     = result.boxes.conf.cpu().numpy()    # (N,)
        cls_ids    = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

        names = result.names  # dict: int -> str

        for box, score, cls_id in zip(boxes_xyxy, scores, cls_ids):
            label = names.get(cls_id, str(cls_id))
            detections.append({
                "label": label,
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "score": float(score),
            })

        # Sort by confidence, keep top_k
        detections.sort(key=lambda x: x["score"], reverse=True)
        return detections[: self.top_k]

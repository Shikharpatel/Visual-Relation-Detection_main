"""Full inference pipeline: image -> ranked relationship triplets.

Pipeline:
  1. YOLO-World detects objects (open-vocabulary)
  2. CLIP encodes each object crop
  3. Build all N×(N-1) subject-object pairs
  4. RelationHead predicts predicate logits per pair
  5. FusionModule applies LLM prior
  6. Rank all triplets by confidence
  7. Output JSON

Usage:
    cd open_vocab_vrd/
    python scripts/infer.py \\
        --image path/to/image.jpg \\
        --config configs/default.yaml \\
        --checkpoint outputs/run_.../best.pt \\
        --output predictions.json \\
        --top_k 20
"""

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from src.detectors.yoloworld_detector import YOLOWorldDetector
from src.encoders.clip_encoder import CLIPEncoder
from src.encoders.spatial_features import compute_spatial_features
from src.models.vrd_model import load_model
from src.utils.cache import DiskCache
from src.utils.io import load_json, load_yaml
from src.utils.labels import PriorLookup


def build_pairs(detections: list) -> list:
    """Return all ordered (subject, object) index pairs from detections.

    Skips self-pairs. For N detections returns N*(N-1) pairs.
    """
    n = len(detections)
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def run_inference(
    image_path: str,
    cfg: dict,
    checkpoint: str,
    top_k: int = 20,
    device_str: str = "cuda",
) -> list:
    """Run the full VRD pipeline on a single image.

    Returns a list of dicts sorted by descending confidence.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    processed_dir = Path(cfg["data"]["processed_dir"])
    pred_vocab = load_json(processed_dir / "pred_vocab.json")
    num_predicates = len(pred_vocab)

    # -- Cache --
    cache_root = Path(cfg["data"]["cache_dir"])
    cache_enabled = cfg["cache"].get("enabled", True)
    clip_cache = DiskCache(
        cache_root / cfg["cache"].get("clip_subdir", "clip_embeddings"),
        enabled=cache_enabled,
    )
    det_cache = DiskCache(
        cache_root / cfg["cache"].get("detector_subdir", "detections"),
        enabled=cache_enabled,
    )

    # -- YOLO-World detector --
    det_cfg = cfg["detector"]
    vocab_file = Path(det_cfg.get("vocab_file", "configs/predicates.yaml"))
    from src.utils.io import load_yaml as _load_yaml
    vocab = _load_yaml(vocab_file).get("objects", [])

    detector = YOLOWorldDetector(
        model_path=det_cfg.get("model_path", "yolov8s-world.pt"),
        vocab=vocab,
        conf_thresh=det_cfg.get("conf_thresh", 0.3),
        top_k=det_cfg.get("top_k", 10),
        nms_iou=det_cfg.get("nms_iou", 0.5),
        cache=det_cache,
        device=str(device),
    )

    # -- CLIP encoder --
    clip_cfg = cfg["clip"]
    clip_encoder = CLIPEncoder(
        model_name=clip_cfg.get("model", "ViT-B/32"),
        device=str(device),
        cache=clip_cache,
        batch_size=clip_cfg.get("batch_size", 32),
    )

    # -- LLM Prior --
    prior_cfg = cfg.get("prior", {})
    prior_lookup = None
    if prior_cfg.get("enabled", False):
        prior_lookup = PriorLookup(Path(prior_cfg["path"]), pred_vocab)

    # -- Model --
    cfg["relation_head"]["input_dim"] = 512 + 512 + cfg["spatial"]["feature_dim"]
    model = load_model(checkpoint, cfg, num_predicates, device)

    # ------------------------------------------------------------------ #
    # Step 1: Detect
    # ------------------------------------------------------------------ #
    image = Image.open(image_path).convert("RGB")
    image_w, image_h = image.size

    print(f"Image: {image_path}  ({image_w}×{image_h})")
    detections = detector.detect(image)
    print(f"Detections ({len(detections)}): {[d['label'] for d in detections]}")

    if len(detections) < 2:
        print("Fewer than 2 objects detected - no pairs to evaluate.")
        return []

    # ------------------------------------------------------------------ #
    # Step 2: CLIP encode all object crops
    # ------------------------------------------------------------------ #
    boxes = [d["box"] for d in detections]
    embs = clip_encoder.encode_crops(image, boxes)  # (N, 512)

    # ------------------------------------------------------------------ #
    # Step 3: Build pairs and run model
    # ------------------------------------------------------------------ #
    pairs = build_pairs(detections)
    if not pairs:
        return []

    subj_embs, obj_embs, spatials, prior_vecs = [], [], [], []
    for (si, oi) in pairs:
        subj_embs.append(embs[si])
        obj_embs.append(embs[oi])
        sp = compute_spatial_features(
            detections[si]["box"], detections[oi]["box"], image_w, image_h
        )
        spatials.append(sp)
        if prior_lookup is not None:
            prior_vecs.append(prior_lookup.get(detections[si]["label"], detections[oi]["label"]))
        else:
            prior_vecs.append(np.zeros(num_predicates, dtype=np.float32))

    subj_t = torch.from_numpy(np.stack(subj_embs)).float().to(device)
    obj_t  = torch.from_numpy(np.stack(obj_embs)).float().to(device)
    sp_t   = torch.from_numpy(np.stack(spatials)).float().to(device)
    pv_t   = torch.from_numpy(np.stack(prior_vecs)).float().to(device)

    prior_in = pv_t if prior_lookup is not None else None

    probs = model.predict(subj_t, obj_t, sp_t, prior_in)  # (P, num_preds)

    # ------------------------------------------------------------------ #
    # Step 4: Collect and rank triplets
    # ------------------------------------------------------------------ #
    triplets = []
    for k, (si, oi) in enumerate(pairs):
        pred_idx   = int(probs[k].argmax())
        confidence = float(probs[k, pred_idx])
        triplets.append({
            "subject":      detections[si]["label"],
            "predicate":    pred_vocab[pred_idx],
            "object":       detections[oi]["label"],
            "confidence":   round(confidence, 4),
            "subject_box":  detections[si]["box"],
            "object_box":   detections[oi]["box"],
            "subject_score": round(detections[si]["score"], 4),
            "object_score":  round(detections[oi]["score"], 4),
        })

    triplets.sort(key=lambda x: x["confidence"], reverse=True)
    return triplets[:top_k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True, help="Path to input image.")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--output",     default="predictions.json")
    parser.add_argument("--top_k",      type=int, default=20)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    triplets = run_inference(
        image_path=args.image,
        cfg=cfg,
        checkpoint=args.checkpoint,
        top_k=args.top_k,
        device_str=device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, indent=2)

    print(f"\nTop-{len(triplets)} triplets:")
    for t in triplets:
        print(f"  ({t['subject']}, {t['predicate']}, {t['object']})  conf={t['confidence']:.3f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

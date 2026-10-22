"""
app.py  —  IB-VRD Flask demo server
=====================================
Loads Experiment B checkpoint (best R@50 76.73 %, mR@50 34.30 %)
and serves a live-inference web frontend.

Run from open_vocab_vrd/:
    python app.py

Opens at http://localhost:5050
"""

import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

from src.detectors.yoloworld_detector import YOLOWorldDetector
from src.encoders.clip_encoder import CLIPEncoder
from src.encoders.spatial_features import compute_spatial_features
from src.models.vrd_model import load_model
from src.utils.cache import DiskCache
from src.utils.io import load_json, load_yaml
from src.utils.labels import PriorLookup

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT  = ROOT / "outputs/run_20260422_231714/best.pt"   # Exp B best
CONFIG_FILE = ROOT / "configs/experiment_b.yaml"
SAMPLES_DIR = ROOT.parent / "vrd_project"                    # test_*.jpg
PORT        = 5050

app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
CORS(app)


# ── Inference engine (singleton, loaded once at startup) ──────────────────────
class VRDInferenceEngine:
    def __init__(self):
        self.ready = False
        self._load()

    def _load(self):
        print("[VRD] Loading model components …")
        cfg = load_yaml(str(CONFIG_FILE))
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.cfg = cfg

        processed_dir = ROOT / cfg["data"]["processed_dir"]
        self.pred_vocab = load_json(processed_dir / "pred_vocab.json")
        self.num_preds  = len(self.pred_vocab)

        cache_root = ROOT / cfg["data"]["cache_dir"]
        # Disable cache for ad-hoc images (not in VG training set)
        clip_cache = DiskCache(
            cache_root / cfg["cache"].get("clip_subdir", "clip_embeddings"),
            enabled=False,
        )
        det_cache = DiskCache(
            cache_root / cfg["cache"].get("detector_subdir", "detections"),
            enabled=False,
        )

        vocab = load_yaml(
            ROOT / cfg["detector"].get("vocab_file", "configs/predicates.yaml")
        ).get("objects", [])

        print(f"[VRD] Loading YOLO-World detector …")
        self.detector = YOLOWorldDetector(
            model_path=str(ROOT / cfg["detector"]["model_path"]),
            vocab=vocab,
            conf_thresh=cfg["detector"].get("conf_thresh", 0.3),
            top_k=cfg["detector"].get("top_k", 10),
            nms_iou=cfg["detector"].get("nms_iou", 0.5),
            cache=det_cache,
            device=device_str,
        )

        print(f"[VRD] Loading CLIP ViT-B/32 …")
        self.clip_enc = CLIPEncoder(
            model_name=cfg["clip"].get("model", "ViT-B/32"),
            device=device_str,
            cache=clip_cache,
            batch_size=cfg["clip"].get("batch_size", 32),
        )

        prior_cfg = cfg.get("prior", {})
        self.prior_lookup = None
        if prior_cfg.get("enabled", False):
            self.prior_lookup = PriorLookup(ROOT / prior_cfg["path"], self.pred_vocab)

        cfg["relation_head"]["input_dim"] = (
            512 + 512 + cfg["spatial"]["feature_dim"]
        )
        print(f"[VRD] Loading RelationHead + FusionModule from {CHECKPOINT.name} …")
        self.model = load_model(str(CHECKPOINT), cfg, self.num_preds, self.device)
        self.ready = True
        print(f"[VRD] Ready  device={device_str}  predicates={self.num_preds}")

    def infer(self, image_pil: Image.Image, top_k: int = 20) -> dict:
        image_w, image_h = image_pil.size

        # Step 1: detect objects
        detections = self.detector.detect(image_pil)
        if len(detections) < 2:
            return {
                "triplets": [],
                "detections": detections,
                "warning": "Fewer than 2 objects detected.",
            }

        # Step 2: CLIP-encode crops
        boxes = [d["box"] for d in detections]
        embs  = self.clip_enc.encode_crops(image_pil, boxes)  # (N, 512)

        # Step 3: build all ordered pairs
        n = len(detections)
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

        subj_e, obj_e, spatials, priors = [], [], [], []
        for si, oi in pairs:
            subj_e.append(embs[si])
            obj_e.append(embs[oi])
            spatials.append(
                compute_spatial_features(
                    detections[si]["box"], detections[oi]["box"], image_w, image_h
                )
            )
            if self.prior_lookup:
                priors.append(
                    self.prior_lookup.get(
                        detections[si]["label"], detections[oi]["label"]
                    )
                )
            else:
                priors.append(np.zeros(self.num_preds, dtype=np.float32))

        subj_t = torch.from_numpy(np.stack(subj_e)).float().to(self.device)
        obj_t  = torch.from_numpy(np.stack(obj_e)).float().to(self.device)
        sp_t   = torch.from_numpy(np.stack(spatials)).float().to(self.device)
        pv_t   = torch.from_numpy(np.stack(priors)).float().to(self.device)

        probs = self.model.predict(
            subj_t, obj_t, sp_t, pv_t if self.prior_lookup else None
        )  # (P, num_preds)

        # Step 4: rank and collect
        triplets = []
        for k, (si, oi) in enumerate(pairs):
            pred_idx   = int(probs[k].argmax())
            confidence = float(probs[k, pred_idx])
            triplets.append({
                "subject":       detections[si]["label"],
                "predicate":     self.pred_vocab[pred_idx],
                "object":        detections[oi]["label"],
                "confidence":    round(confidence, 4),
                "subject_box":   [round(v, 1) for v in detections[si]["box"]],
                "object_box":    [round(v, 1) for v in detections[oi]["box"]],
                "subject_score": round(float(detections[si]["score"]), 4),
                "object_score":  round(float(detections[oi]["score"]), 4),
            })

        triplets.sort(key=lambda x: x["confidence"], reverse=True)

        # Also return unique detections for overlay
        unique_dets = [
            {
                "label": d["label"],
                "score": round(float(d["score"]), 4),
                "box":   [round(v, 1) for v in d["box"]],
            }
            for d in detections
        ]

        return {
            "triplets":   triplets[:top_k],
            "detections": unique_dets,
        }


# Load once at import time
engine = VRDInferenceEngine()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({
        "status":     "ok",
        "ready":      engine.ready,
        "device":     str(engine.device),
        "predicates": engine.num_preds,
        "checkpoint": CHECKPOINT.name,
    })


@app.post("/api/infer")
def infer():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in request"}), 400

    f   = request.files["image"]
    img = Image.open(f.stream).convert("RGB")
    top_k = int(request.form.get("top_k", 20))

    t0      = time.perf_counter()
    result  = engine.infer(img, top_k=top_k)
    elapsed = round((time.perf_counter() - t0) * 1000)

    result["inference_ms"] = elapsed
    result["image_w"]      = img.width
    result["image_h"]      = img.height
    return jsonify(result)


@app.get("/api/samples")
def samples():
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(
        p.name for p in SAMPLES_DIR.iterdir()
        if p.suffix.lower() in exts and p.name.startswith("test_")
    )
    return jsonify({"samples": imgs})


@app.get("/api/sample/<name>")
def sample_image(name):
    safe = Path(name).name           # prevent path traversal
    path = SAMPLES_DIR / safe
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(str(path))


@app.get("/")
def index():
    return send_file(str(ROOT / "frontend/index.html"))


if __name__ == "__main__":
    print(f"\n  IB-VRD Demo → http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=False)
